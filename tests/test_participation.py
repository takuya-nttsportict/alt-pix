"""Unit tests for the temporal participation tracker (alt_pix/participation.py).

Verifies the core claims of the redesign:
  - a moving on-court player scores high (participating),
  - a stationary off-court person (line judge / bystander) scores low,
  - the SERVE exception is handled for free: a player who briefly steps off
    court keeps a high participation score (high on-court fraction over time).

Numpy-only; onnxruntime stubbed like the other perception tests.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if "onnxruntime" not in sys.modules:
    try:
        import onnxruntime  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from alt_pix.court import CourtCalibration
from alt_pix.participation import ParticipationTracker
from alt_pix.tracker import Track


def _court() -> CourtCalibration:
    return CourtCalibration(corners=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
                            player_margin=80.0)


def _track(tid: int, foot_x: float, foot_y: float) -> Track:
    return Track(track_id=tid, bbox=(foot_x - 10, foot_y - 40, foot_x + 10, foot_y),
                 conf=0.9, class_id=0)


def test_moving_on_court_player_scores_high():
    pt = ParticipationTracker(_court(), alpha=0.2, min_frames=10, motion_ref=0.04)
    st = None
    for f in range(60):
        st = pt.update([_track(1, 300 + (f % 6) * 8, 250)])[1]
    assert st.frames == 60
    assert st.on_court_ema > 0.9
    assert st.score >= 0.5  # participating


def test_stationary_off_court_person_scores_low():
    pt = ParticipationTracker(_court(), alpha=0.2, min_frames=10, motion_ref=0.04)
    st = None
    for f in range(60):
        st = pt.update([_track(9, 600, 800)])[9]  # off-court, never moves
    assert st.on_court_ema < 0.1
    assert st.score < 0.5  # not participating


def test_serve_exception_player_stays_high():
    """A server steps off court briefly but stays a participant over time."""
    pt = ParticipationTracker(_court(), alpha=0.05, min_frames=10, motion_ref=0.04)
    st = None
    for f in range(200):
        # On court and moving most of the time; off-court only ~10% (serving).
        if f % 10 == 0:
            tr = _track(1, 500, 560)   # behind the end line (off court), serving
        else:
            tr = _track(1, 300 + (f % 6) * 8, 250)  # rallying on court
        st = pt.update([tr])[1]
    assert st.on_court_ema > 0.7   # majority on court despite serves
    assert st.score >= 0.5         # still classified as participating


def test_appearance_signal_folds_in():
    pt = ParticipationTracker(_court(), alpha=0.3, min_frames=5)
    # Off-court + stationary, but high appearance match keeps a small contribution.
    st = None
    for f in range(40):
        st = pt.update([_track(5, 600, 800)], appear_match={5: 1.0})[5]
    assert st.appear_ema is not None and st.appear_ema > 0.9
    # Still below threshold (only appearance is positive), but non-zero.
    assert 0.0 < st.score < 0.5


def test_sideline_exit_general_grace():
    """A player stepping off the SIDELINE (v-axis) gets the shorter grace."""
    # Court 0..1000px -> u in [0,18]m, v in [0,9]m. y beyond 500px = past sideline.
    pt = ParticipationTracker(_court(), alpha=0.2, min_frames=10, motion_ref=0.04,
                              grace_frames=30, serve_grace_frames=300)
    for _ in range(20):
        pt.update([_track(1, 500, 250)])          # on court
    for f in range(15):
        st = pt.update([_track(1, 500, 800)])[1]   # off past the sideline (v>9)
    assert st.recently_exited is True
    assert st.assume_server is False
    assert st.frames_since_exit == 15
    # After the (short) general grace, the flag clears.
    for _ in range(40):
        st = pt.update([_track(1, 500, 800)])[1]
    assert st.recently_exited is False


def test_endline_exit_assumes_server_long_grace():
    """A player stepping behind the END line (u-axis) is assumed a server."""
    pt = ParticipationTracker(_court(), alpha=0.2, min_frames=10, motion_ref=0.04,
                              grace_frames=30, serve_grace_frames=300)
    for _ in range(20):
        pt.update([_track(1, 500, 250)])          # on court
    # Exit past the end line: x beyond 1000px -> u > 18m.
    for f in range(50):
        st = pt.update([_track(1, 1200, 250)])[1]
    assert st.assume_server is True
    # Still in grace at 50 frames (>general grace of 30) thanks to serve window.
    assert st.recently_exited is True
    assert st.frames_since_exit == 50


def test_never_on_court_not_recently_exited():
    """A person who was never inside the court has recently_exited=False."""
    pt = ParticipationTracker(_court(), alpha=0.2, min_frames=10)
    st = None
    for _ in range(40):
        st = pt.update([_track(9, 1200, 250)])[9]   # never on court
    assert st.recently_exited is False
    assert st.frames_since_exit == 999


def test_game_active_pauses_when_court_empty():
    """game_active is False once the court holds fewer than min_active players."""
    pt = ParticipationTracker(_court(), min_frames=10, min_active_on_court=4.0,
                              active_alpha=0.3)
    # Busy court: 6 players inside -> active.
    for _ in range(30):
        pt.update([_track(i, 200 + 80 * i, 250) for i in range(6)])
    assert pt.game_active is True
    assert pt.on_court_count_ema > 4.0
    # Court empties (timeout): nobody inside.
    for _ in range(30):
        pt.update([_track(99, 1200, 250)])   # single off-court bystander
    assert pt.game_active is False


def test_off_court_zone_endline_sideline_corner():
    """Court metres zones: end-line=player, sideline=referee, corner=ambiguous."""
    c = _court()  # x:0..1000 -> u:0..18, y:0..500 -> v:0..9
    # On court.
    assert c.off_court_zone((490, 210, 510, 250)) == "on"
    # Past the right END line (u>18), between sidelines -> serving zone.
    assert c.off_court_zone((1190, 210, 1210, 250)) == "endline"
    # Past the bottom SIDE line (v>9), between end lines -> referee zone.
    assert c.off_court_zone((490, 760, 510, 800)) == "sideline"
    # Past BOTH (corner) -> ambiguous (line judge).
    assert c.off_court_zone((1190, 760, 1210, 800)) == "corner"


def test_off_zone_recorded_on_state():
    pt = ParticipationTracker(_court(), min_frames=5)
    st = pt.update([_track(1, 1200, 250)])[1]   # behind the right end line
    assert st.off_zone == "endline"
    st = pt.update([_track(2, 500, 800)])[2]     # past the sideline
    assert st.off_zone == "sideline"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
