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


def test_recently_exited_court_flag():
    """A player who steps off court should have recently_exited=True for grace_frames."""
    pt = ParticipationTracker(_court(), alpha=0.2, min_frames=10, motion_ref=0.04,
                              grace_frames=30)
    # Put player on court for a while.
    for _ in range(20):
        pt.update([_track(1, 300, 250)])
    # Step off court.
    for f in range(15):
        states = pt.update([_track(1, 300, 800)])
    st = states[1]
    assert st.recently_exited is True
    assert st.frames_since_exit == 15

    # After grace_frames, recently_exited resets.
    for _ in range(20):
        pt.update([_track(1, 300, 800)])
    st = pt.update([_track(1, 300, 800)])[1]
    assert st.recently_exited is False


def test_never_on_court_not_recently_exited():
    """A person who was never inside the court has recently_exited=False."""
    pt = ParticipationTracker(_court(), alpha=0.2, min_frames=10)
    st = None
    for _ in range(40):
        st = pt.update([_track(9, 600, 800)])[9]
    assert st.recently_exited is False
    assert st.frames_since_exit == 999


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
