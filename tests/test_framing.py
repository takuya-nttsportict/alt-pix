"""Phase 5 framing tests: critically-damped smoothing + state-aware framing.

Drives FramingCalculator directly with synthetic BallState / Track sequences —
no GPU, no models. Asserts the broadcast-smooth design promises
(docs/phase5_framing.md): constant 16:9 aspect, smooth (low-jerk) panning, a
wide hold in NO_PLAY, a wider shot in SERVICE, no hard jump on a one-frame ball
drop, and that the critically-damped spring settles without oscillation.

Run:  python tests/test_framing.py   (or pytest)
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# detector.py imports onnxruntime at load; stub it (never exercised here).
if "onnxruntime" not in sys.modules:
    try:
        import onnxruntime  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from alt_pix.ball_tracker import BallState
from alt_pix.framing import ROI, FramingCalculator, _smooth_damp
import numpy as np
from alt_pix.tracker import Track

_FW, _FH = 1920, 1080


def _ball(x, y, visible=True, predicted=False, conf=0.8) -> BallState:
    return BallState(x=x, y=y, visible=visible, conf=conf, predicted=predicted)


def _track(tid, fx, fy) -> Track:
    return Track(track_id=tid, bbox=(fx - 20, fy - 80, fx + 20, fy), conf=0.9, class_id=0)


def _players(centre_x: float) -> list[Track]:
    return [_track(i, centre_x - 150 + 60 * i, 600) for i in range(6)]


def _roi_center(roi: ROI) -> tuple[float, float]:
    return roi.x + roi.w / 2, roi.y + roi.h / 2


def test_smooth_damp_settles_without_overshoot():
    """The spring converges monotonically to target, never overshooting."""
    cur = np.array([0.0, 0.0])
    vel = np.array([0.0, 0.0])
    target = np.array([100.0, 0.0])
    prev = -1.0
    for _ in range(400):
        cur, vel = _smooth_damp(cur, target, vel, 0.5, 1 / 30)
        assert cur[0] <= 100.0 + 1e-6, "overshoot past target"
        assert cur[0] >= prev - 1e-6, "non-monotonic (oscillation)"
        prev = cur[0]
    assert abs(cur[0] - 100.0) < 1.0, "did not converge"


def test_roi_valid_and_aspect():
    fr = FramingCalculator(_FW, _FH, mode="auto")
    roi = fr.compute(_ball(960, 400), _players(960))
    assert roi.w > 0 and roi.h > 0
    assert 0 <= roi.x <= _FW - roi.w
    assert 0 <= roi.y <= _FH - roi.h
    assert abs(roi.w / roi.h - 16 / 9) < 0.05


def test_aspect_ratio_constant_on_wide_source():
    """16:9 every frame even on an ultra-wide crop source (regression)."""
    fr = FramingCalculator(2160, 650, output_aspect=16 / 9, mode="auto")
    players = _players(1080)
    for state in [_ball(900, 300), _ball(1500, 200),
                  _ball(0, 0, visible=False), _ball(960, 300, predicted=True)] * 15:
        roi = fr.compute(state, players)
        assert abs(roi.w / max(roi.h, 1) - 16 / 9) < 0.05


def test_panning_is_smooth_low_jerk():
    """Following a steadily moving ball produces low frame-to-frame acceleration."""
    fr = FramingCalculator(_FW, _FH, mode="auto")
    centers = []
    for f in range(120):
        bx = 400 + f * 8  # ball pans steadily right
        players = _players(960)  # players anchor centre
        roi = fr.compute(_ball(bx, 400), players)
        centers.append(_roi_center(roi)[0])
    # 2nd difference (acceleration) RMS should be tiny — spring is smooth.
    d2 = [centers[i] - 2 * centers[i - 1] + centers[i - 2] for i in range(2, len(centers))]
    rms = (sum(v * v for v in d2) / len(d2)) ** 0.5
    assert rms < 1.0, f"pan jerk too high: {rms:.3f}px/frame^2"


def test_one_frame_ball_drop_does_not_jump():
    fr = FramingCalculator(_FW, _FH, mode="auto")
    for _ in range(60):
        fr.compute(_ball(900, 400), _players(900))
    c_before = _roi_center(fr.compute(_ball(900, 400), _players(900)))
    c_drop = _roi_center(fr.compute(_ball(0, 0, visible=False), _players(900)))
    jump = abs(c_drop[0] - c_before[0]) + abs(c_drop[1] - c_before[1])
    assert jump < 15.0, f"ball drop jumped {jump:.1f}px"


def test_no_play_holds_wide_and_freezes_pan():
    """NO_PLAY widens the shot and stops chasing the ball."""
    fr = FramingCalculator(_FW, _FH, mode="auto")
    for _ in range(60):
        fr.compute(_ball(1600, 300), _players(1600), game_state="rally")
    roi_rally = fr.compute(_ball(1600, 300), _players(1600), game_state="rally")
    roi_pause = roi_rally
    for _ in range(120):
        roi_pause = fr.compute(_ball(1600, 300), _players(960), game_state="no_play")
    assert roi_pause.w > roi_rally.w, "no_play should widen"
    # Pan should not follow the ball (which sits far right) during no_play.
    c_pause = _roi_center(roi_pause)[0]
    c_rally = _roi_center(roi_rally)[0]
    assert c_pause < c_rally, "no_play should not chase the ball right"


def test_service_is_wider_than_rally():
    """SERVICE pulls to a wider shot than RALLY."""
    fr_r = FramingCalculator(_FW, _FH, mode="auto")
    fr_s = FramingCalculator(_FW, _FH, mode="auto")
    players = _players(960)
    wr = ws = 0
    for _ in range(120):
        wr = fr_r.compute(_ball(960, 400), players, game_state="rally").w
        ws = fr_s.compute(_ball(960, 400), players, game_state="service").w
    assert ws > wr, f"service ({ws}) should be wider than rally ({wr})"


def test_rally_ball_leans_off_player_centroid():
    """In RALLY the ball still pulls the frame off the player centroid."""
    fr = FramingCalculator(_FW, _FH, mode="auto")
    players = _players(500)  # players left
    cx = 0.0
    for _ in range(150):
        cx = _roi_center(fr.compute(_ball(1500, 400), players, game_state="rally"))[0]
    pc = 500.0
    assert cx > pc + 50, "ball did not pull the frame toward itself in rally"


def test_service_focuses_on_server():
    """SERVICE with a server focus point centres on it and zooms tighter."""
    players = _players(960)  # team regrouped centre-court
    server = (300.0, 620.0)  # server stands left, behind the end line
    fr_focus = FramingCalculator(_FW, _FH, mode="auto")
    fr_wide = FramingCalculator(_FW, _FH, mode="auto")
    cx = w_focus = w_wide = 0.0
    for _ in range(150):
        roi_f = fr_focus.compute(_ball(960, 400), players,
                                 game_state="service", focus_xy=server)
        roi_w = fr_wide.compute(_ball(960, 400), players, game_state="service")
        cx, w_focus = _roi_center(roi_f)[0], roi_f.w
        w_wide = roi_w.w
    assert cx < 600, f"camera did not move to the server (cx={cx:.0f})"
    assert w_focus < w_wide, "server focus should be tighter than loose wide"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
