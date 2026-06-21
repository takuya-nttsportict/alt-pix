"""Phase 5 framing tests: smoothness, ball-confidence blend, game-pause hold.

Drives FramingCalculator directly with synthetic BallState / Track sequences —
no GPU, no models. Asserts the *behaviours* the broadcast-smooth design promises
(docs/phase5_framing.md): bounded per-frame motion, no hard ball/wide jump on a
one-frame ball drop, dead-zone jitter rejection, and a wide hold when the game
is paused.

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
from alt_pix.framing import ROI, FramingCalculator
from alt_pix.tracker import Track

_FW, _FH = 1920, 1080


def _ball(x, y, visible=True, predicted=False, conf=0.8) -> BallState:
    return BallState(x=x, y=y, visible=visible, conf=conf, predicted=predicted)


def _track(tid, fx, fy) -> Track:
    return Track(track_id=tid, bbox=(fx - 20, fy - 80, fx + 20, fy), conf=0.9, class_id=0)


def _players(centre_x: float) -> list[Track]:
    """6 players spread around centre_x at mid-height."""
    return [_track(i, centre_x - 150 + 60 * i, 600) for i in range(6)]


def _roi_center(roi: ROI) -> tuple[float, float]:
    return roi.x + roi.w / 2, roi.y + roi.h / 2


def test_roi_valid_and_aspect():
    fr = FramingCalculator(_FW, _FH, mode="auto")
    roi = fr.compute(_ball(960, 400), _players(960))
    assert roi.w > 0 and roi.h > 0
    assert 0 <= roi.x <= _FW - roi.w
    assert 0 <= roi.y <= _FH - roi.h
    # aspect ratio honoured (16:9 within rounding).
    assert abs(roi.w / roi.h - 16 / 9) < 0.05


def test_pan_is_rate_limited():
    """A teleporting ball must not snap the camera; per-frame pan is bounded."""
    fr = FramingCalculator(_FW, _FH, mode="auto")
    fr.compute(_ball(300, 540), _players(300))  # settle left
    c0 = _roi_center(fr.compute(_ball(300, 540), _players(300)))
    # Ball + players jump hard to the right edge.
    c1 = _roi_center(fr.compute(_ball(1700, 540), _players(1700)))
    moved = abs(c1[0] - c0[0])
    # max_pan_frac default 0.04 -> <= ~77px per frame (allow rounding slack).
    assert moved <= 0.04 * _FW + 2, f"pan {moved:.1f}px exceeded rate limit"


def test_one_frame_ball_drop_does_not_jump():
    """A single lost-ball frame should barely move the camera (no wide snap)."""
    fr = FramingCalculator(_FW, _FH, mode="auto")
    for _ in range(40):  # converge on a tracked ball
        fr.compute(_ball(900, 400), _players(900))
    c_before = _roi_center(fr.compute(_ball(900, 400), _players(900)))
    # One frame with the ball lost (players unchanged).
    c_drop = _roi_center(fr.compute(_ball(0, 0, visible=False), _players(900)))
    jump = abs(c_drop[0] - c_before[0]) + abs(c_drop[1] - c_before[1])
    assert jump <= 0.04 * _FW + 2, f"ball drop jumped {jump:.1f}px"


def test_deadzone_rejects_micro_jitter():
    """Tiny ball wobble inside the dead-zone leaves the camera centre fixed."""
    fr = FramingCalculator(_FW, _FH, mode="auto")
    for _ in range(60):
        fr.compute(_ball(960, 540), _players(960))
    c0 = _roi_center(fr.compute(_ball(960, 540), _players(960)))
    # Jitter the ball by a few px (< deadzone 0.03*1920 = 57.6px) for many frames.
    for k in range(20):
        c = _roi_center(fr.compute(_ball(960 + (k % 2) * 10, 540), _players(960)))
    assert abs(c[0] - c0[0]) < 5.0, "camera drifted on sub-deadzone jitter"


def test_predicted_ball_pulls_toward_players():
    """Predicted (interpolated) ball weights players more than a detected ball."""
    fr_vis = FramingCalculator(_FW, _FH, mode="auto")
    fr_pred = FramingCalculator(_FW, _FH, mode="auto")
    # Ball far right, players far left: detected ball should sit further right
    # than the same ball flagged predicted (which leans toward the players).
    players = _players(400)
    cx_vis = cx_pred = 0.0
    for _ in range(30):
        cx_vis = _roi_center(fr_vis.compute(_ball(1500, 540), players))[0]
        cx_pred = _roi_center(fr_pred.compute(_ball(1500, 540, predicted=True), players))[0]
    assert cx_vis > cx_pred + 10, "predicted ball did not lean toward players"


def test_game_pause_holds_wide_not_ball():
    """When paused, the camera ignores the ball and holds a wide player shot."""
    fr = FramingCalculator(_FW, _FH, mode="auto")
    # Active: converge tight on a right-side ball.
    for _ in range(40):
        fr.compute(_ball(1600, 300), _players(1600), game_active=True)
    roi_active = fr.compute(_ball(1600, 300), _players(1600), game_active=True)
    # Now paused with players regrouped centre; ball still off right.
    roi_pause = roi_active
    for _ in range(80):
        roi_pause = fr.compute(_ball(1600, 300), _players(960), game_active=False)
    # Paused shot is wider (bigger ROI) and centred on players, not the ball.
    assert roi_pause.w > roi_active.w, "paused shot should widen"
    assert _roi_center(roi_pause)[0] < _roi_center(roi_active)[0], "should leave the ball"


def test_zoom_is_rate_limited():
    """Size cannot change faster than max_zoom_rate per frame."""
    fr = FramingCalculator(_FW, _FH, mode="auto")
    roi_prev = fr.compute(_ball(960, 540), _players(960))
    # Force a big size target swing: ball lost -> wide players, tight players.
    roi_now = fr.compute(_ball(0, 0, visible=False), _players(960))
    rate = abs(roi_now.w - roi_prev.w) / max(roi_prev.w, 1)
    assert rate <= 0.03 + 0.02, f"zoom rate {rate:.3f} exceeded cap"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
