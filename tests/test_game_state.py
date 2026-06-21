"""Phase 5 game-state estimator tests (no GPU).

Verifies RALLY / SERVICE / NO_PLAY estimation and the hysteresis that keeps the
state from flickering. Uses a tiny fake court + participation so no models load.
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

from alt_pix.ball_tracker import BallState
from alt_pix.game_state import GameStateEstimator


def _ball(x, y, visible=True) -> BallState:
    return BallState(x=x, y=y, visible=visible, conf=0.8, predicted=False)


class _FakeCourt:
    """Maps image x linearly to court u in [0,18]; y ignored."""
    def image_to_court(self, x, y):
        return x / 100.0, 4.5  # x=0 -> u=0, x=1800 -> u=18


class _FakePart:
    def __init__(self, active=True, ema=6.0):
        self._active = active
        self._ema = ema

    @property
    def game_active(self):
        return self._active

    @property
    def on_court_count_ema(self):
        return self._ema


def test_rally_when_ball_midcourt():
    est = GameStateEstimator(_FakeCourt(), _FakePart(active=True))
    for _ in range(20):
        st = est.update(_ball(900, 300), [])  # u=9 (mid)
    assert st == "rally", st


def test_service_when_ball_at_endline():
    est = GameStateEstimator(_FakeCourt(), _FakePart(active=True))
    st = "rally"
    for _ in range(20):
        st = est.update(_ball(50, 300), [])  # u=0.5 -> near endline
    assert st == "service", st


def test_no_play_when_participation_inactive():
    est = GameStateEstimator(_FakeCourt(), _FakePart(active=False, ema=1.0))
    # NO_PLAY transition is immediate (participation EMA already smoothed).
    st = est.update(_ball(900, 300), [])
    assert st == "no_play", st


def test_hysteresis_blocks_single_frame_service_flicker():
    """One stray endline frame must NOT flip RALLY->SERVICE (needs sustained)."""
    est = GameStateEstimator(_FakeCourt(), _FakePart(active=True))
    for _ in range(20):
        est.update(_ball(900, 300), [])  # settle in rally
    st = est.update(_ball(50, 300), [])  # single endline frame
    assert st == "rally", "single frame should not switch to service"


def test_ball_lost_keeps_rally_while_active():
    est = GameStateEstimator(_FakeCourt(), _FakePart(active=True))
    for _ in range(10):
        est.update(_ball(900, 300), [])
    st = est.update(_ball(0, 0, visible=False), [])
    assert st == "rally", "brief ball loss should keep rally"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
