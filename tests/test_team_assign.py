"""Unit tests for sport-selectable team assignment (alt_pix/team_assign.py).

Focuses on the volleyball CourtHalfAssigner (deterministic geometry) and the
factory's sport routing. Numpy-only; onnxruntime is stubbed like the other
perception tests so this runs without the GPU image.

Run:  python tests/test_team_assign.py
  or: python -m pytest tests/test_team_assign.py
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
from alt_pix.team_assign import (
    CourtHalfAssigner,
    ColorClusterAssigner,
    make_team_assigner,
)
from alt_pix.team_classifier import TeamClassifier
from alt_pix.tracker import Track


def _rect_court() -> CourtCalibration:
    """Image rectangle 180x90 mapping linearly to 18m x 9m (image px / 10 = metres)."""
    return CourtCalibration(corners=[(0, 0), (180, 0), (180, 90), (0, 90)])


def _track(tid: int, foot_x: float, foot_y: float) -> Track:
    # 20x20 bbox whose bottom-centre is (foot_x, foot_y).
    return Track(track_id=tid, bbox=(foot_x - 10, foot_y - 20, foot_x + 10, foot_y),
                 conf=0.9, class_id=0)


def test_court_half_splits_by_net():
    """u<9m → team0(L), u>=9m → team1(R). Net at 9m (image x=90)."""
    court = _rect_court()
    a = CourtHalfAssigner(court)
    assert a.ready is True
    tracks = [_track(1, 40, 50), _track(2, 140, 50)]  # u=4m, u=14m
    res = a.update(None, tracks)
    assert res.team_map[1] == 0
    assert res.team_map[2] == 1
    # reasons are human-readable and mention the net.
    assert "net" in res.reason_map[1]
    assert "side L" in res.reason_map[1]
    assert "side R" in res.reason_map[2]


def test_court_half_margin_low_near_net():
    """A player right at the net has low confidence; one deep in court is ~1."""
    court = _rect_court()
    a = CourtHalfAssigner(court, ambiguous_band_m=0.5)
    near_net = _track(1, 92, 50)   # u=9.2m → 0.2m from net < band → margin<1
    deep = _track(2, 20, 50)       # u=2m → far → margin 1.0
    res = a.update(None, [near_net, deep])
    assert res.margin_map[1] < 1.0
    assert res.margin_map[2] == 1.0


def test_factory_volleyball_uses_court_half():
    court = _rect_court()
    a = make_team_assigner("volleyball", court, None)
    assert isinstance(a, CourtHalfAssigner)
    assert "court-half" in a.method


def test_factory_basketball_uses_colour():
    clf = TeamClassifier(backend="lab")  # no torch needed
    a = make_team_assigner("basketball", None, clf)
    assert isinstance(a, ColorClusterAssigner)
    assert "color-cluster" in a.method


def test_factory_volleyball_without_court_falls_back_to_colour():
    clf = TeamClassifier(backend="lab")
    a = make_team_assigner("volleyball", None, clf)
    assert isinstance(a, ColorClusterAssigner)


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
