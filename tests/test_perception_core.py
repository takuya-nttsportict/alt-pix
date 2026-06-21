"""Unit tests for the Phase 4 perception core logic that needs no heavy deps.

Covers the numpy-only pieces (2-means, role classification, court geometry) so
team/role behaviour is reproducible without torch/transformers/onnxruntime.

Run:  python -m pytest tests/test_perception_core.py
  or: python tests/test_perception_core.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# alt_pix.tracker → detector imports onnxruntime, which is only present in the
# GPU runtime image. Stub it so the numpy-only perception core stays testable
# anywhere (the stub is never exercised by these tests).
if "onnxruntime" not in sys.modules:
    try:
        import onnxruntime  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from alt_pix.team_classifier import TeamClassifier, _kmeans2
from alt_pix.tracker import Track


def _track(tid: int, foot_x: float, foot_y: float) -> Track:
    """A 20x40 bbox whose bottom-centre is (foot_x, foot_y)."""
    return Track(track_id=tid,
                 bbox=(foot_x - 10, foot_y - 40, foot_x + 10, foot_y),
                 conf=0.9, class_id=0)


# ── _kmeans2 ────────────────────────────────────────────────────────────────────

def test_kmeans2_separates_two_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0], 0.1, size=(50, 2))
    b = rng.normal([10, 10], 0.1, size=(50, 2))
    X = np.vstack([a, b])
    cents = _kmeans2(X)
    # The two centroids should land near (0,0) and (10,10) in some order.
    far = np.linalg.norm(cents[0] - cents[1])
    assert far > 9.0, f"centroids not separated: {cents}"


def test_kmeans2_degenerate_single_point():
    cents = _kmeans2(np.array([[1.0, 2.0]]))
    assert cents.shape == (2, 2)
    assert np.allclose(cents[0], cents[1])


# ── TeamClassifier predict / outlier distance (stateless) ───────────────────────

def test_team_predict_and_outlier_distance():
    clf = TeamClassifier(backend="hsv")
    # Two well-separated team embeddings.
    team_a = np.tile([0.0, 0.0], (10, 1)) + np.random.default_rng(1).normal(0, 0.01, (10, 2))
    team_b = np.tile([5.0, 5.0], (10, 1)) + np.random.default_rng(2).normal(0, 0.01, (10, 2))
    clf.fit(np.vstack([team_a, team_b]))
    assert clf.ready

    labels = clf.predict(np.array([[0.0, 0.0], [5.0, 5.0]]))
    assert labels[0] != labels[1]  # different teams

    # A referee-coloured point sits far from both centroids.
    d = clf.distance_to_teams(np.array([[2.5, 2.5], [0.0, 0.0]]))
    assert d[0] > d[1]  # midpoint outlier is farther than a team member


def test_team_predict_before_fit_returns_unassigned():
    clf = TeamClassifier(backend="hsv")
    labels = clf.predict(np.array([[1.0, 1.0]]))
    assert labels.tolist() == [-1]


# ── RoleClassifier (temporal participation + colour outlier) ────────────────────

def _moving_track(tid: int, foot_x: float, foot_y: float) -> Track:
    return _track(tid, foot_x, foot_y)


def test_role_classifier_field_bench_referee():
    # Lazy import: court.py needs cv2 which is available in the runtime image.
    try:
        from alt_pix.court import CourtCalibration
        from alt_pix.participation import ParticipationTracker
        from alt_pix.roles import RoleClassifier
    except Exception as e:  # pragma: no cover - cv2 missing in some dev shells
        print(f"SKIP role test (cv2 unavailable): {e!r}")
        return

    court = CourtCalibration(corners=[(0, 0), (1000, 0), (1000, 500), (0, 500)],
                             player_margin=80.0)
    # Short min_frames so the test converges quickly; fast EMA.
    part = ParticipationTracker(court, alpha=0.2, min_frames=20, motion_ref=0.04)
    rc = RoleClassifier(court, participation=part)

    # Accumulate evidence over a window. Players (on-court + moving) become
    # field; a stationary off-court colour outlier becomes referee; a stationary
    # off-court team-coloured person becomes bench.
    roles = {}
    for f in range(60):
        # 6 field players inside the court, all moving each frame.
        field = [_moving_track(10 + i, 200 + 50 * i + (f % 5) * 6, 250) for i in range(6)]
        ref = _track(3, 600, 800)     # off-court, stationary
        bench = _track(2, 100, 800)   # off-court, stationary
        tracks = field + [ref, bench]

        team_map = {t.track_id: 0 for t in tracks}
        dist_map = {t.track_id: 1.0 for t in tracks}
        dist_map[ref.track_id] = 100.0  # colour outlier
        roles = rc.classify(tracks, team_map, dist_map)

    assert roles[10] == "field"          # a moving on-court player
    assert roles[3] == "referee"          # stationary off-court colour outlier
    assert roles[2] == "bench"            # stationary off-court team colour


def test_role_classifier_defers_until_enough_frames():
    try:
        from alt_pix.court import CourtCalibration
        from alt_pix.participation import ParticipationTracker
        from alt_pix.roles import RoleClassifier
    except Exception as e:  # pragma: no cover
        print(f"SKIP defer test (cv2 unavailable): {e!r}")
        return
    court = CourtCalibration(corners=[(0, 0), (1000, 0), (1000, 500), (0, 500)])
    rc = RoleClassifier(court, participation=ParticipationTracker(court, min_frames=30))
    tracks = [_track(1, 500, 250)]
    roles = rc.classify(tracks, {1: 0}, {})
    assert roles[1] == "off"  # not enough history yet


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
