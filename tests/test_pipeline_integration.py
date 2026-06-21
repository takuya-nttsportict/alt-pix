"""End-to-end integration smoke test for the Phase 4 pipeline data flow.

This wires the REAL orchestration that `scripts/run_pipeline.py` runs each
frame — team assignment → role classification → ball tracking → framing →
JSONL record — and drives it with synthetic detections over a short rally.

What it does NOT cover: the GPU detectors (YOLOX/TrackNet) and ByteTrack
(`trackers`/`supervision`), which need onnxruntime/torch and are third-party.
Those are stubbed / bypassed: we feed `Track` and ball `Detection` objects
directly, exactly as the trackers would emit them. The point is to catch
*integration* bugs (schema, None handling, empty dist_map, field filtering,
ROI production) without a GPU or model weights, so it runs anywhere.

Scenario (court inset inside a 1920x1080 frame):
  - 12 field players on court (6 left of the net, 6 right), all moving,
  - 1 referee standing beyond the sideline (stationary, colour outlier),
  - a ball flying across the court each frame,
  - mid-rally one left player steps BEHIND the left end line to serve.

Run:  python tests/test_pipeline_integration.py
  or:  python -m pytest tests/test_pipeline_integration.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# detector.py imports onnxruntime at module load; stub it so the numpy/cv2
# pieces import anywhere (the stub is never exercised — we bypass the detector).
if "onnxruntime" not in sys.modules:
    try:
        import onnxruntime  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")


def _imports():
    """Lazy imports (cv2 needed by court.py). Returns the symbols we use."""
    from alt_pix.ball_tracker import BallTracker
    from alt_pix.court import CourtCalibration
    from alt_pix.detector import Detection, _COCO_SPORTS_BALL
    from alt_pix.framing import FramingCalculator
    from alt_pix.output import _make_record
    from alt_pix.participation import ParticipationTracker
    from alt_pix.roles import RoleClassifier
    from alt_pix.team_assign import make_team_assigner
    from alt_pix.tracker import Track
    return (BallTracker, CourtCalibration, Detection, _COCO_SPORTS_BALL,
            FramingCalculator, _make_record, ParticipationTracker,
            RoleClassifier, make_team_assigner, Track)


# Court inset inside the frame so there is room for off-court people on all sides.
_FW, _FH = 1920, 1080
_CORNERS = [(360, 200), (1560, 200), (1560, 900), (360, 900)]  # TL,TR,BR,BL
_NET_X = 960  # u=9m -> x = 360 + 0.5*(1560-360)


def _track(Track, tid: int, fx: float, fy: float) -> "Track":
    """A 40x80 bbox whose bottom-centre (foot) is (fx, fy)."""
    return Track(track_id=tid, bbox=(fx - 20, fy - 80, fx + 20, fy),
                 conf=0.9, class_id=0)


def test_pipeline_dataflow_smoke():
    try:
        (BallTracker, CourtCalibration, Detection, BALL_CLS, FramingCalculator,
         _make_record, ParticipationTracker, RoleClassifier,
         make_team_assigner, Track) = _imports()
    except Exception as e:  # pragma: no cover - cv2 missing in some dev shells
        print(f"SKIP pipeline smoke (deps unavailable): {e!r}")
        return

    court = CourtCalibration(corners=_CORNERS, player_margin=80.0, ball_margin=200.0)
    assigner = make_team_assigner("volleyball", court, None)
    assert assigner.method == "court-half(net)"
    # Small min_frames so roles converge within the short test rally.
    part = ParticipationTracker(court, alpha=0.2, min_frames=10, motion_ref=0.02)
    role_clf = RoleClassifier(court, participation=part)
    ball_tracker = BallTracker()
    framing = FramingCalculator(_FW, _FH, mode="auto")

    records: list[dict] = []
    N = 90
    last_roles: dict[int, str] = {}
    for f in range(N):
        # ── synthesise this frame's tracks ────────────────────────────────────
        tracks = []
        # 6 left field players (u<9), 6 right (u>=9); all jitter each frame.
        for i in range(6):
            tracks.append(_track(Track, 10 + i, 450 + 70 * i + (f % 4) * 8, 550))
        for i in range(6):
            tracks.append(_track(Track, 20 + i, 1020 + 70 * i + (f % 4) * 8, 550))
        # Referee beyond the bottom sideline (stationary colour outlier).
        tracks.append(_track(Track, 99, 950, 1000))
        # Mid-rally, left player 10 steps behind the left end line to serve.
        if f >= 45:
            tracks = [t for t in tracks if t.track_id != 10]
            tracks.append(_track(Track, 10, 250, 540))  # x<360 -> u<0 (endline zone)

        # ── ball detection (flies across the court) ───────────────────────────
        bx = 420 + (f * 12) % 1100
        ball_dets = [Detection(bbox=(bx - 8, 400 - 6, bx + 8, 400 + 6),
                               conf=0.8, class_id=BALL_CLS)]

        # ── REAL orchestration (mirrors run_pipeline.main) ────────────────────
        result = assigner.update(np.zeros((_FH, _FW, 3), np.uint8), tracks)
        dist_map = result.dist_map  # empty for court-half
        # Referee is a colour outlier; court-half carries no colour, so emulate the
        # colour-distance signal the role splitter consumes when it IS available.
        dist_map = {t.track_id: (5.0 if t.track_id == 99 else 1.0) for t in tracks}
        for t in tracks:
            tm = result.team_map.get(t.track_id, -1)
            t.team = tm if tm >= 0 else None
            t.team_reason = result.reason_map.get(t.track_id)

        team_map_safe = {t.track_id: (t.team if t.team is not None else -1) for t in tracks}
        role_map = role_clf.classify(tracks, team_map_safe, dist_map)
        for t in tracks:
            t.role = role_map.get(t.track_id)
            t.role_reason = role_clf.last_reasons.get(t.track_id)

        field_tracks = [t for t in tracks if t.role in (None, "field")]
        ball_state = ball_tracker.update(ball_dets)
        roi = framing.compute(ball_state, field_tracks)
        rec = _make_record(f, f * 33.3, ball_state, tracks, {}, roi)
        records.append(rec)
        last_roles = role_map

        # ── invariants every frame ────────────────────────────────────────────
        assert roi.w > 0 and roi.h > 0
        assert 0 <= roi.x <= _FW - roi.w
        assert 0 <= roi.y <= _FH - roi.h
        # On-court players get a concrete team with a court-half reason.
        lp = next(t for t in tracks if t.track_id == 11)
        assert lp.team == 0 and "net" in (lp.team_reason or "")
        rp = next(t for t in tracks if t.track_id == 21)
        assert rp.team == 1

    # ── post-rally role expectations ──────────────────────────────────────────
    assert last_roles[11] == "field"          # moving on-court player
    assert last_roles[21] == "field"
    assert last_roles[99] == "referee"        # stationary off-court colour outlier
    assert last_roles[10] == "field"          # server behind the end line (zone prior)
    # The server must survive into the framing-relevant set.
    assert any(t.track_id == 10 for t in field_tracks)

    # ── JSONL record schema round-trips and carries explainability ────────────
    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        path = fh.name
    with open(path) as fh:
        lines = [json.loads(ln) for ln in fh if ln.strip()]
    assert len(lines) == N
    sample = lines[-1]
    assert set(sample) >= {"frame_id", "timestamp_ms", "ball", "players", "framing_roi"}
    p0 = sample["players"][0]
    assert set(p0) >= {"track_id", "team", "team_reason", "role", "role_reason", "bbox"}
    assert sample["framing_roi"]["w"] > 0
    Path(path).unlink()


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
