#!/usr/bin/env python3
"""
Debug / evaluate the Phase 4 perception core (MOT + team + role), isolated from
ball detection / OCR / framing.

For each frame it draws:
  - player bboxes coloured by team (amber / blue) once the classifier warms up,
  - off-court roles dimmed: bench (grey), referee (yellow), off (dark),
  - the court polygon (if --court given).

Prints per-interval stats: tracks, field/bench/referee counts, team-ready flag,
so you can verify the warm-up converges and roles are stable.

Usage:
  python scripts/debug_player_perception.py \\
    --source videos/volley-2.mp4 --person-model models/yolox_m.onnx \\
    --court configs/court.json --out videos/perception_debug.mp4 --max-frames 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.court import CourtCalibration
from alt_pix.detector import YOLOXDetector, _COCO_PERSON
from alt_pix.log_config import setup_logging
from alt_pix.roles import RoleClassifier
from alt_pix.stream import iter_frames
from alt_pix.team_assign import make_team_assigner
from alt_pix.team_classifier import TeamClassifier
from alt_pix.tracker import PlayerTracker
from alt_pix.visualizer import draw_tracks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise Phase 4 player perception")
    p.add_argument("--source", required=True)
    p.add_argument("--person-model", default="models/yolox_m.onnx")
    p.add_argument("--court", default=None)
    p.add_argument("--out", default="perception_debug.mp4")
    p.add_argument("--conf-person", type=float, default=0.4)
    p.add_argument("--sport", choices=["volleyball", "basketball", "generic"],
                   default="volleyball",
                   help="volleyball=court-half(net); others=colour clustering")
    p.add_argument("--team-backend", choices=["siglip", "lab", "hsv"], default="siglip")
    p.add_argument("--scene-margin", type=float, default=300.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    ort_device = "cuda" if args.device.startswith("cuda") else "cpu"

    person_det = YOLOXDetector(args.person_model, conf_thr=args.conf_person,
                               detect_classes={_COCO_PERSON}, device=ort_device)
    tracker = PlayerTracker()
    court = CourtCalibration.load(args.court) if args.court else None
    team_clf = TeamClassifier(backend=args.team_backend, device=args.device)
    assigner = make_team_assigner(args.sport, court, team_clf)
    role_clf = RoleClassifier(court) if court is not None else None
    print(f"── チーム割当方式: {assigner.method}  (sport={args.sport}) ──")

    writer: cv2.VideoWriter | None = None
    n = 0
    for frame_id, ts, frame in iter_frames(args.source):
        person_dets = person_det.detect(frame)
        if court is not None:
            person_dets = [d for d in person_dets
                           if court.is_on_court(d.bbox, args.scene_margin)]

        tracks = tracker.update(person_dets, frame)

        result = assigner.update(frame, tracks)
        for t in tracks:
            tm = result.team_map.get(t.track_id, -1)
            t.team = tm if tm >= 0 else None
            t.team_reason = result.reason_map.get(t.track_id)
        if role_clf is not None:
            safe = {t.track_id: (t.team if t.team is not None else -1) for t in tracks}
            role_map = role_clf.classify(tracks, safe, result.dist_map)
            for t in tracks:
                t.role = role_map.get(t.track_id)
                t.role_reason = role_clf.last_reasons.get(t.track_id)

        vis = frame.copy()
        draw_tracks(vis, tracks, {})
        if court is not None:
            court.draw(vis)

        # 判定理由（参加度の内訳）を非 field の track に重畳して説明可能に。
        for t in tracks:
            if t.role in (None, "field"):
                continue
            x1, y1, x2, y2 = (int(v) for v in t.bbox)
            reason = t.role_reason or ""
            if reason:
                cv2.putText(vis, reason[:60], (x1, y2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)

        n_field = sum(1 for t in tracks if t.role in (None, "field"))
        n_bench = sum(1 for t in tracks if t.role == "bench")
        n_ref = sum(1 for t in tracks if t.role == "referee")
        n_off = sum(1 for t in tracks if t.role == "off")
        game = ""
        if role_clf is not None:
            part = role_clf.participation
            game = (f" | {'PLAY' if part.game_active else 'PAUSED'} "
                    f"oncourt={part.on_court_count_ema:.1f}")
        cv2.putText(vis,
                    f"f={frame_id} tracks={len(tracks)} field={n_field} "
                    f"bench={n_bench} ref={n_ref} off={n_off} {assigner.method}{game}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     25.0, (w, h))
        writer.write(vis)

        n += 1
        if n % 100 == 0:
            print(f"[f={frame_id:5d}] tracks={len(tracks):2d} field={n_field:2d} "
                  f"bench={n_bench:2d} ref={n_ref:2d} off={n_off:2d} "
                  f"team_ready={assigner.ready}")
        if args.max_frames and n >= args.max_frames:
            break

    if writer:
        writer.release()
    print(f"\nWrote {args.out}. Colours: amber/blue=team0/1, grey=bench, "
          f"yellow=referee, dark=off-court. Verify team colours stay stable on "
          f"each player and referees/bench are not coloured as a team.")


if __name__ == "__main__":
    main()
