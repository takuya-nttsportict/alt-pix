#!/usr/bin/env python3
"""
Phase 4 評価 ③ — チーム分類（SigLIP → 2-means）

人物検出 → MOT → TeamClassifier のチーム割当だけを評価する。役割フィルタ
（field/bench/referee）は別評価（④）に切り出し、ここは「2 チームに割れているか」
「同一選手のチームが一貫しているか」に集中する。

GT が無いので team_eval.py のプロキシ指標で評価する:
  - team purity   : track ごとの最頻ラベル一致率（一貫性）
  - flip rate     : ラベル変化 /100 フレーム
  - balance       : team0/team1 の人数比（6:6 が基本）
  - margin        : 割当の確信度（二峰で高い側に山が理想）

出力動画:
  - team0=amber / team1=blue で bbox を着色（warm-up 中は白）
  - [ID t0/t1 m=margin] ラベル
  - 上部に team0/team1 人数・team_ready

評価観点:
  - warm-up 後、同じ選手が同じ色を保つか（色がチカチカ＝flip 多発）
  - 両チームがそれぞれの色でまとまるか（片色に偏れば誤分類）
  - 審判・線審が無理に team 色を付けられていないか（→ ④ role で除外する対象）

パラメータ比較: --team-backend siglip/hsv, --warmup, --refit を振って指標差を見る。

Usage:
  python scripts/debug_team_classification.py \\
    --source videos/volley-2_courtcrop_2.mp4 --person-model models/yolox_m.onnx \\
    --court configs/court_crop_2.json --out videos/team_debug.mp4 --max-frames 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.detector import YOLOXDetector, _COCO_PERSON
from alt_pix.log_config import setup_logging
from alt_pix.stream import iter_frames
from alt_pix.team_classifier import TeamClassifier
from alt_pix.team_eval import TeamStats
from alt_pix.tracker import PlayerTracker

_TEAM_COLOR = {0: (0, 200, 255), 1: (255, 128, 0)}  # amber / blue (BGR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Team classification eval")
    p.add_argument("--source", required=True)
    p.add_argument("--person-model", default="models/yolox_m.onnx")
    p.add_argument("--court", default=None,
                   help="Optional court JSON; restricts to near-court people")
    p.add_argument("--out", default="team_debug.mp4")
    p.add_argument("--conf-person", type=float, default=0.4)
    p.add_argument("--team-backend", choices=["siglip", "lab", "hsv"], default="siglip")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--refit", type=int, default=150)
    p.add_argument("--scene-margin", type=float, default=300.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--fps", type=float, default=25.0)
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
    team_clf = TeamClassifier(backend=args.team_backend, warmup_frames=args.warmup,
                              refit_every=args.refit, device=args.device)

    court = None
    if args.court:
        from alt_pix.court import CourtCalibration
        court = CourtCalibration.load(args.court)

    stats = TeamStats()
    writer: cv2.VideoWriter | None = None
    n = 0

    for frame_id, ts, frame in iter_frames(args.source):
        person_dets = person_det.detect(frame)
        if court is not None:
            person_dets = [d for d in person_dets
                           if court.is_on_court(d.bbox, args.scene_margin)]

        tracks = tracker.update(person_dets, frame)
        team_map, _dist_map = team_clf.update(frame, tracks)
        margin_map = team_clf.last_margins
        stats.update(team_map, margin_map)

        vis = frame.copy()
        if court is not None:
            court.draw(vis)

        n0 = n1 = 0
        for t in tracks:
            x1, y1, x2, y2 = (int(v) for v in t.bbox)
            team = team_map.get(t.track_id, -1)
            if team == 0:
                n0 += 1
            elif team == 1:
                n1 += 1
            color = _TEAM_COLOR.get(team, (255, 255, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            m = margin_map.get(t.track_id)
            tlabel = f"t{team}" if team >= 0 else "t?"
            mlabel = f" m{m:.2f}" if m is not None else ""
            label = f"[{t.track_id} {tlabel}{mlabel}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(vis,
                    f"f={frame_id} tracks={len(tracks)} t0={n0} t1={n1} "
                    f"ready={team_clf.ready}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (w, h))
        writer.write(vis)

        n += 1
        if n % 100 == 0:
            s = stats.summary()
            print(f"[f={frame_id:5d}] t0={n0:2d} t1={n1:2d} ready={team_clf.ready} "
                  f"purity_w={s['purity_weighted']:.3f} "
                  f"flip/100={s['flip_rate_per100']:.2f}")
        if args.max_frames and n >= args.max_frames:
            break

    if writer:
        writer.release()

    print()
    print(stats.format_report())
    print(f"\n  出力動画: {args.out}")
    print("  目視: warm-up 後に同一選手の色が継続するか、両チームが別色でまとまるか。")
    print("  審判/線審が team 色になっていても OK（④ role 評価で除外する）。")


if __name__ == "__main__":
    main()
