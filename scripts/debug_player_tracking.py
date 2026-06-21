#!/usr/bin/env python3
"""
Phase 4 評価 ② — 人物検出 + MOT（ID 安定性）

ボール検出を切り離し、人物検出 → ByteTrack の ID 安定性だけを評価する。
GT が無いので MOTA/IDF1 は出さず、ID 安定性のプロキシ指標（track_eval.py）と
ID 色の軌跡描画による目視で評価する。

出力動画（各フレーム）:
  - 各 track を ID 固有色で bbox + [ID] ラベル描画
  - 足元軌跡 trail を ID 色で残す（同じ選手で色が変わったら ID switch 疑い）
  - 上部に track 数・累計 ID 数・平均 lifetime を表示
  - --court 指定時は scene-margin 内のみ追跡（観客を除外）

終了時に track_eval のレポート（lifetime ヒストグラム等）を表示する。

評価観点:
  - active track 数が実人数（コート上 ~12）付近で安定しているか
  - 軌跡の色が選手ごとに継続するか（switch すると色が変わる）
  - short_ratio / new-ID rate が低いか（fragmentation）

パラメータ比較に使える: --track-thresh / --track-buffer / --min-frames を
振って summary 指標の差を見る。

Usage:
  python scripts/debug_player_tracking.py \\
    --source videos/volley-2.mp4 --person-model models/yolox_m.onnx \\
    --court configs/court.json --out videos/tracking_debug.mp4 --max-frames 600
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.detector import YOLOXDetector, _COCO_PERSON
from alt_pix.log_config import setup_logging
from alt_pix.stream import iter_frames
from alt_pix.track_eval import TrackingStats
from alt_pix.tracker import PlayerTracker


def _id_color(tid: int) -> tuple[int, int, int]:
    """ID から決定的に鮮やかな色を作る（HSV を一周させる）。"""
    h = (tid * 47) % 180  # 47 は 180 と互いに素 → 隣接 ID が離れた色になる
    hsv = np.uint8([[[h, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Player detection + MOT stability eval")
    p.add_argument("--source", required=True)
    p.add_argument("--person-model", default="models/yolox_m.onnx")
    p.add_argument("--court", default=None,
                   help="Optional court JSON; restricts tracking to near-court people")
    p.add_argument("--out", default="tracking_debug.mp4")
    p.add_argument("--conf-person", type=float, default=0.4)
    p.add_argument("--scene-margin", type=float, default=300.0)
    p.add_argument("--track-thresh", type=float, default=0.5)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--min-frames", type=int, default=2,
                   help="minimum_consecutive_frames to confirm a track")
    p.add_argument("--trail-len", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--frame-skip", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    ort_device = "cuda" if args.device.startswith("cuda") else "cpu"

    person_det = YOLOXDetector(args.person_model, conf_thr=args.conf_person,
                               detect_classes={_COCO_PERSON}, device=ort_device)
    tracker = PlayerTracker(
        track_activation_threshold=args.track_thresh,
        lost_track_buffer=args.track_buffer,
        minimum_consecutive_frames=args.min_frames,
        fps=args.fps,
    )

    court = None
    if args.court:
        from alt_pix.court import CourtCalibration
        court = CourtCalibration.load(args.court)

    stats = TrackingStats()
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=args.trail_len))
    seen_recent: dict[int, int] = {}  # tid -> last frame seen (trail GC)

    writer: cv2.VideoWriter | None = None
    n = 0

    for frame_id, ts, frame in iter_frames(args.source, skip_frames=args.frame_skip):
        person_dets = person_det.detect(frame)
        if court is not None:
            person_dets = [d for d in person_dets
                           if court.is_on_court(d.bbox, args.scene_margin)]

        tracks = tracker.update(person_dets, frame)
        track_ids = [t.track_id for t in tracks]
        stats.update(frame_id, track_ids)

        vis = frame.copy()
        if court is not None:
            court.draw(vis)

        for t in tracks:
            x1, y1, x2, y2 = (int(v) for v in t.bbox)
            color = _id_color(t.track_id)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"[{t.track_id}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            foot = (int((x1 + x2) / 2), y2)
            trails[t.track_id].append(foot)
            seen_recent[t.track_id] = frame_id

        # 軌跡描画（少し前に消えた ID も trail だけ残す）
        for tid, pts in list(trails.items()):
            if frame_id - seen_recent.get(tid, -10 ** 9) > args.trail_len:
                del trails[tid]
                continue
            color = _id_color(tid)
            for j in range(1, len(pts)):
                cv2.line(vis, pts[j - 1], pts[j], color, 2)

        s = stats.summary()
        cv2.putText(vis,
                    f"f={frame_id} tracks={len(tracks)} uniqueIDs={s['unique_ids']} "
                    f"life_mean={s['lifetime_mean']:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     25.0, (w, h))
        writer.write(vis)

        n += 1
        if n % 100 == 0:
            print(f"[f={frame_id:5d}] tracks={len(tracks):2d} "
                  f"uniqueIDs={s['unique_ids']:3d} life_mean={s['lifetime_mean']:.1f}")
        if args.max_frames and n >= args.max_frames:
            break

    if writer:
        writer.release()

    print()
    print(stats.format_report())
    print(f"\n  出力動画: {args.out}")
    print("  目視: 同一選手の軌跡色が継続するか（色が変われば ID switch）。")
    print("  密集/オクルージョン（ネット際・レシーブ隊形）で色が入れ替わらないか。")


if __name__ == "__main__":
    main()
