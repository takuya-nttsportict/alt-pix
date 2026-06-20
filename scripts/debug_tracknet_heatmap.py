#!/usr/bin/env python3
"""
Debug / evaluate TrackNet ball detection AND tracking on the heatmap level,
isolated from the rest of the pipeline (no person detector / OCR / framing).

For each frame it writes a stacked video:
  [ top ]    original frame with:
               - all raw candidates (yellow dots)
               - the tracked ball (green circle)
               - predicted position when the ball is Kalman-interpolated (cyan)
  [ bottom ] best-tile heatmap (JET), placed at its tile offset

Prints per-interval stats: raw detection rate vs. tracked-visible rate, so
you can see how much the motion-aware tracker recovers.

Usage:
  python scripts/debug_tracknet_heatmap.py \\
    --source videos/volley-2.mp4 --model models/tracknet_volleyball.pt \\
    --out videos/heatmap_debug.mp4 --max-frames 600
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.ball_tracker import BallTracker
from alt_pix.log_config import setup_logging
from alt_pix.stream import iter_frames
from alt_pix.tracknet import (
    TrackNetDetector,
    _W, _H, _compute_tiles, _preprocess_tile,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise TrackNet ball detection + tracking")
    p.add_argument("--source", required=True)
    p.add_argument("--model", default="models/tracknet_volleyball.pt")
    p.add_argument("--out", default="heatmap_debug.mp4")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--tile-overlap", type=float, default=0.3)
    p.add_argument("--max-disp", type=float, default=300.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--no-track", action="store_true",
                   help="Show raw per-frame best candidate only (no tracker)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    det = TrackNetDetector(args.model, conf_thr=args.conf,
                           tile_overlap=args.tile_overlap, device=args.device)
    tracker = BallTracker(smooth=True, max_disp=args.max_disp)
    device = det._device

    writer: cv2.VideoWriter | None = None
    n_total = 0
    n_raw_hit = 0       # frames with >=1 raw candidate
    n_track_vis = 0     # frames the tracker reports the ball visible
    trail: deque[tuple[int, int]] = deque(maxlen=25)
    peak_hist = np.zeros(11, dtype=int)  # peak distribution in 0.0..1.0 buckets

    # Mirror the detector's tile layout for heatmap visualisation
    tile_xs: list[int] | None = None
    tile_w: int | None = None
    tile_bufs: list[deque] = []

    for frame_id, ts, frame in iter_frames(args.source):
        h, w = frame.shape[:2]

        # ── Raw candidates via the real detector API ─────────────────────────
        candidates = det.detect(frame)
        if candidates:
            n_raw_hit += 1

        # ── Tracking ─────────────────────────────────────────────────────────
        if args.no_track:
            best = max(candidates, key=lambda d: d.conf) if candidates else None
            ball_xy = None
            predicted = False
            conf = 0.0
            if best is not None:
                x1, y1, x2, y2 = best.bbox
                ball_xy = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                conf = best.conf
        else:
            state = tracker.update(candidates)
            ball_xy = (int(state.x), int(state.y)) if state.visible else None
            predicted = state.predicted
            conf = state.conf

        n_total += 1
        if ball_xy is not None:
            if not predicted:
                n_track_vis += 1
            trail.append(ball_xy)

        # ── Heatmap panel (recompute best tile heatmap for display) ──────────
        if tile_xs is None:
            tile_xs = _compute_tiles(w, h, args.tile_overlap)
            tile_w  = min(int(round(h * _W / _H)), w)
            tile_bufs = [deque(maxlen=3) for _ in tile_xs]
        for i, x0 in enumerate(tile_xs):
            tile_bufs[i].append(_preprocess_tile(frame, x0, tile_w))

        best_hm = None
        best_x0 = 0
        best_peak = 0.0
        if len(tile_bufs[0]) >= 3:
            for i, x0 in enumerate(tile_xs):
                stacked = torch.cat(list(tile_bufs[i]), dim=0).unsqueeze(0).to(device)
                with torch.no_grad():
                    hm = torch.sigmoid(det._model(stacked)[0][0, 2]).cpu().numpy()
                if hm.max() > best_peak:
                    best_peak = float(hm.max())
                    best_hm = hm
                    best_x0 = x0

        hm_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        # Track peak histogram
        peak_hist[min(int(best_peak * 10), 10)] += 1

        if best_hm is not None:
            hm_u8 = (np.clip(best_hm, 0, 1) * 255).astype(np.uint8)
            hm_color = cv2.applyColorMap(cv2.resize(hm_u8, (tile_w, h)), cv2.COLORMAP_JET)
            hm_canvas[:, best_x0: best_x0 + tile_w] = hm_color

        # ── Draw overlays ────────────────────────────────────────────────────
        vis = frame.copy()
        # raw candidates: yellow dots
        for d in candidates:
            x1, y1, x2, y2 = d.bbox
            cv2.circle(vis, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 5, (0, 255, 255), -1)
        # trail
        for j in range(1, len(trail)):
            cv2.line(vis, trail[j - 1], trail[j], (0, 200, 0), 2)
        # tracked ball
        if ball_xy is not None:
            color = (255, 255, 0) if predicted else (0, 255, 0)  # cyan=predicted, green=detected
            cv2.circle(vis, ball_xy, 12, color, 2)
            cv2.putText(vis, f"{conf:.2f}{' P' if predicted else ''}",
                        (ball_xy[0] + 14, ball_xy[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        raw_rate = n_raw_hit / n_total * 100
        trk_rate = n_track_vis / n_total * 100
        cv2.putText(vis,
                    f"f={frame_id} cands={len(candidates)} peak={best_peak:.2f} "
                    f"raw={raw_rate:.0f}% track={trk_rate:.0f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        combined = np.vstack([vis, hm_canvas])
        if writer is None:
            ch, cw = combined.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (cw, ch))
        writer.write(combined)

        if n_total % 100 == 0:
            print(f"[f={frame_id:5d}] raw_detect={raw_rate:5.1f}%  "
                  f"tracked_visible={trk_rate:5.1f}%")

        if args.max_frames and n_total >= args.max_frames:
            break

    if writer:
        writer.release()

    print("\n── TrackNet detection + tracking summary ─────")
    print(f"  frames              : {n_total}")
    print(f"  raw detection rate  : {n_raw_hit / max(n_total,1)*100:.1f}%  "
          f"(>=1 candidate above conf={args.conf})")
    print(f"  tracked visible rate: {n_track_vis / max(n_total,1)*100:.1f}%  "
          f"(detected, excludes Kalman-predicted)")
    print(f"  tracker             : {'OFF (--no-track)' if args.no_track else f'ON (max_disp={args.max_disp})'}")
    print(f"  output video        : {args.out}")
    print("\n── heatmap peak distribution (best tile per frame) ──")
    buckets = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(11)]
    buckets[-1] = "1.0"
    for i, cnt in enumerate(peak_hist):
        bar = "█" * min(cnt, 40)
        print(f"  {buckets[i]:9s}: {cnt:4d} {bar}")
    print("\n  Inspect the video: green=detected ball, cyan=predicted (Kalman),")
    print("  yellow dots=raw candidates. A good track stays on the ball and the")
    print("  green marker should NOT jump to stray yellow dots when the ball is lost.")
    print(f"\n  Tip: if peak distribution is mostly in 0.3-0.5 bucket,")
    print(f"  try --conf 0.3 to increase detection rate (at cost of more false positives).")


if __name__ == "__main__":
    main()
