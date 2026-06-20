#!/usr/bin/env python3
"""
Debug / evaluate the TrackNet ball detector by visualising its heatmaps.

For each frame it shows the tile with the best peak and writes a side-by-side
video: [ left ] original frame with detection marked, [ right ] heatmap JET.
Also prints per-interval peak statistics.

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

from alt_pix.log_config import setup_logging
from alt_pix.stream import iter_frames
from alt_pix.tracknet import (
    TrackNetDetector, TrackNetV2,
    _MEAN, _STD, _W, _H,
    _preprocess_tile, _compute_tiles,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise TrackNet ball heatmaps")
    p.add_argument("--source", required=True)
    p.add_argument("--model", default="models/tracknet_volleyball.pt")
    p.add_argument("--out", default="heatmap_debug.mp4")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--tile-overlap", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    det = TrackNetDetector(args.model, conf_thr=args.conf,
                           tile_overlap=args.tile_overlap, device=args.device)

    writer: cv2.VideoWriter | None = None
    peaks: list[float] = []
    n_hit = 0
    n_total = 0

    # Per-tile buffers for raw heatmap access (mirrors TrackNetDetector internals)
    tile_xs: list[int] | None = None
    tile_w: int | None = None
    tile_bufs: list[deque] = []
    device = det._device

    for frame_id, ts, frame in iter_frames(args.source):
        h, w = frame.shape[:2]

        # Initialise tile layout
        if tile_xs is None:
            tile_xs = _compute_tiles(w, h, args.tile_overlap)
            tile_w  = min(int(round(h * _W / _H)), w)
            tile_bufs = [deque(maxlen=3) for _ in tile_xs]

        for i, x0 in enumerate(tile_xs):
            tile_bufs[i].append(_preprocess_tile(frame, x0, tile_w))

        if len(tile_bufs[0]) < 3:
            continue

        # Run all tiles, find best peak and its heatmap
        best_peak = 0.0
        best_hm: np.ndarray | None = None
        best_x0 = 0
        ball_xy: tuple[int, int] | None = None

        for i, x0 in enumerate(tile_xs):
            stacked = torch.cat(list(tile_bufs[i]), dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                logits  = det._model(stacked)[0]
                heatmap = torch.sigmoid(logits[0, 2])
            hm   = heatmap.cpu().numpy()
            peak = float(hm.max())
            if peak > best_peak:
                best_peak = peak
                best_hm   = hm
                best_x0   = x0
                if peak >= args.conf:
                    ym, xm = np.unravel_index(hm.argmax(), hm.shape)
                    cx = int(x0 + xm * tile_w / _W)
                    cy = int(ym * h / _H)
                    ball_xy = (cx, cy)

        peaks.append(best_peak)
        n_total += 1

        # Visualise
        hm_u8     = (np.clip(best_hm, 0, 1) * 255).astype(np.uint8)
        hm_full   = cv2.resize(hm_u8, (tile_w, h), interpolation=cv2.INTER_LINEAR)
        hm_color  = cv2.applyColorMap(hm_full, cv2.COLORMAP_JET)

        # Pad heatmap panel to full frame width
        hm_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        hm_canvas[:, best_x0: best_x0 + tile_w] = hm_color

        vis = frame.copy()
        if ball_xy is not None:
            n_hit += 1
            cv2.circle(vis, ball_xy, 12, (0, 255, 0), 2)
            cv2.putText(vis, f"{best_peak:.2f}", (ball_xy[0] + 14, ball_xy[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(vis, f"f={frame_id} peak={best_peak:.3f} thr={args.conf}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(hm_canvas,
                    f"tile x0={best_x0} tw={tile_w}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        combined = np.vstack([vis, hm_canvas])
        if writer is None:
            ch, cw = combined.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     25.0, (cw, ch))
        writer.write(combined)

        if n_total % 100 == 0:
            arr = np.array(peaks[-100:])
            print(f"[f={frame_id:5d}] peak last100: "
                  f"min={arr.min():.3f} mean={arr.mean():.3f} max={arr.max():.3f}  "
                  f"hit_rate={n_hit / n_total * 100:.1f}%")

        if args.max_frames and n_total >= args.max_frames:
            break

    if writer:
        writer.release()

    arr = np.array(peaks) if peaks else np.array([0.0])
    print("\n── TrackNet heatmap summary ──────────────────")
    print(f"  frames           : {n_total}")
    print(f"  peak  min/mean/max: {arr.min():.3f} / {arr.mean():.3f} / {arr.max():.3f}")
    print(f"  detections (>{args.conf}): {n_hit}  ({n_hit / max(n_total, 1) * 100:.1f}%)")
    print(f"  output video     : {args.out}")


if __name__ == "__main__":
    main()
