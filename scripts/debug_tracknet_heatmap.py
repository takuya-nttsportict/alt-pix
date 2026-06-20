#!/usr/bin/env python3
"""
Debug / evaluate the TrackNet ball detector by visualising its raw heatmap.

For each frame it writes a side-by-side video:
    [ left ] original frame with the detected peak marked
    [ right ] sigmoid heatmap (channel 2 = current frame), JET-coloured

It also prints per-interval peak statistics so you can judge whether the
model is firing at all and pick a sensible --conf threshold.

Usage:
  python scripts/debug_tracknet_heatmap.py \\
    --source game.mp4 \\
    --model  models/tracknet_volleyball.pt \\
    --out    heatmap_debug.mp4 \\
    --max-frames 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.log_config import setup_logging
from alt_pix.stream import iter_frames
from alt_pix.tracknet import TrackNetDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise TrackNet ball heatmaps")
    p.add_argument("--source", required=True)
    p.add_argument("--model", default="models/tracknet_volleyball.pt")
    p.add_argument("--out", default="heatmap_debug.mp4")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)

    det = TrackNetDetector(args.model, conf_thr=args.conf, device=args.device)

    # Monkey-tap the model output: re-run detect() but also keep the raw heatmap.
    writer: cv2.VideoWriter | None = None
    peaks: list[float] = []
    n_hit = 0
    n_total = 0

    for frame_id, ts, frame in iter_frames(args.source):
        h, w = frame.shape[:2]
        det._ensure_transforms(h, w)
        det._buf.append(det._preprocess(frame))
        if len(det._buf) < 3:
            continue

        stacked = torch.cat(list(det._buf), dim=0).unsqueeze(0).to(det._device)
        with torch.no_grad():
            logits = det._model(stacked)[0]
            hm = torch.sigmoid(logits[0, 2]).cpu().numpy()   # (H, W) in [0,1]

        peak = float(hm.max())
        peaks.append(peak)
        n_total += 1

        # Heatmap → colour, resized to frame size
        hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(cv2.resize(hm_u8, (w, h)), cv2.COLORMAP_JET)

        vis = frame.copy()
        if peak >= args.conf:
            n_hit += 1
            dets = det._postprocess(torch.from_numpy(hm))
            if dets:
                x1, y1, x2, y2 = (int(v) for v in dets[0].bbox)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(vis, (cx, cy), 10, (0, 255, 0), 2)
                cv2.putText(vis, f"{peak:.2f}", (cx + 12, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(vis, f"f={frame_id} peak={peak:.3f} thr={args.conf}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        combined = np.hstack([vis, hm_color])
        if writer is None:
            ch, cw = combined.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     25.0, (cw, ch))
        writer.write(combined)

        if n_total % 100 == 0:
            arr = np.array(peaks[-100:])
            print(f"[f={frame_id:5d}] peak last100: "
                  f"min={arr.min():.3f} mean={arr.mean():.3f} max={arr.max():.3f} "
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
    print("\nIf max peak is near 0 → preprocessing/weights mismatch.")
    print("If peaks are high but scattered → lower/raise --conf and inspect the video.")


if __name__ == "__main__":
    main()
