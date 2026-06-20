#!/usr/bin/env python3
"""
Analyze TrackNet detection rate at different confidence thresholds.

Scans frames and reports what fraction would be detected at conf=0.1..0.9,
so you can choose the right threshold trade-off.

Usage:
  python scripts/analyze_detection_rate.py \\
    --source videos/volley-2.mp4 --model models/tracknet_volleyball.pt \\
    --max-frames 600
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.stream import iter_frames
from alt_pix.tracknet import (
    TrackNetDetector,
    _W, _H, _compute_tiles, _preprocess_tile, _detect_blobs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze detection rate vs confidence threshold")
    p.add_argument("--source", required=True)
    p.add_argument("--model", default="models/tracknet_volleyball.pt")
    p.add_argument("--tile-overlap", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=600)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model directly (bypass TrackNetDetector to get raw peaks)
    from alt_pix.tracknet import TrackNetV2
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model = TrackNetV2(9, 3)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    tile_xs: list[int] | None = None
    tile_w: int | None = None
    tile_bufs: list[deque] = []

    # Per-frame best peak (across all tiles)
    peaks: list[float] = []
    frame_ids: list[int] = []   # frame_id aligned with peaks
    # Per-tile peak distributions
    n_total = 0

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for frame_id, ts, frame in iter_frames(args.source):
        h, w = frame.shape[:2]

        if tile_xs is None:
            tile_xs = _compute_tiles(w, h, args.tile_overlap)
            tile_w = min(int(round(h * _W / _H)), w)
            tile_bufs = [deque(maxlen=3) for _ in tile_xs]
            print(f"Tiles: {len(tile_xs)}  tile_w={tile_w}  xs={tile_xs}")

        for i, x0 in enumerate(tile_xs):
            tile_bufs[i].append(_preprocess_tile(frame, x0, tile_w))

        n_total += 1
        if len(tile_bufs[0]) < 3:
            continue

        best_peak = 0.0
        for i, x0 in enumerate(tile_xs):
            stacked = torch.cat(list(tile_bufs[i]), dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                hm = torch.sigmoid(model(stacked)[0][0, 2]).cpu().numpy()
            best_peak = max(best_peak, float(hm.max()))

        peaks.append(best_peak)
        frame_ids.append(frame_id)

        if n_total % 100 == 0:
            so_far = np.array(peaks)
            print(f"[f={frame_id}] frames={len(peaks)}  "
                  f"mean_peak={so_far.mean():.3f}  "
                  f"det@0.5={100*(so_far>0.5).mean():.1f}%")

        if args.max_frames and n_total >= args.max_frames:
            break

    peaks_arr = np.array(peaks)
    print(f"\n── Detection rate analysis ({len(peaks)} frames, {len(tile_xs)} tiles) ──")
    print(f"  peak stats: min={peaks_arr.min():.3f}  mean={peaks_arr.mean():.3f}  "
          f"max={peaks_arr.max():.3f}  median={np.median(peaks_arr):.3f}")
    print()
    print("  conf_thr  detect_rate  frames_detected")
    for thr in thresholds:
        rate = (peaks_arr > thr).mean() * 100
        n = int((peaks_arr > thr).sum())
        bar = "█" * int(rate / 2)
        print(f"  {thr:.1f}       {rate:5.1f}%       {n:4d}    {bar}")

    print("\n  Peak histogram (best tile per frame):")
    hist, edges = np.histogram(peaks_arr, bins=10, range=(0, 1))
    for i, cnt in enumerate(hist):
        lo, hi = edges[i], edges[i + 1]
        bar = "█" * min(cnt, 40)
        print(f"  {lo:.1f}-{hi:.1f}: {cnt:4d}  {bar}")

    # ── Bimodality check ──────────────────────────────────────────────────────
    # If detections are confident-or-nothing, the right threshold is the valley
    # between the two modes, NOT whatever hits an arbitrary target rate.
    low_mass = int((peaks_arr < 0.2).sum())     # "blind" mode
    high_mass = int((peaks_arr > 0.6).sum())    # "confident" mode
    mid_mass = len(peaks_arr) - low_mass - high_mass
    bimodal = mid_mass < 0.5 * min(low_mass, high_mass)
    print(f"\n  Mode masses: blind(<0.2)={low_mass}  mid(0.2-0.6)={mid_mass}  "
          f"confident(>0.6)={high_mass}")

    if bimodal:
        print("  → BIMODAL: model is confident-or-blind. Keep --conf in the valley")
        print("    (~0.5). Lowering it only adds low-confidence false positives.")
        print("    The real lever is the blind frames, not the threshold.")
    else:
        target_rate = 40.0
        best_thr = None
        for thr in thresholds:
            if (peaks_arr > thr).mean() * 100 >= target_rate:
                best_thr = thr
        if best_thr is not None:
            print(f"  Recommended --conf {best_thr} for ≥{target_rate:.0f}% detection rate")
        else:
            print(f"  WARNING: can't reach {target_rate:.0f}% detection even at conf=0.1")

    # ── Gap-length analysis at conf=0.5 (can the Kalman bridge the misses?) ────
    thr = 0.5
    detected = peaks_arr > thr
    gaps: list[tuple[int, int]] = []   # (start_index, length) into peaks_arr
    run = 0
    for i, d in enumerate(detected):
        if not d:
            run += 1
        elif run > 0:
            gaps.append((i - run, run))
            run = 0
    if run > 0:
        gaps.append((len(detected) - run, run))

    print(f"\n── Miss-gap analysis at conf={thr} ──")
    if gaps:
        lengths = np.array([g[1] for g in gaps])
        bridgeable = int((lengths <= 5).sum())   # MAX_MISS_FRAMES = 5
        print(f"  {len(gaps)} gaps  total_missed={int(lengths.sum())}  "
              f"longest={lengths.max()}  median={int(np.median(lengths))}")
        print(f"  gaps ≤5 frames (Kalman-bridgeable): {bridgeable}/{len(gaps)} "
              f"({100*bridgeable/len(gaps):.0f}%)")
        gh, _ = np.histogram(lengths, bins=[1, 2, 4, 6, 11, 21, 10**6])
        labels = ["1", "2-3", "4-5", "6-10", "11-20", "21+"]
        for lab, cnt in zip(labels, gh):
            bar = "█" * min(int(cnt), 40)
            print(f"  gap {lab:6s}: {cnt:4d}  {bar}")

        # Show WHERE the long gaps are, so they can be checked in the video.
        long_gaps = sorted([g for g in gaps if g[1] >= 11],
                           key=lambda g: g[1], reverse=True)
        if long_gaps:
            print(f"\n  Long gaps (≥11 frames) — scrub the video to these frames:")
            for start_idx, length in long_gaps:
                f0 = frame_ids[start_idx]
                f1 = frame_ids[min(start_idx + length - 1, len(frame_ids) - 1)]
                print(f"    frames {f0:5d}–{f1:5d}  ({length} frames missed)")
            print("  If these sit at the clip start / between rallies → out of")
            print("  play (expected). If mid-rally → genuine model failure.")
        print("\n  Interpretation:")
        print("  - Many short gaps (≤5) → Kalman interpolation fills them → good.")
        print("  - Few long gaps (>10) → ball genuinely out of play (timeout, dead")
        print("    ball) OR model failure; check the frame ranges above.")
    else:
        print("  No gaps — ball detected every frame.")


if __name__ == "__main__":
    main()
