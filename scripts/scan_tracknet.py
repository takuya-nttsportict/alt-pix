#!/usr/bin/env python3
"""
Scan a clip across MANY frames and ALL tiles to find the global maximum
TrackNet response. Decides definitively whether the model is alive.

Rationale: a single timestamp may simply contain no ball. By sweeping a
few hundred frames over overlapping full-height 16:9 tiles, a working model
MUST produce at least one high peak when the ball is visible. If the global
maximum stays low everywhere, the weights/architecture are the problem,
not the resolution.

Usage:
  python scripts/scan_tracknet.py \\
    --source videos/volley-2.mp4 --model models/tracknet_volleyball.pt \\
    --max-frames 1500
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

from alt_pix.stream import iter_frames
from alt_pix import tracknet as T

_MEAN, _STD = T._MEAN, T._STD
_W, _H = T._W, T._H


def _to_tensor(crop_bgr: np.ndarray, imagenet: bool) -> torch.Tensor:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rs = cv2.resize(rgb, (_W, _H), interpolation=cv2.INTER_LINEAR)
    t = rs.astype(np.float32) / 255.0
    if imagenet:
        t = (t - _MEAN) / _STD
    return torch.from_numpy(t).permute(2, 0, 1).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="models/tracknet_volleyball.pt")
    ap.add_argument("--max-frames", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-imagenet", action="store_true",
                    help="Use /255 only (skip ImageNet normalisation)")
    args = ap.parse_args()
    imagenet = not args.no_imagenet

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model = T.TrackNetV2(9, 3)
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)

    # Per-tile 3-frame buffers
    bufs: dict[int, deque] = {}
    tiles: list[tuple[int, int]] | None = None

    g_peak = 0.0
    g_info = None
    hist = np.zeros(11)   # histogram of peaks in 0.0..1.0 buckets

    n = 0
    for fid, _, frame in iter_frames(args.source):
        h, w = frame.shape[:2]
        if tiles is None:
            tw = min(int(round(h * _W / _H)), w)
            nt = max(1, int(np.ceil((w - tw) / (tw * 0.7))) + 1)
            xs = [0] if nt == 1 else [int(round(i * (w - tw) / (nt - 1))) for i in range(nt)]
            tiles = [(x, tw) for x in xs]
            for i in range(len(tiles)):
                bufs[i] = deque(maxlen=3)
            print(f"tiles: {[(x, x+tw) for x, tw in tiles]}  (each {tw}x{h})")

        frame_peak = 0.0
        for i, (x0, tw) in enumerate(tiles):
            bufs[i].append(_to_tensor(frame[:, x0:x0 + tw], imagenet))
            if len(bufs[i]) < 3:
                continue
            stack = torch.cat(list(bufs[i]), dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                sg = torch.sigmoid(model(stack)[0][0])     # (3,H,W)
            peak = float(sg.max())
            frame_peak = max(frame_peak, peak)
            if peak > g_peak:
                ch = int(sg.amax(dim=(1, 2)).argmax())
                yx = np.unravel_index(int(sg[ch].argmax().cpu()), (_H, _W))
                g_peak = peak
                g_info = (fid, i, x0, ch, yx)

        if frame_peak > 0:
            hist[min(int(frame_peak * 10), 10)] += 1

        n += 1
        if n % 200 == 0:
            print(f"[{n} frames] global_max={g_peak:.3f}")
        if args.max_frames and n >= args.max_frames:
            break

    print("\n── scan summary ──────────────────────────────")
    print(f"  frames scanned : {n}")
    print(f"  imagenet norm  : {imagenet}")
    print(f"  GLOBAL MAX PEAK: {g_peak:.3f}")
    if g_info:
        fid, ti, x0, ch, yx = g_info
        # map heatmap (512x288) coords back into the tile, then full frame
        tw = tiles[ti][1]
        fx = x0 + yx[1] * tw / _W
        fy = yx[0] * h / _H
        print(f"  at frame={fid} tile={ti}(x0={x0}) ch={ch} "
              f"heat_xy={yx[::-1]} → frame_xy=({fx:.0f},{fy:.0f})")
    print(f"  peak histogram (0.0..1.0): {hist.astype(int).tolist()}")
    print("\nVerdict:")
    print("  GLOBAL MAX > 0.5  → model is ALIVE; tiling solves it.")
    print("  GLOBAL MAX < 0.1  → model is DEAD regardless of input;")
    print("                      the checkpoint/architecture is wrong.")


if __name__ == "__main__":
    main()
