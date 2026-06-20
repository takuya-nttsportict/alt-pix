#!/usr/bin/env python3
"""
Numerical diagnosis for the TrackNet ball detector.

Answers three questions with hard numbers instead of guesswork:
  1. What is the source resolution / aspect ratio?  (Is the ball being
     downscaled to sub-pixel size when squeezed into 512x288?)
  2. Are the pretrained weights actually doing anything? Prints the RAW
     logit min/mean/max for ALL 3 output channels (not just channel 2).
  3. Which preprocessing wakes the model? Re-runs the SAME 3 frames under
     several preprocessing variants and reports the resulting peak.

Usage:
  python scripts/diagnose_tracknet.py \\
    --source videos/volley-2.mp4 --model models/tracknet_volleyball.pt \\
    --at-frame 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.stream import iter_frames
from alt_pix import tracknet as T

_MEAN = T._MEAN
_STD = T._STD
_W, _H = T._W, T._H


def _collect_frames(source: str, at_frame: int) -> list[np.ndarray]:
    """Return the 3 consecutive BGR frames ending at `at_frame`."""
    buf: list[np.ndarray] = []
    for fid, _, frame in iter_frames(source):
        buf.append(frame)
        if len(buf) > 3:
            buf.pop(0)
        if fid >= at_frame and len(buf) == 3:
            return buf
    return buf


def _forward(model, stack9: torch.Tensor, device) -> torch.Tensor:
    with torch.no_grad():
        logits = model(stack9.unsqueeze(0).to(device))[0][0]   # (3, H, W)
    return logits.cpu()


def _stack(frames_chw: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(frames_chw, dim=0)   # (9, H, W)


def _prep_affine_imagenet(frame, trans):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    w = cv2.warpAffine(rgb, trans, (_W, _H), flags=cv2.INTER_LINEAR)
    t = w.astype(np.float32) / 255.0
    t = (t - _MEAN) / _STD
    return torch.from_numpy(t).permute(2, 0, 1).contiguous()


def _prep_affine_div255(frame, trans):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    w = cv2.warpAffine(rgb, trans, (_W, _H), flags=cv2.INTER_LINEAR)
    t = w.astype(np.float32) / 255.0
    return torch.from_numpy(t).permute(2, 0, 1).contiguous()


def _prep_resize_imagenet(frame, _trans):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    w = cv2.resize(rgb, (_W, _H), interpolation=cv2.INTER_LINEAR)
    t = w.astype(np.float32) / 255.0
    t = (t - _MEAN) / _STD
    return torch.from_numpy(t).permute(2, 0, 1).contiguous()


def _prep_resize_div255(frame, _trans):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    w = cv2.resize(rgb, (_W, _H), interpolation=cv2.INTER_LINEAR)
    t = w.astype(np.float32) / 255.0
    return torch.from_numpy(t).permute(2, 0, 1).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="models/tracknet_volleyball.pt")
    ap.add_argument("--at-frame", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Checkpoint inspection ─────────────────────────────────────────────────
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    print("── checkpoint ────────────────────────────────")
    if isinstance(ck, dict):
        print(f"  top-level keys: {list(ck.keys())[:8]}")
    sd = ck.get("model_state_dict", ck.get("model", ck.get("state_dict", ck))) \
        if isinstance(ck, dict) else ck
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    print(f"  n params tensors: {len(sd)}")
    print(f"  inc.double_conv.0.weight shape: {tuple(sd['inc.double_conv.0.weight'].shape)}")
    print(f"  outc.conv.weight shape:        {tuple(sd['outc.conv.weight'].shape)}")

    model = T.TrackNetV2(9, 3)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  load missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"    MISSING (first 5): {missing[:5]}")
    if unexpected:
        print(f"    UNEXPECTED (first 5): {unexpected[:5]}")
    model.eval().to(device)

    # ── Frames ────────────────────────────────────────────────────────────────
    frames = _collect_frames(args.source, args.at_frame)
    if len(frames) < 3:
        print("ERROR: could not collect 3 frames")
        return
    h, w = frames[0].shape[:2]
    print("\n── source ────────────────────────────────────")
    print(f"  resolution: {w}x{h}   aspect={w/h:.3f}  (16:9={16/9:.3f})")
    scale_x = _W / w
    scale_y = _H / h
    print(f"  plain-resize scale: x={scale_x:.4f} y={scale_y:.4f}")
    print(f"  a 12px ball would become ~{12*min(scale_x, scale_y):.1f}px after resize")

    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    s = float(max(h, w))
    trans = T._get_affine_transform(center, s, (_W, _H), inv=0)
    iso = _W / max(h, w)
    print(f"  affine isotropic scale: {iso:.4f}  "
          f"(content fills {w*iso:.0f}x{h*iso:.0f} of {_W}x{_H})")
    print(f"  a 12px ball would become ~{12*iso:.1f}px after affine")

    # ── Per-channel raw logits under the CURRENT preprocessing ────────────────
    stack = _stack([_prep_affine_imagenet(f, trans) for f in frames])
    logits = _forward(model, stack, device)
    print("\n── raw logits per output channel (affine+imagenet) ──")
    for c in range(logits.shape[0]):
        lc = logits[c]
        sg = torch.sigmoid(lc)
        print(f"  ch{c}: logit[min={lc.min():.2f} mean={lc.mean():.2f} max={lc.max():.2f}]  "
              f"sigmoid_max={sg.max():.3f}")

    # ── Preprocessing sweep (peak sigmoid on channel 2) ───────────────────────
    print("\n── preprocessing sweep (max sigmoid over all channels) ──")
    variants = [
        ("affine + imagenet (current)", _prep_affine_imagenet),
        ("affine + /255 only",          _prep_affine_div255),
        ("plain resize + imagenet",     _prep_resize_imagenet),
        ("plain resize + /255 only",    _prep_resize_div255),
    ]
    for name, fn in variants:
        st = _stack([fn(f, trans) for f in frames])
        lg = _forward(model, st, device)
        sg = torch.sigmoid(lg)
        print(f"  {name:32s}: max_sigmoid={sg.max():.3f}  "
              f"argmax_ch={int(sg.amax(dim=(1,2)).argmax())}")

    # ── Tiling test: run on near-native-resolution 16:9 tiles ─────────────────
    # Full height, width = h*16/9, overlapping horizontally to cover the frame.
    tile_w = int(round(h * _W / _H))          # 16:9 tile at native height
    tile_w = min(tile_w, w)
    n_tiles = max(1, int(np.ceil((w - tile_w) / (tile_w * 0.7))) + 1)
    if n_tiles == 1:
        xs = [0]
    else:
        xs = [int(round(i * (w - tile_w) / (n_tiles - 1))) for i in range(n_tiles)]
    print(f"\n── tiling test ({n_tiles} tiles, each {tile_w}x{h} → {_W}x{_H}) ──")
    print(f"  tile resize scale: {_W/tile_w:.3f}  "
          f"(12px ball → ~{12*_W/tile_w:.1f}px)")
    best_overall = 0.0
    for xi, x0 in enumerate(xs):
        crops = [f[:, x0:x0 + tile_w] for f in frames]
        chw = []
        for c in crops:
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            rs = cv2.resize(rgb, (_W, _H), interpolation=cv2.INTER_LINEAR)
            t = rs.astype(np.float32) / 255.0
            t = (t - _MEAN) / _STD
            chw.append(torch.from_numpy(t).permute(2, 0, 1).contiguous())
        lg = _forward(model, _stack(chw), device)
        sg = torch.sigmoid(lg)
        peak = float(sg.max())
        best_overall = max(best_overall, peak)
        ch = int(sg.amax(dim=(1, 2)).argmax())
        print(f"  tile {xi} x=[{x0:4d}:{x0+tile_w:4d}]: max_sigmoid={peak:.3f}  argmax_ch={ch}")
    print(f"  >>> best tile peak = {best_overall:.3f}")

    print("\nInterpretation:")
    print("  - If a TILE peak is high (>0.5) while full-frame stays ~0.007 →")
    print("    CONFIRMED: resolution was the problem; adopt tiling in tracknet.py.")
    print("  - If tiles are still dead → revisit preprocessing/weights.")


if __name__ == "__main__":
    main()
