#!/usr/bin/env python3
"""Render the virtual-camera (framed) output video from a pipeline JSONL log.

Phase 5 evaluation aid: instead of judging framing from bbox / ROI overlays,
this actually CROPS each source frame to its `framing_roi` and resizes to a
fixed broadcast output size — so you watch the would-be broadcast feed directly.

It reuses the ROI coordinates already in out.jsonl (no GPU, no models, no
re-run of detection). Frame i of the source is matched to the record with
frame_id == i. Frames without a ROI (or missing records) fall back to a
letterbox-free full-frame resize.

Usage:
  python scripts/render_framed.py \\
    --source videos/volley-2_courtcrop_2.mp4 \\
    --jsonl  out_phase5.jsonl \\
    --out    framed_phase5.mp4 \\
    --size   1280x720

  # side-by-side with the source (for A/B review)
  python scripts/render_framed.py --source ... --jsonl ... --out ... --compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.stream import iter_frames


def _load_rois(path: str) -> dict[int, dict]:
    """{frame_id: framing_roi dict} from a pipeline JSONL log."""
    rois: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            roi = rec.get("framing_roi")
            if roi:
                rois[rec["frame_id"]] = roi
    return rois


def _parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _crop_resize(frame: np.ndarray, roi: dict, out_size: tuple[int, int]) -> np.ndarray:
    """Crop frame to ROI then fit-resize to out_size preserving aspect ratio.

    The ROI should already have the target aspect ratio (enforced in framing.py),
    so the letterbox padding should be zero in practice. Fit-resize is used
    defensively so any rounding in the stored integer ROI never causes stretching.
    """
    fh, fw = frame.shape[:2]
    x = max(0, min(int(roi["x"]), fw - 1))
    y = max(0, min(int(roi["y"]), fh - 1))
    w = max(1, min(int(roi["w"]), fw - x))
    h = max(1, min(int(roi["h"]), fh - y))
    crop = frame[y:y + h, x:x + w]
    ow, oh = out_size
    ch, cw = crop.shape[:2]
    # Scale uniformly; if ROI is exactly the right aspect the canvas fill is 0px.
    scale = min(ow / cw, oh / ch)
    rw, rh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((oh, ow, 3), np.uint8)
    ox, oy = (ow - rw) // 2, (oh - rh) // 2
    canvas[oy:oy + rh, ox:ox + rw] = resized
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description="Render framed (cropped) output video")
    p.add_argument("--source", required=True, help="Source video (same one fed to the pipeline)")
    p.add_argument("--jsonl", required=True, help="Pipeline JSONL log with framing_roi")
    p.add_argument("--out", required=True, help="Output mp4 path")
    p.add_argument("--size", default="1280x720", help="Output WxH (default 1280x720)")
    p.add_argument("--fps", type=float, default=30.0, help="Output fps (default 30)")
    p.add_argument("--compare", action="store_true",
                   help="Stack the framed output above a full-frame reference")
    p.add_argument("--max-frames", type=int, default=0, help="Limit frames (0 = all)")
    args = p.parse_args()

    out_size = _parse_size(args.size)
    rois = _load_rois(args.jsonl)
    if not rois:
        print(f"No framing_roi found in {args.jsonl}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(rois)} ROIs from {args.jsonl}")

    writer: cv2.VideoWriter | None = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    n_written = n_fallback = 0

    for frame_idx, _ts, frame in iter_frames(args.source):
        if args.max_frames and frame_idx >= args.max_frames:
            break

        roi = rois.get(frame_idx)
        if roi is not None:
            framed = _crop_resize(frame, roi, out_size)
        else:
            framed = cv2.resize(frame, out_size, interpolation=cv2.INTER_LINEAR)
            n_fallback += 1

        if args.compare:
            ref = cv2.resize(frame, out_size, interpolation=cv2.INTER_LINEAR)
            cv2.putText(framed, "FRAMED", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
            cv2.putText(ref, "SOURCE", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (200, 200, 200), 2)
            out_frame = np.vstack([framed, ref])
        else:
            out_frame = framed

        if writer is None:
            h, w = out_frame.shape[:2]
            writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
            print(f"Writing {w}x{h} @ {args.fps}fps -> {args.out}")
        writer.write(out_frame)
        n_written += 1

        if n_written % 500 == 0:
            print(f"  {n_written} frames…")

    if writer is not None:
        writer.release()
    print(f"Done. {n_written} frames written "
          f"({n_fallback} full-frame fallbacks where no ROI) -> {args.out}")


if __name__ == "__main__":
    main()
