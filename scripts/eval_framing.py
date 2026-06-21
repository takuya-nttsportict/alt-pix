#!/usr/bin/env python3
"""Evaluate framing quality from a pipeline JSONL log (Phase 5).

Reproducible metrics (CLAUDE.md principle 2) computed offline from out.jsonl —
no GPU, no re-run needed. Quantifies the broadcast-smooth design's goals:

  ball_in_roi%      fraction of visible-ball frames where the ball lies inside
                    the ROI  (higher = the subject stays framed)
  field_in_roi%     mean fraction of field players' bbox area inside the ROI
                    (higher = the rally stays in shot)
  pan_jitter_px     RMS of the *2nd difference* of the ROI centre = camera
                    acceleration. High-frequency content the eye reads as
                    jitter; LOWER is smoother.
  pan_speed_px      mean |1st difference| of the ROI centre per frame (pixels
                    of pan per frame; informational)
  zoom_jitter       RMS 2nd difference of ROI width / mean width (unitless);
                    LOWER = fewer zoom in/out oscillations
  roi_off_frac      fraction of frames the ROI clamps to a frame edge
                    (informational — heavy clamping means under-zoom)

Usage:
  python scripts/eval_framing.py out.jsonl
  python scripts/eval_framing.py a.jsonl b.jsonl   # compare two runs side by side
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _center(roi: dict) -> tuple[float, float]:
    return roi["x"] + roi["w"] / 2.0, roi["y"] + roi["h"] / 2.0


def _box_overlap_frac(bbox: list[float], roi: dict) -> float:
    """Fraction of bbox area inside roi."""
    bx1, by1, bx2, by2 = bbox
    area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area <= 0:
        return 0.0
    rx1, ry1 = roi["x"], roi["y"]
    rx2, ry2 = roi["x"] + roi["w"], roi["y"] + roi["h"]
    ix = max(0.0, min(bx2, rx2) - max(bx1, rx1))
    iy = max(0.0, min(by2, ry2) - max(by1, ry1))
    return (ix * iy) / area


def evaluate(records: list[dict], frame_w: int | None = None) -> dict:
    cxs, ws = [], []
    ball_in = ball_tot = 0
    field_fracs: list[float] = []
    off_edge = 0
    n_roi = 0

    for r in records:
        roi = r.get("framing_roi")
        if not roi:
            continue
        n_roi += 1
        cx, _ = _center(roi)
        cxs.append(cx)
        ws.append(roi["w"])

        ball = r.get("ball") or {}
        if ball.get("visible"):
            ball_tot += 1
            if roi["x"] <= ball["x"] <= roi["x"] + roi["w"] and \
               roi["y"] <= ball["y"] <= roi["y"] + roi["h"]:
                ball_in += 1

        for p in r.get("players", []):
            if p.get("role") in (None, "field") and p.get("bbox"):
                field_fracs.append(_box_overlap_frac(p["bbox"], roi))

        if frame_w and (roi["x"] <= 0 or roi["x"] + roi["w"] >= frame_w):
            off_edge += 1

    def _rms_2nd_diff(seq: list[float]) -> float:
        if len(seq) < 3:
            return 0.0
        d2 = [seq[i] - 2 * seq[i - 1] + seq[i - 2] for i in range(2, len(seq))]
        return math.sqrt(sum(v * v for v in d2) / len(d2))

    def _mean_abs_1st(seq: list[float]) -> float:
        if len(seq) < 2:
            return 0.0
        return sum(abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))) / (len(seq) - 1)

    mean_w = sum(ws) / len(ws) if ws else 1.0
    return {
        "frames": n_roi,
        "ball_in_roi_pct": 100.0 * ball_in / ball_tot if ball_tot else float("nan"),
        "field_in_roi_pct": 100.0 * sum(field_fracs) / len(field_fracs) if field_fracs else float("nan"),
        "pan_jitter_px": _rms_2nd_diff(cxs),
        "pan_speed_px": _mean_abs_1st(cxs),
        "zoom_jitter": _rms_2nd_diff(ws) / mean_w if mean_w else 0.0,
        "roi_off_frac": off_edge / n_roi if (frame_w and n_roi) else float("nan"),
    }


def _fmt(m: dict) -> str:
    return (
        f"  frames           : {m['frames']}\n"
        f"  ball_in_roi      : {m['ball_in_roi_pct']:.1f}%   (higher=subject framed)\n"
        f"  field_in_roi     : {m['field_in_roi_pct']:.1f}%   (higher=rally in shot)\n"
        f"  pan_jitter (acc) : {m['pan_jitter_px']:.2f} px   (LOWER=smoother)\n"
        f"  pan_speed        : {m['pan_speed_px']:.2f} px/frame\n"
        f"  zoom_jitter      : {m['zoom_jitter']:.4f}       (LOWER=less hunting)\n"
        f"  roi_off_frac     : {m['roi_off_frac']:.2f}\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate framing quality from JSONL")
    p.add_argument("jsonl", nargs="+", help="One or more pipeline JSONL logs")
    p.add_argument("--frame-w", type=int, default=None,
                   help="Frame width (px) to compute roi_off_frac")
    args = p.parse_args()

    for path in args.jsonl:
        recs = _load(path)
        m = evaluate(recs, args.frame_w)
        print(f"\n=== {Path(path).name} ===")
        print(_fmt(m))


if __name__ == "__main__":
    main()
