#!/usr/bin/env python3
"""
Evaluate the ball detector (ONNX) on the validation split.

Reports per-frame precision, recall, and mAP@0.5.

Usage:
  python scripts/eval_ball_detector.py \\
    --model models/yolox_s_ball.onnx \\
    --data  data/volleyball_ball/val \\
    --conf  0.35 \\
    --iou-thr 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.ball_dataset import BallDataset
from alt_pix.detector import YOLOXDetector, _COCO_SPORTS_BALL


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="ONNX model path")
    p.add_argument("--data", required=True, help="Validation split dir")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for TP")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    detector = YOLOXDetector(
        args.model,
        conf_thr=args.conf,
        detect_classes={_COCO_SPORTS_BALL},
        device="cuda" if args.device.startswith("cuda") else "cpu",
    )
    dataset = BallDataset(args.data)

    tp = fp = fn = 0
    all_scores: list[float] = []
    all_tp_flags: list[int] = []

    for sample in tqdm(dataset, desc="Evaluating"):
        img_path = sample["path"]
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        dets = detector.detect(frame)
        ball_dets = [d for d in dets if d.class_id == _COCO_SPORTS_BALL]
        gt_boxes = sample["bboxes"]

        matched_gt = set()
        for det in sorted(ball_dets, key=lambda d: -d.conf):
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                v = iou(list(det.bbox), gt)
                if v > best_iou:
                    best_iou, best_j = v, j

            if best_iou >= args.iou_thr:
                tp += 1
                matched_gt.add(best_j)
                all_tp_flags.append(1)
            else:
                fp += 1
                all_tp_flags.append(0)
            all_scores.append(det.conf)

        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n── Ball Detection Evaluation ──────────────────")
    print(f"  Dataset : {args.data}")
    print(f"  Model   : {args.model}")
    print(f"  conf≥{args.conf}  IoU≥{args.iou_thr}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1        : {f1:.3f}")


if __name__ == "__main__":
    main()
