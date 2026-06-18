#!/usr/bin/env python3
"""
Prepare Roboflow volleyball ball dataset for YOLOX training.

Downloads and converts a COCO-format Roboflow export to YOLOX directory layout:
  data/volleyball_ball/
    train/  images/ + labels/ (YOLO txt format, class 0 = ball)
    val/    images/ + labels/
    test/   images/ + labels/ (optional)

Usage:
  # Download from Roboflow (requires API key):
  python scripts/prepare_ball_dataset.py \\
    --roboflow-key YOUR_KEY \\
    --workspace volleytrack \\
    --project volleyball-tracking-sgety \\
    --version 1 \\
    --out-dir data/volleyball_ball

  # Or convert an already-downloaded COCO JSON:
  python scripts/prepare_ball_dataset.py \\
    --coco-dir /path/to/roboflow_export \\
    --out-dir data/volleyball_ball
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def _yolo_label(ann: dict, img_w: int, img_h: int, class_id: int = 0) -> str:
    x, y, w, h = ann["bbox"]  # COCO: top-left x,y and w,h
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def convert_coco_to_yolo(
    coco_json: Path,
    src_img_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
) -> int:
    """Convert a COCO annotation JSON to YOLO txt labels. Returns image count."""
    with open(coco_json) as f:
        coco = json.load(f)

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    id2img = {img["id"]: img for img in coco["images"]}
    img_anns: dict[int, list] = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    count = 0
    for img_id, img_meta in id2img.items():
        src = src_img_dir / img_meta["file_name"]
        if not src.exists():
            continue
        dst = dst_img_dir / img_meta["file_name"]
        shutil.copy2(src, dst)

        lbl_path = dst_lbl_dir / (Path(img_meta["file_name"]).stem + ".txt")
        lines = [
            _yolo_label(ann, img_meta["width"], img_meta["height"])
            for ann in img_anns[img_id]
        ]
        lbl_path.write_text("\n".join(lines))
        count += 1

    return count


def download_roboflow(key: str, workspace: str, project: str, version: int, out_dir: Path) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("pip install roboflow")

    rf = Roboflow(api_key=key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("coco", location=str(out_dir))
    return Path(dataset.location)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--roboflow-key", help="Roboflow API key")
    g.add_argument("--coco-dir", help="Path to already-downloaded Roboflow COCO export")
    p.add_argument("--workspace", default="volleytrack")
    p.add_argument("--project", default="volleyball-tracking-sgety")
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--out-dir", default="data/volleyball_ball")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)

    if args.roboflow_key:
        print("Downloading from Roboflow …")
        coco_root = download_roboflow(
            args.roboflow_key, args.workspace, args.project, args.version, out / "_raw"
        )
    else:
        coco_root = Path(args.coco_dir)

    total = 0
    for split in ("train", "valid", "test"):
        json_path = coco_root / split / "_annotations.coco.json"
        if not json_path.exists():
            continue
        dst_split = "val" if split == "valid" else split
        n = convert_coco_to_yolo(
            json_path,
            coco_root / split,
            out / dst_split / "images",
            out / dst_split / "labels",
        )
        print(f"  {dst_split}: {n} images")
        total += n

    print(f"Done. Total: {total} images → {out}")


if __name__ == "__main__":
    main()
