"""
PyTorch Dataset for YOLO-format volleyball ball annotations.

Used for evaluation and quick sanity-checks without the full YOLOX trainer.
Expected directory layout (output of prepare_ball_dataset.py):

  data/volleyball_ball/
    train/
      images/  *.jpg
      labels/  *.txt   (class cx cy w h, normalized, class=0 for ball)
    val/
      images/  *.jpg
      labels/  *.txt
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BallDataset(Dataset):
    """Single-class volleyball ball dataset in YOLO txt format.

    Args:
        root: Path to split directory (e.g. data/volleyball_ball/val).
        img_size: Resize images to this square size for inference.
    """

    def __init__(self, root: str | Path, img_size: int = 640) -> None:
        self._root = Path(root)
        self._img_size = img_size
        self._imgs = sorted((self._root / "images").glob("*.jpg"))
        self._imgs += sorted((self._root / "images").glob("*.png"))

    def __len__(self) -> int:
        return len(self._imgs)

    def __getitem__(self, idx: int) -> dict:
        img_path = self._imgs[idx]
        lbl_path = self._root / "labels" / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (self._img_size, self._img_size))
        tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0

        bboxes: list[list[float]] = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, bw, bh = map(float, parts)
                    # Convert to pixel coords in original resolution
                    bboxes.append([
                        (cx - bw / 2) * w,
                        (cy - bh / 2) * h,
                        (cx + bw / 2) * w,
                        (cy + bh / 2) * h,
                    ])

        return {
            "image": tensor,
            "bboxes": bboxes,  # list of [x1,y1,x2,y2] in original pixels
            "path": str(img_path),
            "orig_size": (h, w),
        }
