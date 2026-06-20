#!/usr/bin/env python3
"""Fine-tune TrackNetV3 on volleyball dataset.

Dataset format (YOLO-style):
  data/tracknet_volleyball/
    train/
      images/    *.jpg or *.png
      labels/    *.txt  — one line per ball: "0 cx cy w h" (normalised)
    val/
      images/
      labels/

If you only have Roboflow COCO exports, first run:
  python scripts/prepare_ball_dataset.py --coco-dir ... --out-dir data/volleyball_ball

Then use data/volleyball_ball/ directly (same directory layout).

Usage:
  # Fine-tune from TrackNetV3 pre-trained weights:
  python scripts/fine_tune_tracknet.py \\
    --data   data/volleyball_ball \\
    --ckpt   models/tracknet_v3.pt \\
    --out    models/tracknet_v3_volleyball.pt \\
    --epochs 50 \\
    --batch  8 \\
    --device cuda

  # Train from scratch (no pre-trained checkpoint):
  python scripts/fine_tune_tracknet.py \\
    --data   data/volleyball_ball \\
    --out    models/tracknet_v3_volleyball.pt \\
    --epochs 100

Design notes:
  - TrackNetV3 requires 3 consecutive frames as input.
  - For image-only datasets, we synthesise pseudo-sequence by augmenting
    the same image 3 times (translation/blur jitter).  This teaches the
    model ball appearance without temporal info.
  - The output is a Gaussian heatmap centred at the ball label position.
  - Loss: focal-like binary cross-entropy (handles extreme class imbalance —
    the ball is tiny compared to the background).
"""

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from alt_pix.tracknet import TrackNetV3

# ── Heatmap generation ────────────────────────────────────────────────────────

def _make_heatmap(
    cx_norm: float, cy_norm: float,
    out_h: int, out_w: int,
    sigma: float = 5.0,
) -> np.ndarray:
    """Generate a 2D Gaussian heatmap.  Returns (out_h, out_w) float32 in [0,1]."""
    cx = cx_norm * out_w
    cy = cy_norm * out_h

    xs = np.arange(out_w, dtype=np.float32)
    ys = np.arange(out_h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    heatmap = np.exp(-((xv - cx) ** 2 + (yv - cy) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TrackNetDataset(Dataset):
    """Image + YOLO-format label → 9-channel input tensor + heatmap target.

    Each item returns (input_9ch, heatmap) where input_9ch is built by
    synthesising 3 slightly-jittered versions of the same image (or using
    neighbouring frames if a sequence is supplied).
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        input_size: tuple[int, int] = (512, 288),  # (W, H)
        sigma: float = 5.0,
        augment: bool = True,
    ) -> None:
        self.input_w, self.input_h = input_size
        self.sigma = sigma
        self.augment = augment

        img_dir = data_dir / split / "images"
        lbl_dir = data_dir / split / "labels"

        self.samples: list[tuple[Path, float, float]] = []  # (img_path, cx_norm, cy_norm)
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            lines = lbl_path.read_text().strip().splitlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cx, cy = float(parts[1]), float(parts[2])
                img_stem = lbl_path.stem
                for ext in (".jpg", ".jpeg", ".png"):
                    img_path = img_dir / (img_stem + ext)
                    if img_path.exists():
                        self.samples.append((img_path, cx, cy))
                        break

        print(f"  {split}: {len(self.samples)} labelled ball instances")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, cx, cy = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            # Return blank if image unreadable
            blank = torch.zeros(9, self.input_h, self.input_w)
            heat = torch.zeros(1, self.input_h, self.input_w)
            return blank, heat

        # Build 3 pseudo-frames via jitter
        frames = [self._jitter(img) for _ in range(3)]

        # Preprocess each frame: resize → (H, W, 3) → normalise → (3, H, W)
        tensors = []
        for f in frames:
            resized = cv2.resize(f, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(resized).float() / 255.0   # (H, W, 3)
            tensors.append(t.permute(2, 0, 1))              # (3, H, W)

        input_9ch = torch.cat(tensors, dim=0)  # (9, H, W)

        heatmap = _make_heatmap(cx, cy, self.input_h, self.input_w, self.sigma)
        heat_t = torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)

        return input_9ch, heat_t

    def _jitter(self, img: np.ndarray) -> np.ndarray:
        if not self.augment:
            return img
        # Small random translation
        h, w = img.shape[:2]
        dx = random.randint(-int(w * 0.02), int(w * 0.02))
        dy = random.randint(-int(h * 0.02), int(h * 0.02))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (w, h))
        # Random slight blur to mimic motion
        if random.random() < 0.5:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        return img


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalBCELoss(nn.Module):
    """Focal-BCE for imbalanced heatmap regression."""

    def __init__(self, gamma: float = 2.0, pos_weight: float = 10.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        # Up-weight positive pixels
        w = 1.0 + (self.pos_weight - 1.0) * target
        return (w * loss).mean()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data)

    print(f"Data dir  : {data_dir}")
    print(f"Device    : {device}")
    print(f"Epochs    : {args.epochs}")
    print(f"Batch size: {args.batch}")

    train_ds = TrackNetDataset(data_dir, "train", augment=True)
    val_ds   = TrackNetDataset(data_dir, "val",   augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=2, pin_memory=True)

    model = TrackNetV3().to(device)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        print(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys  ({len(missing)}): {missing[:5]} …")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
    else:
        print("Training from scratch (no pretrained checkpoint).")

    criterion = FocalBCELoss(gamma=2.0, pos_weight=args.pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = math.inf
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for inp, target in train_dl:
            inp, target = inp.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(inp)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= max(len(train_dl), 1)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, target in val_dl:
                inp, target = inp.to(device), target.to(device)
                pred = model(inp)
                val_loss += criterion(pred, target).item()
        val_loss /= max(len(val_dl), 1)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), str(out_path))
            print(f"  ✓ Saved best model → {out_path}  (val={val_loss:.4f})")

    print(f"\nTraining complete.  Best val loss: {best_val:.4f}")
    print(f"Model saved to: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune TrackNetV3 for volleyball ball detection")
    p.add_argument("--data",   required=True, help="Dataset root (contains train/ and val/)")
    p.add_argument("--ckpt",   default=None,  help="Initial checkpoint (.pt).  Omit to train from scratch.")
    p.add_argument("--out",    default="models/tracknet_v3_volleyball.pt", help="Output model path")
    p.add_argument("--epochs", type=int,   default=50)
    p.add_argument("--batch",  type=int,   default=8)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=10.0,
                   help="Weight for ball pixels in focal loss (higher = more recall)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
