"""TrackNetV2 ball detector.

Exact architecture from nttcom/WASB-SBDT (MIT licence, BMVC 2023):
  src/models/unet2d.py + src/models/unet2d_parts.py

State-dict key mapping verified against volleyball checkpoint:
  inc.double_conv.{0,2,3,5}.*       DoubleConv: Conv→ReLU→BN→Conv→ReLU→BN
  down{1}.maxpool_conv.1.double_conv.*  Down2
  down{2,3}.maxpool_conv.1.triple_conv.*  Down3
  up1.conv.triple_conv.*  Up3
  up{2,3}.conv.double_conv.*  Up2
  outc.conv.*

Input : (B, 9, H, W)  — 3 consecutive frames, ImageNet-normalised RGB tiles
Output: (B, 3, H, W)  — raw logits per frame; channel 2 = most recent frame.

Preprocessing (verified via scan_tracknet.py on 3840x800 volleyball footage):
  BGR → RGB → crop 16:9 tile at native height → resize to 512x288 →
  ÷255 → ImageNet mean/std normalise.

Tiling rationale:
  The source footage is 3840x800 (aspect 4.8).  Shrinking the whole frame to
  512x288 would compress a 12px ball to ~1.6px, making the model blind.
  Instead we split the frame into overlapping 16:9 tiles (height=frame_height,
  width=frame_height*512/288) and run TrackNet on each tile independently.
  The tile with the highest sigmoid peak wins.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from .detector import Detection, _COCO_SPORTS_BALL

logger = logging.getLogger(__name__)

_W = 512
_H = 288

# WASB-SBDT preprocessing: ToTensor (÷255) → Normalize(ImageNet mean/std) on RGB.
# Verified against src/dataloaders/__init__.py build_img_transforms().
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Building blocks ───────────────────────────────────────────────────────────

class _DoubleConv(nn.Module):
    """Conv→ReLU→BN→Conv→ReLU→BN. Attr name must be double_conv."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),   # [0]
            nn.ReLU(inplace=True),                                # [1]
            nn.BatchNorm2d(out_ch),                               # [2]
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),  # [3]
            nn.ReLU(inplace=True),                                # [4]
            nn.BatchNorm2d(out_ch),                               # [5]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class _TripleConv(nn.Module):
    """Conv→ReLU→BN ×3. Attr name must be triple_conv."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),   # [0]
            nn.ReLU(inplace=True),                                # [1]
            nn.BatchNorm2d(out_ch),                               # [2]
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),  # [3]
            nn.ReLU(inplace=True),                                # [4]
            nn.BatchNorm2d(out_ch),                               # [5]
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),  # [6]
            nn.ReLU(inplace=True),                                # [7]
            nn.BatchNorm2d(out_ch),                               # [8]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.triple_conv(x)


class _Down2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class _Down3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), _TripleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class _Up2(nn.Module):
    """Upsample + cat([skip, up(x)]) + DoubleConv. Skip channels first (WASB order)."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = _DoubleConv(skip_ch + in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([skip, self.up(x)], dim=1))


class _Up3(nn.Module):
    """Upsample + cat([skip, up(x)]) + TripleConv. Skip channels first (WASB order)."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = _TripleConv(skip_ch + in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([skip, self.up(x)], dim=1))


class _OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── Model ─────────────────────────────────────────────────────────────────────

class TrackNetV2(nn.Module):
    """Exact replica of WASB-SBDT TrackNetV2 (unet2d.py).

    Output: dict {0: (B, n_classes, H, W)} — raw logits, no sigmoid.
    """

    def __init__(self, n_channels: int = 9, n_classes: int = 3) -> None:
        super().__init__()
        self.inc   = _DoubleConv(n_channels, 64)
        self.down1 = _Down2(64, 128)
        self.down2 = _Down3(128, 256)
        self.down3 = _Down3(256, 512)
        self.up1   = _Up3(512, 256, 256)
        self.up2   = _Up2(256, 128, 128)
        self.up3   = _Up2(128, 64, 64)
        self.outc  = _OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x  = self.up1(x4, x3)
        x  = self.up2(x,  x2)
        x  = self.up3(x,  x1)
        return {0: self.outc(x)}


# ── Tiling helpers ────────────────────────────────────────────────────────────

def _compute_tiles(frame_w: int, frame_h: int, tile_overlap: float = 0.3) -> list[int]:
    """Return list of tile x-offsets for overlapping 16:9 tiles at frame height.

    Each tile is (tile_w × frame_h) where tile_w = frame_h × (512/288).
    For standard 16:9 footage a single tile covers the whole frame.
    """
    tile_w = min(int(math.ceil(frame_h * _W / _H)), frame_w)
    if tile_w >= frame_w:
        return [0]
    step = int(math.ceil(tile_w * (1.0 - tile_overlap)))
    xs: list[int] = []
    x = 0
    while x + tile_w <= frame_w:
        xs.append(x)
        x += step
    # Always include a tile ending at the right edge
    if not xs or xs[-1] + tile_w < frame_w:
        xs.append(frame_w - tile_w)
    return xs


def _preprocess_tile(frame_bgr: np.ndarray, x0: int, tile_w: int) -> torch.Tensor:
    """Crop a tile, resize to 512×288, normalise. Returns (3, H, W) float32."""
    crop = frame_bgr[:, x0: x0 + tile_w]               # (h, tile_w, 3) BGR
    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rs   = cv2.resize(rgb, (_W, _H), interpolation=cv2.INTER_LINEAR)
    t    = rs.astype(np.float32) / 255.0
    t    = (t - _MEAN) / _STD
    return torch.from_numpy(t).permute(2, 0, 1).contiguous()


# ── Inference wrapper ─────────────────────────────────────────────────────────

class TrackNetDetector:
    """Frame-by-frame ball detector using TrackNetV2 (WASB-SBDT) with tiling.

    For ultrawide panoramic footage (e.g. 3840×800) the full frame cannot be
    naively resized to 512×288 because a 12px ball shrinks to <2px and
    disappears.  Instead the frame is split into overlapping 16:9 tiles at
    native height; TrackNet runs on each tile and the highest-confidence
    result is returned.

    For standard 16:9 footage a single tile covers the whole frame, so there
    is no overhead.

    Call detect(frame) once per frame.  Returns [] for the first 2 frames
    while the 3-frame buffer warms up.

    Args:
        model_path : Path to .pt checkpoint.
        conf_thr   : Minimum sigmoid peak to report a detection (default 0.5).
        tile_overlap: Fractional overlap between adjacent tiles (default 0.3).
        device     : 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_thr: float = 0.5,
        tile_overlap: float = 0.3,
        device: str = "cuda",
    ) -> None:
        self._conf_thr    = conf_thr
        self._tile_overlap = tile_overlap
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._model = TrackNetV2(n_channels=9, n_classes=3)
        ckpt = torch.load(str(model_path), map_location=self._device, weights_only=False)
        if isinstance(ckpt, dict):
            state = ckpt.get("model_state_dict",
                             ckpt.get("model", ckpt.get("state_dict", ckpt)))
        else:
            state = ckpt
        state = {(k[len("module."):] if k.startswith("module.") else k): v
                 for k, v in state.items()}
        self._model.load_state_dict(state, strict=True)
        self._model.eval().to(self._device)

        # Tile layout and per-tile 3-frame buffers — initialised on first frame.
        self._tile_xs:  list[int] | None = None   # x-offsets of tiles
        self._tile_w:   int | None = None          # tile width in pixels
        self._frame_hw: tuple[int, int] | None = None
        self._bufs:     list[deque[torch.Tensor]] = []

        logger.info(
            f"TrackNetV2 loaded: {model_path}  device={self._device}  "
            f"conf_thr={conf_thr}  tile_overlap={tile_overlap}"
        )

    # ── public ────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return at most 1 Detection (the ball), or [] if below threshold."""
        h, w = frame.shape[:2]
        self._ensure_tiles(h, w)

        # Append new preprocessed tile crops to each buffer
        for i, x0 in enumerate(self._tile_xs):
            self._bufs[i].append(_preprocess_tile(frame, x0, self._tile_w))

        if len(self._bufs[0]) < 3:
            return []   # buffer warming up

        best_peak = 0.0
        best_det: Detection | None = None

        for i, x0 in enumerate(self._tile_xs):
            stacked = torch.cat(list(self._bufs[i]), dim=0).unsqueeze(0).to(self._device)
            with torch.no_grad():
                logits  = self._model(stacked)[0]         # (1, 3, H, W)
                heatmap = torch.sigmoid(logits[0, 2])     # (H, W)

            hm   = heatmap.cpu().numpy()
            peak = float(hm.max())

            if peak > best_peak:
                best_peak = peak
                if peak >= self._conf_thr:
                    ym, xm = np.unravel_index(hm.argmax(), hm.shape)
                    # Map heatmap coords → full-frame coords
                    cx = x0 + float(xm) * self._tile_w / _W
                    cy = float(ym) * h / _H
                    r  = max(min(w, h) * 0.02, 4.0)
                    best_det = Detection(
                        bbox=(cx - r, cy - r, cx + r, cy + r),
                        conf=peak,
                        class_id=_COCO_SPORTS_BALL,
                    )

        if best_det is None:
            logger.debug(f"TrackNet: best_peak={best_peak:.3f} < thr={self._conf_thr} → no ball")
            return []

        cx = (best_det.bbox[0] + best_det.bbox[2]) / 2
        cy = (best_det.bbox[1] + best_det.bbox[3]) / 2
        logger.debug(
            f"TrackNet: ball=({cx:.1f},{cy:.1f})  peak={best_peak:.3f}  "
            f"tiles={len(self._tile_xs)}"
        )
        return [best_det]

    def reset(self) -> None:
        """Clear all tile buffers (call after stream discontinuity)."""
        for buf in self._bufs:
            buf.clear()

    # ── private ───────────────────────────────────────────────────────────────

    def _ensure_tiles(self, h: int, w: int) -> None:
        """Compute tile layout on first call or if frame size changes."""
        if self._frame_hw == (h, w):
            return
        self._tile_xs = _compute_tiles(w, h, self._tile_overlap)
        self._tile_w  = min(int(math.ceil(h * _W / _H)), w)
        self._bufs    = [deque(maxlen=3) for _ in self._tile_xs]
        self._frame_hw = (h, w)
        logger.info(
            f"TrackNet: frame={w}×{h}  tile_w={self._tile_w}  "
            f"n_tiles={len(self._tile_xs)}  xs={self._tile_xs}"
        )
