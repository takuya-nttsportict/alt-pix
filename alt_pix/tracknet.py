"""TrackNetV2 ball detector.

Exact architecture from nttcom/WASB-SBDT (MIT licence, BMVC 2023):
  src/models/unet2d.py + src/models/unet2d_parts.py

Input : (B, 9, H, W)  — 3 consecutive BGR frames, each normalised to [0,1]
Output: (B, 3, H, W)  — raw logits, one heatmap per input frame
        sigmoid applied in post-processing; channel 2 = most recent frame.

Config used for volleyball weights (src/configs/model/tracknetv2.yaml):
  frames_in=3, frames_out=3, inp=512×288, bilinear=True, mode='nearest'
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from .detector import Detection, _COCO_SPORTS_BALL

logger = logging.getLogger(__name__)

# Inference resolution from WASB config (must be divisible by 8)
_W = 512
_H = 288


# ── Building blocks (matches unet2d_parts.py bn_first=False) ─────────────────

def _cbr(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d → BatchNorm2d → ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _DoubleConv(nn.Module):
    """Two CBR blocks. Matches DoubleConv(bn_first=False)."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(_cbr(in_ch, out_ch), _cbr(out_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TripleConv(nn.Module):
    """Three CBR blocks. Matches TripleConv(bn_first=False)."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _cbr(in_ch, out_ch),
            _cbr(out_ch, out_ch),
            _cbr(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down2(nn.Module):
    """MaxPool2d × 2 + DoubleConv. Matches Down(n=2, ...)."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down3(nn.Module):
    """MaxPool2d × 2 + TripleConv. Matches Down(n=3, ...)."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _TripleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up2(nn.Module):
    """Nearest upsample + cat skip + DoubleConv. Matches Up(n=2, bilinear=True, mode='nearest')."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class _Up3(nn.Module):
    """Nearest upsample + cat skip + TripleConv. Matches Up(n=3, bilinear=True, mode='nearest')."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = _TripleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


# ── Model ─────────────────────────────────────────────────────────────────────

class TrackNetV2(nn.Module):
    """Exact replica of WASB-SBDT TrackNetV2 (unet2d.py).

    TrackNetV2(n_channels=9, n_classes=3, bilinear=True, mode='nearest', halve_channel=False)

    Encoder: 3 downsampling stages (8× total reduction)
    Decoder: 3 upsampling stages with skip connections
    Output:  dict {0: (B, n_classes, H, W)} — raw logits, no sigmoid
    """

    def __init__(self, n_channels: int = 9, n_classes: int = 3) -> None:
        super().__init__()
        self.inc   = _DoubleConv(n_channels, 64)    # (B, 64, H, W)
        self.down1 = _Down2(64, 128)                # (B, 128, H/2, W/2)
        self.down2 = _Down3(128, 256)               # (B, 256, H/4, W/4)
        self.down3 = _Down3(256, 512)               # (B, 512, H/8, W/8)
        self.up1   = _Up3(512, 256, 256)            # (B, 256, H/4, W/4)
        self.up2   = _Up2(256, 128, 128)            # (B, 128, H/2, W/2)
        self.up3   = _Up2(128, 64, 64)              # (B, 64, H, W)
        self.outc  = nn.Conv2d(64, n_classes, 1)   # (B, n_classes, H, W)

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x  = self.up1(x4, x3)
        x  = self.up2(x,  x2)
        x  = self.up3(x,  x1)
        return {0: self.outc(x)}


# ── Inference wrapper ─────────────────────────────────────────────────────────

class TrackNetDetector:
    """Frame-by-frame ball detector using TrackNetV2 (WASB-SBDT).

    Call detect(frame) once per frame.  Maintains an internal 3-frame buffer;
    returns [] for the first 2 frames while the buffer fills.

    Args:
        model_path : Path to .pt checkpoint (state dict).
        input_size : (width, height) for inference — must be divisible by 8.
        conf_thr   : Minimum sigmoid heatmap peak to report a detection.
        device     : 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (_W, _H),
        conf_thr: float = 0.5,
        device: str = "cuda",
    ) -> None:
        self._w, self._h = input_size
        self._conf_thr = conf_thr
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._model = TrackNetV2(n_channels=9, n_classes=3)
        ckpt = torch.load(str(model_path), map_location=self._device, weights_only=True)
        # Accept plain state dict or wrapped {"model": ..., "state_dict": ...}
        if isinstance(ckpt, dict):
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        else:
            state = ckpt
        # Strip DataParallel 'module.' prefix if present
        state = {(k[len("module."):] if k.startswith("module.") else k): v
                 for k, v in state.items()}
        self._model.load_state_dict(state, strict=True)
        self._model.eval().to(self._device)

        self._buf: deque[torch.Tensor] = deque(maxlen=3)  # (3, H, W) per frame

        logger.info(
            f"TrackNetV2 loaded: {model_path}  "
            f"device={self._device}  input={self._w}×{self._h}  conf_thr={conf_thr}"
        )

    # ── public ────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return a list of at most 1 Detection (the ball) or [] if not found."""
        h_orig, w_orig = frame.shape[:2]
        self._buf.append(self._preprocess(frame))

        if len(self._buf) < 3:
            return []  # buffer warm-up

        stacked = torch.cat(list(self._buf), dim=0).unsqueeze(0).to(self._device)  # (1, 9, H, W)

        with torch.no_grad():
            logits = self._model(stacked)[0]          # (1, 3, H, W)
            heatmap = torch.sigmoid(logits[0, 2])     # (H, W) — most recent frame

        return self._postprocess(heatmap, w_orig, h_orig)

    def reset(self) -> None:
        """Clear the frame buffer (call after stream discontinuity)."""
        self._buf.clear()

    # ── private ───────────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Resize to (W, H), normalise to [0,1]. Returns (3, H, W) float32."""
        resized = cv2.resize(frame, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).float() / 255.0   # (H, W, 3)
        return t.permute(2, 0, 1)                        # (3, H, W)

    def _postprocess(
        self,
        heatmap: torch.Tensor,
        w_orig: int,
        h_orig: int,
    ) -> list[Detection]:
        hm = heatmap.cpu().numpy()   # (H, W) float32 in [0,1]
        peak = float(hm.max())

        if peak < self._conf_thr:
            logger.debug(f"TrackNet: peak={peak:.3f} < thr={self._conf_thr} → no ball")
            return []

        ym, xm = np.unravel_index(hm.argmax(), hm.shape)

        # Scale from inference resolution back to original frame
        cx = float(xm) * w_orig / self._w
        cy = float(ym) * h_orig / self._h

        # Radius ~2% of shorter dimension, minimum 4px
        r = max(min(w_orig, h_orig) * 0.02, 4.0)

        logger.debug(f"TrackNet: ball=({cx:.1f},{cy:.1f})  peak={peak:.3f}  r={r:.1f}")

        return [Detection(
            bbox=(cx - r, cy - r, cx + r, cy + r),
            conf=peak,
            class_id=_COCO_SPORTS_BALL,
        )]
