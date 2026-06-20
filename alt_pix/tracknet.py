"""TrackNetV3 ball detector for volleyball.

Architecture: VGG16-BN encoder + U-Net decoder, heatmap output.
Input: 3 consecutive frames stacked to (B, 9, H, W).
Output: (B, 1, H, W) probability heatmap — peak = ball location.

Reference: https://github.com/Chang-Chia-Chi/TrackNet
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from .detector import Detection, _COCO_SPORTS_BALL

logger = logging.getLogger(__name__)

# Default inference resolution (width, height)
# Must be divisible by 16 for the 4× MaxPool encoder
_DEFAULT_W = 512
_DEFAULT_H = 288


# ── Model ─────────────────────────────────────────────────────────────────────

class _CBR(nn.Module):
    """Conv → BN → ReLU block."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrackNetV3(nn.Module):
    """VGG16-BN encoder, U-Net decoder.

    Input : (B, 9, H, W)  — 3 consecutive BGR frames, normalised to [0,1]
    Output: (B, 1, H, W)  — sigmoid heatmap
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────────
        self.e1 = nn.Sequential(_CBR(9, 64), _CBR(64, 64))           # (B, 64, H, W)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.e2 = nn.Sequential(_CBR(64, 128), _CBR(128, 128))        # (B,128, H/2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.e3 = nn.Sequential(_CBR(128, 256), _CBR(256, 256), _CBR(256, 256))  # (B,256,H/4)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.e4 = nn.Sequential(_CBR(256, 512), _CBR(512, 512), _CBR(512, 512))  # (B,512,H/8)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.e5 = nn.Sequential(_CBR(512, 512), _CBR(512, 512), _CBR(512, 512))  # (B,512,H/16)

        # ── Decoder ───────────────────────────────────────────────────────
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # after cat with e4 → 512+512=1024 in
        self.d5 = nn.Sequential(_CBR(1024, 512), _CBR(512, 512), _CBR(512, 256))

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # after cat with e3 → 256+256=512 in
        self.d4 = nn.Sequential(_CBR(512, 256), _CBR(256, 256), _CBR(256, 128))

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # after cat with e2 → 128+128=256 in
        self.d3 = nn.Sequential(_CBR(256, 128), _CBR(128, 64))

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # after cat with e1 → 64+64=128 in
        self.d2 = nn.Sequential(_CBR(128, 64), _CBR(64, 64))

        # Output 3 heatmaps (one per input frame) — matches qaz812345/TrackNetV3 weights
        self.head = nn.Sequential(nn.Conv2d(64, 3, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool1(e1))
        e3 = self.e3(self.pool2(e2))
        e4 = self.e4(self.pool3(e3))
        e5 = self.e5(self.pool4(e4))

        d = self.d5(torch.cat([self.up5(e5), e4], dim=1))
        d = self.d4(torch.cat([self.up4(d),  e3], dim=1))
        d = self.d3(torch.cat([self.up3(d),  e2], dim=1))
        d = self.d2(torch.cat([self.up2(d),  e1], dim=1))
        return self.head(d)


# ── Inference wrapper ─────────────────────────────────────────────────────────

class TrackNetDetector:
    """Frame-by-frame ball detector using TrackNetV3.

    Call detect(frame) once per frame.  The detector maintains an internal
    3-frame buffer; returns [] for the first 2 frames (warm-up).

    Args:
        model_path: Path to .pt checkpoint produced by download_tracknet_weights.py
                    or fine_tune_tracknet.py.
        input_size: (width, height) for inference.  Must be divisible by 16.
        conf_thr:   Minimum heatmap peak to report a detection.
        device:     'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (_DEFAULT_W, _DEFAULT_H),
        conf_thr: float = 0.5,
        device: str = "cuda",
    ) -> None:
        self._w, self._h = input_size
        self._conf_thr = conf_thr
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._model = TrackNetV3()
        ckpt = torch.load(str(model_path), map_location=self._device, weights_only=True)
        # Accept either a raw state_dict or a {'model': state_dict, ...} wrapper
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        self._model.load_state_dict(state, strict=True)
        self._model.eval().to(self._device)

        self._buf: deque[torch.Tensor] = deque(maxlen=3)  # stores (3, H, W) per frame

        logger.info(
            f"TrackNetV3 loaded from {model_path}  "
            f"device={self._device}  input={self._w}×{self._h}  conf_thr={conf_thr}"
        )

    # ── public ────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return a list of at most 1 Detection (the ball) or [] if not found."""
        h_orig, w_orig = frame.shape[:2]
        tensor = self._preprocess(frame)  # (3, H, W) float32 on CPU
        self._buf.append(tensor)

        if len(self._buf) < 3:
            return []  # warm-up: need 3 frames

        # Stack → (1, 9, H, W)
        stacked = torch.cat(list(self._buf), dim=0).unsqueeze(0).to(self._device)

        with torch.no_grad():
            out = self._model(stacked)  # (1, 3, H, W)
            # Channel 2 = heatmap for the 3rd (most recent) frame
            heatmap = out[0, 2]  # (H, W)

        return self._postprocess(heatmap, w_orig, h_orig)

    def reset(self) -> None:
        """Clear the frame buffer (e.g. after a stream discontinuity)."""
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
        """Find peak in heatmap; scale back to original frame coordinates."""
        hm = heatmap.cpu().numpy()                    # (H, W) float32
        peak = float(hm.max())
        if peak < self._conf_thr:
            logger.debug(f"TrackNet: peak={peak:.3f} < thr={self._conf_thr:.3f} → no ball")
            return []

        ym, xm = np.unravel_index(hm.argmax(), hm.shape)  # row, col

        # Scale from inference resolution back to original
        cx = float(xm) * w_orig / self._w
        cy = float(ym) * h_orig / self._h

        # Estimate radius as ~2% of the shorter dimension
        r = min(w_orig, h_orig) * 0.02
        r = max(r, 4.0)  # at least 4px

        logger.debug(f"TrackNet: ball at ({cx:.1f},{cy:.1f})  peak={peak:.3f}  r={r:.1f}")

        return [Detection(
            bbox=(cx - r, cy - r, cx + r, cy + r),
            conf=peak,
            class_id=_COCO_SPORTS_BALL,
        )]
