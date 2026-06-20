"""TrackNetV2 ball detector.

Exact architecture from nttcom/WASB-SBDT (MIT licence, BMVC 2023):
  src/models/unet2d.py + src/models/unet2d_parts.py

State-dict key mapping verified against volleyball checkpoint:
  inc.double_conv.{0,2,3,5}.*        DoubleConv: Conv→ReLU→BN→Conv→ReLU→BN
  down1.maxpool_conv.1.double_conv.*  Down2: MaxPool + DoubleConv
  down2/3.maxpool_conv.1.triple_conv.{0,2,3,5,6,8}.*  Down3: MaxPool + TripleConv
  up1.conv.triple_conv.*  Up3
  up2/3.conv.double_conv.*  Up2
  outc.conv.*

Input : (B, 9, H, W)  — 3 consecutive BGR frames, each normalised to [0,1]
Output: (B, 3, H, W)  — raw logits, one heatmap per input frame
        sigmoid applied in post-processing; channel 2 = most recent frame.
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

_W = 512
_H = 288

# WASB-SBDT preprocessing: ToTensor (÷255) → Normalize(ImageNet mean/std) on RGB.
# Verified against src/dataloaders/__init__.py build_img_transforms().
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Affine warp helpers (faithful copy of WASB src/utils/image.py) ───────────
# WASB maps a centred square region (side = max(h, w)) of the source frame into
# the (W, H) input via an affine transform, then maps heatmap peaks back with
# the inverse.  We replicate exactly so coordinates match the training domain.

def _get_dir(src_point: list[float], rot_rad: float) -> list[float]:
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [src_point[0] * cs - src_point[1] * sn,
            src_point[0] * sn + src_point[1] * cs]


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_affine_transform(center, scale, output_size, inv: int = 0) -> np.ndarray:
    scale = np.array([scale, scale], dtype=np.float32)
    src_w = scale[0]
    dst_w, dst_h = output_size

    src_dir = _get_dir([0, src_w * -0.5], 0.0)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _affine_point(pt: tuple[float, float], t: np.ndarray) -> tuple[float, float]:
    v = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
    out = t @ v
    return float(out[0]), float(out[1])


# ── Building blocks ───────────────────────────────────────────────────────────

class _DoubleConv(nn.Module):
    """Conv→ReLU→BN→Conv→ReLU→BN. Attr name must be double_conv."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),  # [0]
            nn.ReLU(inplace=True),                               # [1]
            nn.BatchNorm2d(out_ch),                              # [2]
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True), # [3]
            nn.ReLU(inplace=True),                               # [4]
            nn.BatchNorm2d(out_ch),                              # [5]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class _TripleConv(nn.Module):
    """Conv→ReLU→BN ×3. Attr name must be triple_conv."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),  # [0]
            nn.ReLU(inplace=True),                               # [1]
            nn.BatchNorm2d(out_ch),                              # [2]
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True), # [3]
            nn.ReLU(inplace=True),                               # [4]
            nn.BatchNorm2d(out_ch),                              # [5]
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True), # [6]
            nn.ReLU(inplace=True),                               # [7]
            nn.BatchNorm2d(out_ch),                              # [8]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.triple_conv(x)


class _Down2(nn.Module):
    """MaxPool + DoubleConv. Attr name must be maxpool_conv."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class _Down3(nn.Module):
    """MaxPool + TripleConv. Attr name must be maxpool_conv."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), _TripleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class _Up2(nn.Module):
    """Upsample + cat skip + DoubleConv.

    Concatenation order matches WASB Up.forward: torch.cat([skip, upsampled]).
    The trained weights expect skip channels first.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = _DoubleConv(skip_ch + in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([skip, self.up(x)], dim=1))


class _Up3(nn.Module):
    """Upsample + cat skip + TripleConv.

    Concatenation order matches WASB Up.forward: torch.cat([skip, upsampled]).
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = _TripleConv(skip_ch + in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([skip, self.up(x)], dim=1))


class _OutConv(nn.Module):
    """1×1 conv. Attr name must be conv."""
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


# ── Inference wrapper ─────────────────────────────────────────────────────────

class TrackNetDetector:
    """Frame-by-frame ball detector using TrackNetV2 (WASB-SBDT).

    Call detect(frame) once per frame.  Maintains an internal 3-frame buffer;
    returns [] for the first 2 frames while the buffer fills.

    Args:
        model_path : Path to .pt checkpoint (state dict or model_state_dict wrapper).
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
        ckpt = torch.load(str(model_path), map_location=self._device, weights_only=False)
        if isinstance(ckpt, dict):
            state = ckpt.get("model_state_dict", ckpt.get("model", ckpt.get("state_dict", ckpt)))
        else:
            state = ckpt
        state = {(k[len("module."):] if k.startswith("module.") else k): v
                 for k, v in state.items()}
        self._model.load_state_dict(state, strict=True)
        self._model.eval().to(self._device)

        self._buf: deque[torch.Tensor] = deque(maxlen=3)

        # Affine transforms are derived lazily once the first frame size is known.
        self._trans: np.ndarray | None = None       # original → (W, H)
        self._trans_inv: np.ndarray | None = None    # (W, H) → original
        self._src_hw: tuple[int, int] | None = None

        logger.info(
            f"TrackNetV2 loaded: {model_path}  "
            f"device={self._device}  input={self._w}×{self._h}  conf_thr={conf_thr}"
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return a list of at most 1 Detection (the ball) or [] if not found."""
        h_orig, w_orig = frame.shape[:2]
        self._ensure_transforms(h_orig, w_orig)
        self._buf.append(self._preprocess(frame))

        if len(self._buf) < 3:
            return []

        stacked = torch.cat(list(self._buf), dim=0).unsqueeze(0).to(self._device)  # (1, 9, H, W)

        with torch.no_grad():
            logits = self._model(stacked)[0]          # (1, 3, H, W)
            heatmap = torch.sigmoid(logits[0, 2])     # (H, W) — most recent frame

        return self._postprocess(heatmap)

    def reset(self) -> None:
        """Clear the frame buffer (call after stream discontinuity)."""
        self._buf.clear()

    # ── private ───────────────────────────────────────────────────────────────

    def _ensure_transforms(self, h: int, w: int) -> None:
        """Build the affine warp + inverse for the current frame size (WASB).

        WASB uses center=(w/2,h/2), scale=max(h,w); it maps a centred square
        region into (W, H).  Cached until the frame size changes.
        """
        if self._src_hw == (h, w):
            return
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        scale = float(max(h, w))
        self._trans = _get_affine_transform(center, scale, (self._w, self._h), inv=0)
        self._trans_inv = _get_affine_transform(center, scale, (self._w, self._h), inv=1)
        self._src_hw = (h, w)
        logger.debug(f"TrackNet: affine transform built for frame {w}×{h}")

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """BGR→RGB, affine warp to (W, H), ÷255, ImageNet normalise. Returns (3, H, W)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        warped = cv2.warpAffine(rgb, self._trans, (self._w, self._h),
                                flags=cv2.INTER_LINEAR)            # (H, W, 3) uint8 RGB
        t = warped.astype(np.float32) / 255.0
        t = (t - _MEAN) / _STD                                     # ImageNet normalise
        return torch.from_numpy(t).permute(2, 0, 1).contiguous()   # (3, H, W)

    def _postprocess(self, heatmap: torch.Tensor) -> list[Detection]:
        hm = heatmap.cpu().numpy()
        peak = float(hm.max())

        if peak < self._conf_thr:
            logger.debug(f"TrackNet: peak={peak:.3f} < thr={self._conf_thr} → no ball")
            return []

        ym, xm = np.unravel_index(hm.argmax(), hm.shape)

        # Map peak from (W, H) heatmap space back to original frame via inverse affine.
        cx, cy = _affine_point((float(xm), float(ym)), self._trans_inv)

        w_orig = self._src_hw[1]
        h_orig = self._src_hw[0]
        r = max(min(w_orig, h_orig) * 0.02, 4.0)

        logger.debug(f"TrackNet: ball=({cx:.1f},{cy:.1f})  peak={peak:.3f}  r={r:.1f}")

        return [Detection(
            bbox=(cx - r, cy - r, cx + r, cy + r),
            conf=peak,
            class_id=_COCO_SPORTS_BALL,
        )]
