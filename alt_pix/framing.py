"""Virtual camera framing: compute a crop ROI from ball + player positions.

The ROI is smoothed with Exponential Moving Average (EMA) to avoid
jarring camera jumps.  Two modes:
  - 'ball': tight crop centered on the ball (spike/serve focus)
  - 'wide': crop that includes all players on court (rally overview)

The caller can switch modes or blend them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ball_tracker import BallState
from .tracker import Track


@dataclass
class ROI:
    x: int  # top-left column
    y: int  # top-left row
    w: int  # width
    h: int  # height


_EMA_ALPHA = 0.15  # smoothing factor (lower = smoother but slower)
_BALL_MARGIN = 0.30  # fraction of frame size added around ball
_WIDE_MARGIN = 0.10  # fraction of player BBox union added as padding
_MIN_ROI_FRAC = 0.25  # ROI must be at least this fraction of frame


class FramingCalculator:
    """Computes per-frame virtual camera ROI.

    Args:
        frame_w: Source frame width (pixels).
        frame_h: Source frame height (pixels).
        output_aspect: Desired output aspect ratio (w/h). Default 16/9.
        mode: 'ball', 'wide', or 'auto' (switches based on ball visibility).
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        output_aspect: float = 16 / 9,
        mode: str = "auto",
    ) -> None:
        self._fw = frame_w
        self._fh = frame_h
        self._aspect = output_aspect
        self._mode = mode
        # EMA state: [cx, cy, w, h] in float
        self._ema: np.ndarray | None = None

    def _clamp_roi(self, cx: float, cy: float, w: float, h: float) -> ROI:
        """Force ROI to maintain aspect ratio, stay within frame, and respect min size."""
        # Enforce minimum size
        w = max(w, self._fw * _MIN_ROI_FRAC)
        h = max(h, self._fh * _MIN_ROI_FRAC)

        # Adjust to target aspect ratio
        if w / h > self._aspect:
            h = w / self._aspect
        else:
            w = h * self._aspect

        # Clamp to frame bounds
        w = min(w, self._fw)
        h = min(h, self._fh)
        x = int(np.clip(cx - w / 2, 0, self._fw - w))
        y = int(np.clip(cy - h / 2, 0, self._fh - h))
        return ROI(x=x, y=y, w=int(w), h=int(h))

    def _ball_roi(self, ball: BallState) -> tuple[float, float, float, float]:
        margin_w = self._fw * _BALL_MARGIN
        margin_h = self._fh * _BALL_MARGIN
        return ball.x, ball.y, margin_w * 2, margin_h * 2

    def _wide_roi(self, tracks: list[Track]) -> tuple[float, float, float, float]:
        if not tracks:
            return self._fw / 2, self._fh / 2, self._fw * 0.8, self._fh * 0.8
        x1s = [t.bbox[0] for t in tracks]
        y1s = [t.bbox[1] for t in tracks]
        x2s = [t.bbox[2] for t in tracks]
        y2s = [t.bbox[3] for t in tracks]
        bx1, by1, bx2, by2 = min(x1s), min(y1s), max(x2s), max(y2s)
        pw = (bx2 - bx1) * _WIDE_MARGIN
        ph = (by2 - by1) * _WIDE_MARGIN
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        w = (bx2 - bx1) + pw * 2
        h = (by2 - by1) + ph * 2
        return cx, cy, w, h

    def compute(self, ball: BallState, tracks: list[Track]) -> ROI:
        """Return the smoothed crop ROI for this frame."""
        if self._mode == "ball" or (self._mode == "auto" and ball.visible):
            cx, cy, w, h = self._ball_roi(ball)
        else:
            cx, cy, w, h = self._wide_roi(tracks)

        raw = np.array([cx, cy, w, h], dtype=np.float64)
        if self._ema is None:
            self._ema = raw.copy()
        else:
            self._ema = _EMA_ALPHA * raw + (1.0 - _EMA_ALPHA) * self._ema

        return self._clamp_roi(*self._ema)
