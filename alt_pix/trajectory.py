"""
Parabolic trajectory smoothing for ball tracking.

After the Kalman filter provides frame-by-frame estimates, we fit a
2-D parabola (physical model for ballistic flight) over a sliding
window of N recent ball positions.  This suppresses jitter and gives
a physics-consistent interpolation through occlusion gaps.

The parabola fit is used only for smoothing visualization and for
computing the framing ROI; the raw Kalman estimate is still written
to the JSON log.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Point2D:
    frame_id: int
    x: float
    y: float


class ParabolicSmoother:
    """Fit a parabola over recent ball positions and return a smoothed point.

    Args:
        window: Number of recent frames to fit over.
        min_points: Minimum points required before fitting (otherwise passthrough).
    """

    def __init__(self, window: int = 15, min_points: int = 5) -> None:
        self._window = window
        self._min = min_points
        self._buf: deque[Point2D] = deque(maxlen=window)

    def update(self, frame_id: int, x: float, y: float) -> tuple[float, float]:
        """Add a new point and return the smoothed (x, y) for this frame."""
        self._buf.append(Point2D(frame_id, x, y))

        if len(self._buf) < self._min:
            return x, y

        fids = np.array([p.frame_id for p in self._buf], dtype=np.float64)
        xs = np.array([p.x for p in self._buf], dtype=np.float64)
        ys = np.array([p.y for p in self._buf], dtype=np.float64)

        # Normalise frame ids to avoid ill-conditioned Vandermonde
        t = fids - fids[-1]  # current frame at t=0

        try:
            cx = np.polyfit(t, xs, 1)   # x is roughly linear (constant velocity)
            cy = np.polyfit(t, ys, 2)   # y is parabolic (gravity)
        except np.linalg.LinAlgError:
            return x, y

        sx = float(np.polyval(cx, 0.0))
        sy = float(np.polyval(cy, 0.0))
        return sx, sy

    def predict(self, future_frames: int = 1) -> tuple[float, float] | None:
        """Extrapolate ball position N frames ahead using the fitted parabola.

        Returns None if not enough history to fit.
        """
        if len(self._buf) < self._min:
            return None

        fids = np.array([p.frame_id for p in self._buf], dtype=np.float64)
        xs = np.array([p.x for p in self._buf], dtype=np.float64)
        ys = np.array([p.y for p in self._buf], dtype=np.float64)
        t = fids - fids[-1]

        try:
            cx = np.polyfit(t, xs, 1)
            cy = np.polyfit(t, ys, 2)
        except np.linalg.LinAlgError:
            return None

        sx = float(np.polyval(cx, float(future_frames)))
        sy = float(np.polyval(cy, float(future_frames)))
        return sx, sy
