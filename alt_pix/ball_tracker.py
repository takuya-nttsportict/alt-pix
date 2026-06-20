"""Ball tracking: motion-aware candidate selection + Kalman interpolation.

The detector (TrackNet) emits MULTIPLE ball candidates per frame.  Selecting
the single highest-confidence one independently each frame makes the track
jump to spurious heatmap peaks whenever the ball is occluded or lost.

To match the accuracy of the WASB-SBDT reference (nttcom, MIT), candidate
selection uses the same OnlineTracker strategy:
  1. constant-acceleration motion prediction from the last 3 positions,
  2. gating: reject candidates farther than `max_disp` from the previous
     position,
  3. scoring: prefer candidates close to the predicted position
     (score = conf - quality_weight × dist_to_prediction / max_disp).

When the ball is not detected (fast motion, occlusion, net crossing),
a constant-velocity Kalman filter predicts the position for up to
MAX_MISS_FRAMES frames before marking the ball as lost.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .detector import Detection, _COCO_SPORTS_BALL
from .trajectory import ParabolicSmoother

MAX_MISS_FRAMES = 5  # frames to keep predicting without a detection
MAX_DISP = 300.0     # max ball displacement between frames (px); WASB default
QUALITY_WEIGHT = 0.5  # weight of prediction-proximity vs. raw confidence


def _center(det: Detection) -> np.ndarray:
    x1, y1, x2, y2 = det.bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


@dataclass
class BallState:
    x: float
    y: float
    visible: bool
    conf: float
    predicted: bool   # True when position is Kalman-predicted, not detected
    smoothed: bool = False  # True when parabolic smoothing was applied


class BallKalmanFilter:
    """2-D constant-velocity Kalman filter for ball position."""

    def __init__(self) -> None:
        # State: [x, y, vx, vy]
        self._F = np.eye(4, dtype=np.float64)
        self._F[0, 2] = self._F[1, 3] = 1.0  # position += velocity

        self._H = np.zeros((2, 4), dtype=np.float64)
        self._H[0, 0] = self._H[1, 1] = 1.0  # observe position only

        self._Q = np.diag([1.0, 1.0, 10.0, 10.0])  # process noise
        self._R = np.diag([5.0, 5.0])               # measurement noise

        self._x: np.ndarray | None = None
        self._P = np.eye(4, dtype=np.float64) * 100

    @property
    def initialized(self) -> bool:
        return self._x is not None

    def init(self, cx: float, cy: float) -> None:
        self._x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)

    def predict(self) -> tuple[float, float]:
        assert self._x is not None
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return float(self._x[0]), float(self._x[1])

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        assert self._x is not None
        z = np.array([cx, cy], dtype=np.float64)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ self._H) @ self._P
        return float(self._x[0]), float(self._x[1])


class BallTracker:
    """Combines YOLOX detections with Kalman interpolation + parabolic smoothing.

    Args:
        ball_class_id: COCO (or custom) class ID that represents the ball.
        smooth: Enable parabolic trajectory smoother on top of Kalman output.
    """

    def __init__(
        self,
        ball_class_id: int = _COCO_SPORTS_BALL,
        smooth: bool = True,
        max_disp: float = MAX_DISP,
        quality_weight: float = QUALITY_WEIGHT,
    ) -> None:
        self._cls = ball_class_id
        self._kf = BallKalmanFilter()
        self._smoother = ParabolicSmoother() if smooth else None
        self._miss = 0
        self._frame_id = 0
        self._max_disp = max_disp
        self._quality_weight = quality_weight
        # Last 3 accepted (visible) positions, newest last — for motion prediction.
        self._hist: deque[np.ndarray] = deque(maxlen=3)

    def _predict_xy(self) -> np.ndarray | None:
        """Constant-acceleration prediction from the last 3 positions (WASB)."""
        if len(self._hist) < 3:
            return None
        xy3, xy2, xy1 = self._hist[0], self._hist[1], self._hist[2]  # oldest→newest
        acc = (xy1 - xy2) - (xy2 - xy3)
        vel = (xy1 - xy2) + acc
        return xy1 + vel + acc / 2.0

    def _select_ball(self, detections: list[Detection]) -> Detection | None:
        """Pick the candidate that is both confident and motion-consistent.

        Implements WASB OnlineTracker selection: gate by max_disp from the
        previous position, then score by conf minus distance to the
        motion-predicted position.
        """
        balls = [d for d in detections if d.class_id == self._cls]
        if not balls:
            return None

        last = self._hist[-1] if self._hist else None
        xy_pred = self._predict_xy()

        # Gate: drop candidates too far from the previous accepted position.
        if last is not None:
            gated = [d for d in balls
                     if np.linalg.norm(_center(d) - last) < self._max_disp]
            if gated:
                balls = gated

        def score(d: Detection) -> float:
            s = d.conf
            if xy_pred is not None:
                dist = float(np.linalg.norm(_center(d) - xy_pred))
                s -= self._quality_weight * (dist / self._max_disp)
            return s

        return max(balls, key=score)

    def update(self, detections: list[Detection]) -> BallState:
        self._frame_id += 1
        det = self._select_ball(detections)

        if det is not None:
            x1, y1, x2, y2 = det.bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if not self._kf.initialized:
                self._kf.init(cx, cy)
                px, py = cx, cy
            else:
                self._kf.predict()
                px, py = self._kf.update(cx, cy)

            self._miss = 0
            self._hist.append(np.array([px, py], dtype=np.float64))

            if self._smoother is not None:
                sx, sy = self._smoother.update(self._frame_id, px, py)
                return BallState(x=sx, y=sy, visible=True, conf=det.conf, predicted=False, smoothed=True)

            return BallState(x=px, y=py, visible=True, conf=det.conf, predicted=False)

        # No detection — clear motion history (gap breaks the const-accel model)
        self._hist.clear()

        # Kalman predict for a few frames to bridge short gaps
        if self._kf.initialized and self._miss < MAX_MISS_FRAMES:
            px, py = self._kf.predict()
            self._miss += 1

            if self._smoother is not None:
                pred = self._smoother.predict(future_frames=self._miss)
                if pred is not None:
                    px, py = pred

            return BallState(x=px, y=py, visible=True, conf=0.0, predicted=True, smoothed=self._smoother is not None)

        return BallState(x=0.0, y=0.0, visible=False, conf=0.0, predicted=False)
