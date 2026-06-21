"""Court calibration: filter detections to the volleyball court region.

Usage:
  1. Specify the 4 court corners in image pixel coordinates (clockwise from
     top-left), either in a config file or interactively with the picker script.
  2. Pass a CourtCalibration instance to the pipeline; it will filter detections
     that fall clearly outside the court.

Court geometry:
  - The 4 corners define a quadrilateral (the court surface projected into the
    camera image).
  - A homography maps between image coords and a canonical court plane
    (useful for bird's-eye visualisation and distance estimation).
  - Margin parameters allow detections slightly outside the court polygon, since:
      * the ball flies high above the net (large vertical margin)
      * players step slightly outside the boundary lines
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Real volleyball court dimensions (metres) — used for homography
_COURT_W = 18.0  # length
_COURT_H = 9.0   # width


@dataclass
class CourtCalibration:
    """4-corner court calibration in image pixel coordinates.

    corners: sequence of 4 (x, y) points, clockwise from top-left corner of
             the court as seen in the image:
               [top-left, top-right, bottom-right, bottom-left]

    player_margin: pixels outside the polygon that still count as "in court"
                   for player detections (players step outside the lines).
    ball_margin:   pixels outside the polygon for ball detections (ball flies
                   above net, outside boundary on serves, etc.).
    """

    corners: list[tuple[float, float]]
    player_margin: float = 80.0
    ball_margin: float = 200.0

    def __post_init__(self) -> None:
        if len(self.corners) != 4:
            raise ValueError("Exactly 4 court corners required")
        self._poly = np.array(self.corners, dtype=np.float32)  # (4, 2)
        self._H, _ = cv2.findHomography(
            self._poly,
            np.array([
                [0, 0],
                [_COURT_W, 0],
                [_COURT_W, _COURT_H],
                [0, _COURT_H],
            ], dtype=np.float32),
        )
        logger.info(f"Court calibration loaded. Corners: {self.corners}")

    # ── Filtering ─────────────────────────────────────────────────────────────

    def _point_in_court(self, x: float, y: float, margin: float) -> bool:
        """Return True if (x,y) is within `margin` pixels of the court polygon."""
        dist = cv2.pointPolygonTest(self._poly, (float(x), float(y)), measureDist=True)
        return dist >= -margin

    def filter_players(self, bboxes: list[tuple]) -> list[bool]:
        """Return mask: True = detection is within player margin of court."""
        result = []
        for x1, y1, x2, y2 in bboxes:
            # Use the bottom-centre of the bounding box as the foot position
            foot_x = (x1 + x2) / 2
            foot_y = y2
            result.append(self._point_in_court(foot_x, foot_y, self.player_margin))
        return result

    def foot_distance(self, bbox: tuple) -> float:
        """Signed distance (px) from the player's foot to the court polygon.

        Positive = inside, negative = outside. Used for on-court vs off-court
        role classification (field players vs bench/referee on the sideline).
        """
        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2
        foot_y = y2
        return float(cv2.pointPolygonTest(self._poly, (foot_x, foot_y), measureDist=True))

    def is_on_court(self, bbox: tuple, margin: float | None = None) -> bool:
        """True if the player's foot is within `margin` px of the court polygon."""
        m = self.player_margin if margin is None else margin
        return self.foot_distance(bbox) >= -m

    def filter_ball(self, bboxes: list[tuple]) -> list[bool]:
        """Return mask: True = detection is within ball margin of court."""
        result = []
        for x1, y1, x2, y2 in bboxes:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            result.append(self._point_in_court(cx, cy, self.ball_margin))
        return result

    def off_court_zone(self, bbox: tuple, margin_m: float = 0.5) -> str:
        """Classify a foot's position relative to the court, in court metres.

        Court metres (from the 4 corners [TL,TR,BR,BL]):
          u in [0,18] = length (END lines at u=0 / u=18 = TL-BL / TR-BR),
          v in [0,9]  = width  (SIDE lines at v=0 / v=9 = TL-TR / BL-BR).

        Returns:
          "on"       inside the court (within margin_m of the lines).
          "endline"  clearly past an END line but BETWEEN the sidelines — the
                     serving zone behind TL-BL / TR-BR. A person here is very
                     likely a PLAYER (a server), per the user's spatial prior.
          "sideline" clearly past a SIDE line but BETWEEN the end lines — above
                     TL-TR / below BL-BR, where referees / line judges / the
                     bench stand. A person here is very likely a NON-player.
          "corner"   past BOTH (the corner region): ambiguous — a line judge may
                     stand here, so leave the decision to other signals.
        """
        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2
        foot_y = y2
        try:
            u, v = self.image_to_court(foot_x, foot_y)
        except Exception:  # pragma: no cover - degenerate homography
            return "corner"
        u_out = u < -margin_m or u > _COURT_W + margin_m
        v_out = v < -margin_m or v > _COURT_H + margin_m
        if not u_out and not v_out:
            return "on"
        if u_out and not v_out:
            return "endline"
        if v_out and not u_out:
            return "sideline"
        return "corner"

    # ── Coordinate transforms ─────────────────────────────────────────────────

    def image_to_court(self, x: float, y: float) -> tuple[float, float]:
        """Map image pixel (x, y) → court plane metres (u, v)."""
        pt = np.array([[[x, y]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pt, self._H)
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])

    # ── Visualisation ─────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray) -> None:
        """Draw court boundary and corner labels on frame in-place."""
        pts = self._poly.astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        labels = ["TL", "TR", "BR", "BL"]
        for pt, label in zip(pts, labels):
            cv2.circle(frame, tuple(pt), 6, (0, 255, 255), -1)
            cv2.putText(frame, label, (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        data = {
            "corners": self.corners,
            "player_margin": self.player_margin,
            "ball_margin": self.ball_margin,
        }
        Path(path).write_text(json.dumps(data, indent=2))
        logger.info(f"Court calibration saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "CourtCalibration":
        data = json.loads(Path(path).read_text())
        return cls(
            corners=[tuple(c) for c in data["corners"]],
            player_margin=data.get("player_margin", 80.0),
            ball_margin=data.get("ball_margin", 200.0),
        )
