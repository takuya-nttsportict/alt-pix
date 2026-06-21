"""Debug visualization: draw BBoxes, track IDs, jersey numbers, ball, and ROI."""

from __future__ import annotations

import cv2
import numpy as np

from .ball_tracker import BallState
from .framing import ROI
from .tracker import Track

_PALETTE = [
    (0, 255, 0),    # green
    (255, 128, 0),  # orange
    (0, 200, 255),  # cyan
    (200, 0, 255),  # purple
    (255, 255, 0),  # yellow
]


def _color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


# Team / role colours (BGR). Teams get two distinct hues; off-court roles are
# de-emphasised so the eye stays on the field players (framing-relevant ones).
_TEAM_COLOR = {0: (0, 200, 255), 1: (255, 128, 0)}   # team 0 = amber, team 1 = blue
_ROLE_COLOR = {"referee": (0, 255, 255), "bench": (140, 140, 140), "off": (110, 110, 110)}


def _track_color(track: Track) -> tuple[int, int, int]:
    """Colour a track by role first (referee/bench/off), else by team, else id."""
    if track.role in _ROLE_COLOR:
        return _ROLE_COLOR[track.role]
    if track.team in _TEAM_COLOR:
        return _TEAM_COLOR[track.team]
    return _color(track.track_id)


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    jersey_map: dict[int, str],
) -> None:
    """Draw player BBoxes, track IDs, jersey numbers, team and role in-place."""
    for track in tracks:
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        color = _track_color(track)
        # Off-court roles drawn thinner to recede visually.
        thick = 1 if track.role in ("bench", "off") else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        jersey = jersey_map.get(track.track_id, "?")
        parts = [f"[{track.track_id}]"]
        if jersey and jersey != "?":
            parts.insert(0, f"#{jersey}")
        if track.team is not None and track.team >= 0:
            parts.append(f"T{track.team}")
        if track.role and track.role != "field":
            parts.append(track.role)
        label = " ".join(parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )


def draw_ball(frame: np.ndarray, ball: BallState) -> None:
    """Draw ball center; dashed circle when predicted."""
    if not ball.visible:
        return
    cx, cy = int(ball.x), int(ball.y)
    color = (0, 0, 255) if not ball.predicted else (0, 100, 255)
    cv2.circle(frame, (cx, cy), 12, color, 2)
    cv2.circle(frame, (cx, cy), 3, color, -1)
    if ball.predicted:
        cv2.putText(
            frame, "pred", (cx + 14, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )


def draw_roi(frame: np.ndarray, roi: ROI) -> None:
    """Draw the virtual camera framing rectangle."""
    cv2.rectangle(
        frame,
        (roi.x, roi.y),
        (roi.x + roi.w, roi.y + roi.h),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def annotate(
    frame: np.ndarray,
    tracks: list[Track],
    ball: BallState,
    jersey_map: dict[int, str],
    roi: ROI | None,
) -> np.ndarray:
    """Return an annotated copy of frame."""
    out = frame.copy()
    draw_tracks(out, tracks, jersey_map)
    draw_ball(out, ball)
    if roi is not None:
        draw_roi(out, roi)
    return out
