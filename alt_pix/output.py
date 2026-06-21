"""JSON log output and optional annotated video writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np

from .ball_tracker import BallState
from .framing import ROI
from .tracker import Track


def _make_record(
    frame_id: int,
    timestamp_ms: float,
    ball: BallState,
    tracks: list[Track],
    jersey_map: dict[int, str],
    roi: ROI | None,
) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp_ms": round(timestamp_ms, 1),
        "ball": {
            "x": round(ball.x, 1),
            "y": round(ball.y, 1),
            "conf": round(ball.conf, 3),
            "visible": ball.visible,
            "predicted": ball.predicted,
            "smoothed": ball.smoothed,
        },
        "players": [
            {
                "track_id": t.track_id,
                "jersey": jersey_map.get(t.track_id),
                "team": t.team,
                "role": t.role,
                "bbox": [round(v, 1) for v in t.bbox],
                "conf": round(t.conf, 3),
            }
            for t in tracks
        ],
        "framing_roi": (
            {"x": roi.x, "y": roi.y, "w": roi.w, "h": roi.h} if roi else None
        ),
    }


class JSONLWriter:
    """Append-mode JSONL writer (one JSON object per line)."""

    def __init__(self, path: str | Path) -> None:
        self._f: TextIO = open(path, "w", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class VideoWriter:
    """Wraps cv2.VideoWriter for annotated output."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._w = cv2.VideoWriter(str(path), fourcc, fps, size)

    def write(self, frame: np.ndarray) -> None:
        self._w.write(frame)

    def close(self) -> None:
        self._w.release()
