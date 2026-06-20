"""Multi-object tracking via boxmot v19+ (Apache 2.0).

boxmot 19.x uses the unified `Boxmot` class instead of per-tracker classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .detector import Detection, _COCO_PERSON


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    conf: float
    class_id: int
    jersey_number: str | None = None


def _dets_to_boxmot_input(detections: list[Detection]) -> np.ndarray:
    """Convert Detection list → (N, 6) float32 array [x1,y1,x2,y2,conf,cls]."""
    if not detections:
        return np.empty((0, 6), dtype=np.float32)
    return np.array([[*d.bbox, d.conf, d.class_id] for d in detections], dtype=np.float32)


class PlayerTracker:
    """Wraps boxmot Boxmot class for person tracking (boxmot v19+).

    Args:
        method: tracker name, e.g. 'strongsort', 'bytetrack', 'ocsort'.
        reid_model: OSNet weights name ('osnet_x0_25') or local .pt path.
        device: 'cuda:0' or 'cpu'.
    """

    def __init__(
        self,
        method: Literal["strongsort", "bytetrack", "ocsort", "botsort"] = "strongsort",
        reid_model: str = "osnet_x0_25",
        device: str = "cuda:0",
    ) -> None:
        import torch
        from boxmot import Boxmot

        reid_path = Path(reid_model)
        # boxmot auto-downloads weights if only a name (no suffix) is given
        if not reid_path.suffix:
            reid_path = Path(reid_model + ".pt")

        self._tracker = Boxmot(
            method=method,
            reid_weights=reid_path,
            device=torch.device(device),
            half=device.startswith("cuda"),
            per_class=False,
        )

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Update tracker with new detections; return active tracks."""
        dets = _dets_to_boxmot_input(
            [d for d in detections if d.class_id == _COCO_PERSON]
        )
        # boxmot returns (N, 7): x1, y1, x2, y2, track_id, conf, cls
        raw = self._tracker.update(dets, frame)
        if raw is None or len(raw) == 0:
            return []
        return [
            Track(
                track_id=int(row[4]),
                bbox=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                conf=float(row[5]),
                class_id=int(row[6]),
            )
            for row in raw
        ]
