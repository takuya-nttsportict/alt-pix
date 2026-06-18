"""Multi-object tracking via boxmot (StrongSORT / ByteTrack).

boxmot (Apache 2.0) bundles StrongSORT, ByteTrack, OC-SORT, etc. with
OSNet ReID models.  Install: pip install boxmot

StrongSORT is preferred for volleyball because appearance features (ReID)
maintain player IDs through brief occlusions (net, other players).
ByteTrack is offered as a faster fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    jersey_number: str | None = None  # filled in by OCR stage


def _dets_to_boxmot_input(detections: list[Detection]) -> np.ndarray:
    """Convert Detection list → (N, 6) float32 array [x1,y1,x2,y2,conf,cls]."""
    if not detections:
        return np.empty((0, 6), dtype=np.float32)
    rows = [
        [*d.bbox, d.conf, d.class_id] for d in detections
    ]
    return np.array(rows, dtype=np.float32)


class PlayerTracker:
    """Wraps boxmot tracker for person tracking.

    Args:
        method: 'strongsort' (default) or 'bytetrack'.
        reid_model: OSNet weights path or name ('osnet_x0_25', 'osnet_ain_x1_0').
                    Only used by StrongSORT.
        device: 'cuda:0' or 'cpu'.
    """

    def __init__(
        self,
        method: Literal["strongsort", "bytetrack"] = "strongsort",
        reid_model: str = "osnet_x0_25",
        device: str = "cuda:0",
    ) -> None:
        try:
            from boxmot import StrongSORT, ByteTrack
        except ImportError as e:
            raise ImportError(
                "Install boxmot: pip install boxmot"
            ) from e

        if method == "strongsort":
            self._tracker = StrongSORT(
                model_weights=Path(reid_model),
                device=device,
                fp16=True,
            )
        else:
            self._tracker = ByteTrack(
                device=device,
                fp16=True,
            )

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Update tracker with new detections; return active tracks."""
        dets = _dets_to_boxmot_input(
            [d for d in detections if d.class_id == _COCO_PERSON]
        )
        # boxmot returns (N, 7): x1, y1, x2, y2, track_id, conf, cls
        raw = self._tracker.update(dets, frame)
        tracks = []
        for row in raw:
            x1, y1, x2, y2, tid, conf, cls = row[:7]
            tracks.append(
                Track(
                    track_id=int(tid),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    conf=float(conf),
                    class_id=int(cls),
                )
            )
        return tracks
