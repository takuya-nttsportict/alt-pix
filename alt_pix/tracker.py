"""Multi-object tracking via boxmot (StrongSORT / ByteTrack).

boxmot (Apache 2.0) bundles StrongSORT, ByteTrack, OC-SORT, etc. with
OSNet ReID models.  Install: pip install boxmot

StrongSORT is preferred for volleyball because appearance features (ReID)
maintain player IDs through brief occlusions (net, other players).
ByteTrack is offered as a faster fallback.
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


def _make_tracker(method: str, reid_model: str, device: str):
    """Instantiate a boxmot tracker, handling API differences across versions."""
    try:
        # boxmot >= 10.x: create_tracker factory is the recommended API
        from boxmot import create_tracker
        tracker = create_tracker(
            tracker_type=method,
            tracker_config=None,       # use built-in defaults
            reid_weights=Path(reid_model),
            device=device,
            half=True,
            per_class=False,
        )
        return tracker
    except (ImportError, TypeError):
        pass

    # Fallback: try direct class import (older boxmot versions)
    try:
        if method == "strongsort":
            from boxmot import StrongSort as TrackerCls  # noqa: N812
        else:
            from boxmot import ByteTrack as TrackerCls   # noqa: N812

        # Older API may not accept reid_weights / half
        try:
            return TrackerCls(reid_weights=Path(reid_model), device=device, half=True)
        except TypeError:
            return TrackerCls(model_weights=Path(reid_model), device=device, fp16=True)
    except ImportError:
        pass

    raise ImportError(
        "Could not initialise boxmot tracker. "
        "Run: pip install boxmot"
    )


class PlayerTracker:
    """Wraps boxmot tracker for person tracking.

    Args:
        method: 'strongsort' (default) or 'bytetrack'.
        reid_model: OSNet weights name or path ('osnet_x0_25', 'osnet_ain_x1_0').
                    Downloaded automatically by boxmot on first use.
        device: 'cuda:0' or 'cpu'.
    """

    def __init__(
        self,
        method: Literal["strongsort", "bytetrack"] = "strongsort",
        reid_model: str = "osnet_x0_25",
        device: str = "cuda:0",
    ) -> None:
        self._tracker = _make_tracker(method, reid_model, device)

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Update tracker with new detections; return active tracks."""
        dets = _dets_to_boxmot_input(
            [d for d in detections if d.class_id == _COCO_PERSON]
        )
        # boxmot returns (N, 7): x1, y1, x2, y2, track_id, conf, cls
        raw = self._tracker.update(dets, frame)
        if raw is None or len(raw) == 0:
            return []
        tracks = []
        for row in raw:
            x1, y1, x2, y2, tid, conf, cls = row[:7]
            tracks.append(Track(
                track_id=int(tid),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                conf=float(conf),
                class_id=int(cls),
            ))
        return tracks
