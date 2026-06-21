"""Multi-object tracking via Roboflow trackers (Apache 2.0).

Uses `trackers.ByteTrackTracker` + `supervision.Detections` (numpy 2.x compatible).

Input  : list[Detection] from detector.py + BGR frame
Output : list[Track] with stable track IDs

Install: pip install trackers supervision
"""

from __future__ import annotations

from dataclasses import dataclass
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
    team: int | None = None      # 0/1 from TeamClassifier, None until ready
    role: str | None = None      # "field" / "bench" / "referee" / "off" (Phase 4)
    team_reason: str | None = None  # why this team was assigned (explainability)
    role_reason: str | None = None  # why this role was assigned (explainability)


def _dets_to_sv(detections: list[Detection]):
    """Convert Detection list → supervision.Detections."""
    import supervision as sv

    if not detections:
        return sv.Detections.empty()

    xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
    confidence = np.array([d.conf for d in detections], dtype=np.float32)
    class_id = np.array([d.class_id for d in detections], dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


class PlayerTracker:
    """Wraps ByteTrackTracker for person tracking.

    Args:
        method: Currently only 'bytetrack' is supported via the trackers package.
                'strongsort' will be added when an appearance-based tracker
                with numpy 2.x support becomes available.
        track_activation_threshold: Min detection confidence to start a new track.
        lost_track_buffer: Frames to keep a lost track alive (covers brief occlusions).
        minimum_consecutive_frames: Frames required to confirm a new track.
        fps: Source video / stream frame rate (used for time-scaling).
    """

    def __init__(
        self,
        method: Literal["bytetrack"] = "bytetrack",
        track_activation_threshold: float = 0.5,
        lost_track_buffer: int = 30,
        minimum_consecutive_frames: int = 2,
        fps: float = 30.0,
    ) -> None:
        from trackers import ByteTrackTracker

        self._tracker = ByteTrackTracker(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_consecutive_frames=minimum_consecutive_frames,
            frame_rate=fps,
        )

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Update tracker with new detections; return confirmed active tracks.

        tracker_id == -1 means the track is not yet confirmed
        (fewer than minimum_consecutive_frames seen); those are filtered out.
        """
        person_dets = [d for d in detections if d.class_id == _COCO_PERSON]
        sv_dets = _dets_to_sv(person_dets)

        tracked = self._tracker.update(sv_dets)

        tracks = []
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            if tid < 0:
                continue  # unconfirmed track
            x1, y1, x2, y2 = tracked.xyxy[i]
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            cls = int(tracked.class_id[i]) if tracked.class_id is not None else _COCO_PERSON
            tracks.append(Track(
                track_id=tid,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                conf=conf,
                class_id=cls,
            ))
        return tracks
