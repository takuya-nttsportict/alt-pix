"""Jersey number recognition via EasyOCR (Apache 2.0).

Pipeline:
  1. Crop the upper-body region from each player bounding box.
  2. Enhance contrast (CLAHE) and resize for OCR.
  3. Run EasyOCR digit recognition.
  4. Filter results to 0–99 (valid volleyball jersey numbers).
  5. Accumulate per-track ID votes; majority across N frames wins.

Install: pip install easyocr
"""

from __future__ import annotations

import re
from collections import Counter

import cv2
import numpy as np

from .tracker import Track

_JERSEY_RE = re.compile(r"^\d{1,2}$")
_VOTE_WINDOW = 15  # frames to collect votes per track


def _crop_upper_body(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    """Crop the top ~60% of a player bounding box (torso / jersey area)."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h = y2 - y1
    y2_crop = y1 + int(h * 0.60)
    crop = frame[max(0, y1) : y2_crop, max(0, x1) : x2]
    if crop.size == 0:
        return None
    return crop


def _enhance(img: np.ndarray) -> np.ndarray:
    """Grayscale + CLAHE contrast enhancement → RGB for EasyOCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    target_h = 64
    scale = target_h / max(enhanced.shape[0], 1)
    resized = cv2.resize(
        enhanced,
        (max(1, int(enhanced.shape[1] * scale)), target_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


class JerseyOCR:
    """Per-track jersey number recognizer with frame-level voting.

    Args:
        use_gpu: Whether to use GPU for EasyOCR.
        vote_window: Number of recent frames to aggregate votes from.
    """

    def __init__(self, use_gpu: bool = True, vote_window: int = _VOTE_WINDOW) -> None:
        try:
            import easyocr
        except ImportError as e:
            raise ImportError("Install EasyOCR: pip install easyocr") from e

        self._reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        self._vote_window = vote_window
        self._votes: dict[int, list[str]] = {}

    def _ocr_crop(self, crop: np.ndarray) -> str | None:
        results = self._reader.readtext(crop, detail=1, allowlist="0123456789")
        for (_bbox, text, conf) in results:
            text = text.strip().lstrip("0") or "0"
            if conf >= 0.5 and _JERSEY_RE.match(text) and 0 <= int(text) <= 99:
                return text
        return None

    def update(self, frame: np.ndarray, tracks: list[Track]) -> dict[int, str]:
        """Run OCR on new frame; return {track_id: jersey_number} for all known tracks."""
        for track in tracks:
            crop = _crop_upper_body(frame, track.bbox)
            if crop is None:
                continue
            enhanced = _enhance(crop)
            number = self._ocr_crop(enhanced)
            if number is not None:
                buf = self._votes.setdefault(track.track_id, [])
                buf.append(number)
                if len(buf) > self._vote_window:
                    buf.pop(0)

        result: dict[int, str] = {}
        for tid, buf in self._votes.items():
            if buf:
                winner, _ = Counter(buf).most_common(1)[0]
                result[tid] = winner
        return result
