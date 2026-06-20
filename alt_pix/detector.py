"""YOLOX ONNX detector with correct grid-based output decoding.

YOLOX ONNX exports raw grid predictions (not decoded boxes).
Official decoding logic: demo/ONNXRuntime/onnx_inference.py in YOLOX repo.

Output shape: (1, num_anchors, 5 + num_classes)
  - num_anchors = H/8 * W/8 + H/16 * W/16 + H/32 * W/32  (e.g. 8400 for 640x640)
  - [:, :, 0:2] = raw dx, dy offset within grid cell  → decoded to cx, cy
  - [:, :, 2:4] = raw log(w), log(h)                  → decoded to w, h
  - [:, :, 4]   = objectness confidence
  - [:, :, 5:]  = per-class scores
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_COCO_PERSON = 0
_COCO_SPORTS_BALL = 32


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in original pixel coords
    conf: float
    class_id: int


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _letterbox(img: np.ndarray, target: int = 640) -> tuple[np.ndarray, float, int, int]:
    """Resize with padding to (target x target). Returns (padded, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad = np.full((target, target, 3), 114, dtype=np.uint8)
    pad_x = (target - nw) // 2
    pad_y = (target - nh) // 2
    pad[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    blob = pad.astype(np.float32).transpose(2, 0, 1)[np.newaxis]  # NCHW
    return blob, scale, pad_x, pad_y


# ── YOLOX grid decoding ───────────────────────────────────────────────────────

def _decode_yolox(raw: np.ndarray, input_size: int = 640) -> np.ndarray:
    """Decode YOLOX raw output using grid offsets and strides.

    Args:
        raw: shape (1, num_anchors, 5 + num_classes) — raw network output
        input_size: square input resolution used during inference

    Returns:
        decoded: same shape; [:, :, 0:4] are now (cx, cy, w, h) in input-image pixels
    """
    out = raw[0].copy()  # (num_anchors, 5 + C)
    grids = []
    strides_list = []
    for stride in (8, 16, 32):
        gs = input_size // stride
        xv, yv = np.meshgrid(np.arange(gs), np.arange(gs))
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2)  # (gs*gs, 2)
        grids.append(grid)
        strides_list.append(np.full((gs * gs, 1), stride, dtype=np.float32))

    grids_all = np.concatenate(grids, axis=0)          # (num_anchors, 2)
    strides_all = np.concatenate(strides_list, axis=0) # (num_anchors, 1)

    out[:, :2] = (out[:, :2] + grids_all) * strides_all   # cx, cy
    out[:, 2:4] = np.exp(out[:, 2:4]) * strides_all       # w, h
    return out


# ── Postprocessing ────────────────────────────────────────────────────────────

def _postprocess(
    raw: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    conf_thr: float,
    class_ids: set[int],
    input_size: int = 640,
) -> list[Detection]:
    decoded = _decode_yolox(raw, input_size)          # (N, 5+C)

    obj_conf = decoded[:, 4]
    cls_scores = decoded[:, 5:]
    cls_ids = cls_scores.argmax(axis=1)
    cls_conf = cls_scores[np.arange(len(cls_ids)), cls_ids]
    scores = obj_conf * cls_conf

    mask = (scores >= conf_thr) & np.isin(cls_ids, list(class_ids))
    if not mask.any():
        return []

    d = decoded[mask]
    s = scores[mask]
    c = cls_ids[mask]

    cx, cy, bw, bh = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    x1 = (cx - bw / 2 - pad_x) / scale
    y1 = (cy - bh / 2 - pad_y) / scale
    x2 = (cx + bw / 2 - pad_x) / scale
    y2 = (cy + bh / 2 - pad_y) / scale

    # NMS per class
    detections: list[Detection] = []
    for cls in set(c.tolist()):
        cmask = c == cls
        boxes_xywh = np.stack([x1[cmask], y1[cmask],
                                x2[cmask] - x1[cmask],
                                y2[cmask] - y1[cmask]], axis=1).tolist()
        sc = s[cmask].tolist()
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, sc, conf_thr, 0.45)
        if len(idxs) == 0:
            continue
        ci_list = np.where(cmask)[0]
        for i in np.array(idxs).flatten():
            gi = ci_list[i]
            detections.append(Detection(
                bbox=(float(x1[gi]), float(y1[gi]), float(x2[gi]), float(y2[gi])),
                conf=float(s[gi]),
                class_id=int(cls),
            ))
    return detections


# ── Detector class ────────────────────────────────────────────────────────────

class YOLOXDetector:
    """YOLOX ONNX inference wrapper.

    Args:
        model_path: Path to .onnx file exported from YOLOX.
        input_size: Square resolution the model was trained/exported at (default 640).
        conf_thr: Minimum objectness × class confidence.
        detect_classes: Set of COCO class IDs to return.
        device: 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        conf_thr: float = 0.4,
        detect_classes: set[int] | None = None,
        device: str = "cuda",
    ) -> None:
        self._input_size = input_size
        self._conf_thr = conf_thr
        self._classes = detect_classes or {_COCO_PERSON}

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        out_shape = self._session.get_outputs()[0].shape
        logger.debug(f"YOLOX model loaded: {model_path}  output shape: {out_shape}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        blob, scale, pad_x, pad_y = _letterbox(frame, self._input_size)
        raw = self._session.run(None, {self._input_name: blob})
        dets = _postprocess(raw[0], scale, pad_x, pad_y,
                            self._conf_thr, self._classes, self._input_size)
        logger.debug(f"detect: {len(dets)} detections (classes={self._classes})")
        return dets

    @property
    def person_class(self) -> int:
        return _COCO_PERSON

    @property
    def ball_class(self) -> int:
        return _COCO_SPORTS_BALL
