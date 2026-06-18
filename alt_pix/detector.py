"""Person and ball detection via YOLOX (Apache 2.0, Megvii).

YOLOX ONNX weights are used for inference to avoid a hard dependency on the
full YOLOX training framework.  Weights can be downloaded from:
  https://github.com/Megvii-BaseDetection/YOLOX/releases

Alternatively, the PyTorch checkpoint can be exported with:
  python -m yolox.tools.export_onnx --name yolox-m --ckpt yolox_m.pth
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel coords)
    conf: float
    class_id: int  # 0=person, 80=sports_ball (COCO)


_COCO_PERSON = 0
_COCO_SPORTS_BALL = 32  # volleyball not in COCO; fine-tune remaps to this id


def _preprocess(img: np.ndarray, input_size: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Letterbox resize + normalize to [0,1] float32 CHW."""
    h, w = img.shape[:2]
    ih, iw = input_size
    scale = min(iw / w, ih / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
    pad_y = (ih - nh) // 2
    pad_x = (iw - nw) // 2
    padded[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

    blob = padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis]  # NCHW
    return blob, scale, (pad_x, pad_y)


def _postprocess(
    output: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    conf_thr: float,
    class_ids: set[int],
) -> list[Detection]:
    """Decode YOLOX flat output [N, 5+num_classes] and apply NMS."""
    detections: list[Detection] = []
    pad_x, pad_y = pad
    # output shape: (1, num_boxes, 5 + num_classes)
    preds = output[0]  # (num_boxes, 5+C)
    obj_conf = preds[:, 4]
    cls_scores = preds[:, 5:]
    cls_ids = cls_scores.argmax(axis=1)
    cls_conf = cls_scores[np.arange(len(cls_ids)), cls_ids]
    scores = obj_conf * cls_conf

    mask = (scores >= conf_thr) & np.isin(cls_ids, list(class_ids))
    preds, scores, cls_ids = preds[mask], scores[mask], cls_ids[mask]
    if len(preds) == 0:
        return []

    # cx, cy, w, h → x1, y1, x2, y2
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = ((cx - bw / 2) - pad_x) / scale
    y1 = ((cy - bh / 2) - pad_y) / scale
    x2 = ((cx + bw / 2) - pad_x) / scale
    y2 = ((cy + bh / 2) - pad_y) / scale

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), conf_thr, iou_threshold=0.45
    )
    if len(indices) == 0:
        return []

    for i in np.array(indices).flatten():
        detections.append(
            Detection(
                bbox=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                conf=float(scores[i]),
                class_id=int(cls_ids[i]),
            )
        )
    return detections


class YOLOXDetector:
    """Wraps a YOLOX ONNX model for person and/or ball detection.

    Args:
        model_path: Path to the ONNX export of a YOLOX checkpoint.
        input_size: Model input resolution (H, W). Default 640×640.
        conf_thr: Minimum objectness × class confidence to keep.
        detect_classes: Set of COCO class IDs to return (default: person only).
        device: 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (640, 640),
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

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single BGR frame and return detections."""
        blob, scale, pad = _preprocess(frame, self._input_size)
        outputs = self._session.run(None, {self._input_name: blob})
        return _postprocess(outputs[0], scale, pad, self._conf_thr, self._classes)

    @property
    def person_class(self) -> int:
        return _COCO_PERSON

    @property
    def ball_class(self) -> int:
        return _COCO_SPORTS_BALL
