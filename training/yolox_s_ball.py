"""
YOLOX-s experiment config for volleyball ball detection.

Fine-tunes the COCO pretrained yolox_s checkpoint on a single-class
(volleyball) dataset.

Usage:
  python -m yolox.tools.train \\
    -f training/yolox_s_ball.py \\
    -c yolox_s.pth \\
    --num_machines 1 \\
    --num_gpus 1 \\
    -b 16 \\
    -o           # occupy GPU memory upfront

After training, export to ONNX:
  python scripts/export_ball_onnx.py \\
    --exp training/yolox_s_ball.py \\
    --ckpt training/runs/yolox_s_ball/best_ckpt.pth \\
    --out models/yolox_s_ball.onnx
"""

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolox_s_ball"

        # ── Model ──────────────────────────────────────────────────────────
        self.depth = 0.33   # YOLOX-s depth multiplier
        self.width = 0.50   # YOLOX-s width multiplier
        self.num_classes = 1  # single class: volleyball

        # ── Dataset ────────────────────────────────────────────────────────
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "volleyball_ball"
        )
        self.train_ann = "train/labels"
        self.val_ann = "val/labels"

        # Input resolution: 640×640 balances speed and small-object accuracy
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = (14, 26)  # multi-scale training (×32)

        # ── Training ───────────────────────────────────────────────────────
        self.max_epoch = 100
        self.warmup_epochs = 5
        self.no_aug_epochs = 15   # disable mosaic/mixup for last N epochs
        self.basic_lr_per_img = 0.01 / 64  # scales with batch size
        self.weight_decay = 5e-4

        # ── Augmentation ───────────────────────────────────────────────────
        # Keep mosaic; it helps with small-object detection (the ball).
        self.mosaic_prob = 1.0
        self.mixup_prob = 0.5
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        # ── Evaluation ─────────────────────────────────────────────────────
        self.eval_interval = 5  # evaluate every N epochs
        self.test_conf = 0.35
        self.nmsthre = 0.45

        # ── Output ─────────────────────────────────────────────────────────
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "training", "runs"
        )

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """Override to use YOLO-format label loader."""
        from yolox.data import YOLOXDataset, TrainTransform

        return YOLOXDataset(
            data_dir=self.data_dir,
            json_file=None,           # YOLO txt format, not COCO JSON
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )
