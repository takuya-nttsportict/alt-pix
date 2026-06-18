#!/usr/bin/env python3
"""
Export a trained YOLOX ball-detection checkpoint to ONNX.

Usage:
  python scripts/export_ball_onnx.py \\
    --exp training/yolox_s_ball.py \\
    --ckpt training/runs/yolox_s_ball/best_ckpt.pth \\
    --out models/yolox_s_ball.onnx \\
    [--input-size 640]    # must match training resolution
    [--opset 17]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, help="Experiment .py file")
    p.add_argument("--ckpt", required=True, help="Trained checkpoint .pth")
    p.add_argument("--out", default="models/yolox_s_ball.onnx")
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load experiment config
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp", args.exp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exp = mod.Exp()

    # Build model and load weights
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    if args.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    sz = args.input_size
    dummy = torch.zeros(1, 3, sz, sz).to(next(model.parameters()).device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["images"],
        output_names=["output"],
        opset_version=args.opset,
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"Exported: {args.out}  (opset={args.opset}, input={sz}×{sz})")


if __name__ == "__main__":
    main()
