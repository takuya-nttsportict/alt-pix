# ─────────────────────────────────────────────────────────────────────────────
# alt-pix: Volleyball tracking pipeline
#
# Base: NVIDIA PyTorch container (CUDA 12.4, Python 3.10, torch 2.3+)
# This image ships torch/torchvision compiled against NumPy 2.x, avoiding
# the NumPy 1.x / 2.x ABI conflict that arises with generic CUDA images.
#
# Build:
#   docker build -t alt-pix .
#
# Run (GPU):
#   docker run --gpus all --rm -it \
#     -v /path/to/videos:/data \
#     -v /path/to/models:/opt/alt-pix/models \
#     alt-pix \
#     python scripts/run_pipeline.py --source /data/game.mp4 ...
# ─────────────────────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Suppress interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# System libs needed by OpenCV headless and PyAV (FFmpeg) at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/alt-pix

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker layer cache is reused on code-only changes.
COPY requirements.txt .

# Upgrade pip, then install in a single layer to keep image size down.
# torch/torchvision from the base image satisfy the version constraints;
# pip will not reinstall them unless a newer version is required.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        # Explicit numpy pin: base image has numpy 1.x; upgrade to 2.x first
        # so all subsequent packages link against the correct ABI.
        "numpy>=2.0.0" && \
    pip install --no-cache-dir \
        av>=12.0.0 \
        "onnxruntime-gpu>=1.18.0" \
        "boxmot>=10.0.0" \
        "easyocr>=1.7.0" \
        "opencv-python-headless>=4.10.0" \
        "scipy>=1.13.0" \
        "tqdm>=4.66.0" \
        "pyyaml>=6.0.2" \
        "loguru>=0.7.0"

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Smoke-test: verify critical imports load without error
RUN python -c "\
import numpy; print('numpy', numpy.__version__); \
import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available()); \
import cv2; print('cv2', cv2.__version__); \
import av; print('av ok'); \
import easyocr; print('easyocr ok'); \
import onnxruntime; print('onnxruntime', onnxruntime.__version__); \
"

# Default: drop into bash so the user can run scripts interactively.
# Override with e.g.: docker run ... alt-pix python scripts/run_pipeline.py ...
CMD ["bash"]
