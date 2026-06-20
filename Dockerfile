# ─────────────────────────────────────────────────────────────────────────────
# alt-pix: Volleyball tracking pipeline
#
# Base: NVIDIA PyTorch container (CUDA 12.4, Python 3.10, torch 2.4+)
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

ENV DEBIAN_FRONTEND=noninteractive

# System libs for FFmpeg (PyAV) and OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Remove the NumPy 1.x-linked cv2 that ships in the base image ─────────────
# The base image installs opencv-python (linked against numpy 1.x) at the
# system level. We must purge it before upgrading numpy to 2.x, otherwise
# the old .so is found first and raises "_ARRAY_API not found".
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python \
    || true

# ── Upgrade numpy to 2.x BEFORE everything else ──────────────────────────────
# All subsequent packages must resolve against the new ABI.
RUN pip install --no-cache-dir --upgrade pip "numpy>=2.0.0"

WORKDIR /opt/alt-pix
COPY requirements.txt .

# ── Install remaining dependencies ───────────────────────────────────────────
RUN pip install --no-cache-dir \
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

# Smoke-test: CUDA availability is False at build time (no GPU during build);
# runtime GPU access is confirmed separately via --gpus flag.
RUN python -c "import numpy; print('numpy', numpy.__version__)" && \
    python -c "import torch; print('torch', torch.__version__)" && \
    python -c "import cv2; print('cv2', cv2.__version__)" && \
    python -c "import av; print('av ok')" && \
    python -c "import easyocr; print('easyocr ok')" && \
    python -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)" && \
    echo "All imports OK"

CMD ["bash"]
