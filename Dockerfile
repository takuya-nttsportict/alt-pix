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

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Remove NumPy 1.x-linked packages from base image ─────────────────────────
# The base image ships opencv, torchvision, pyarrow, pandas all compiled
# against numpy 1.x.  We uninstall them here and reinstall numpy-2.x-
# compatible wheels in subsequent steps.
RUN pip uninstall -y \
        opencv-python opencv-python-headless opencv-contrib-python \
        torchvision \
        pyarrow \
    2>/dev/null || true

# Physically remove cv2 .so that pip uninstall leaves behind
RUN find /usr/local/lib/python3.10/dist-packages -maxdepth 2 \
        \( -name "cv2*" -o -name "opencv*" \) -exec rm -rf {} + 2>/dev/null || true

# ── Step 1: upgrade numpy to 2.x ─────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip "numpy>=2.0.0"

# ── Step 2: reinstall torchvision from cu124 wheels (numpy 2.x compatible) ───
RUN pip install --no-cache-dir \
        torchvision \
        --index-url https://download.pytorch.org/whl/cu124

# ── Step 3: install application dependencies ──────────────────────────────────
WORKDIR /opt/alt-pix
COPY requirements.txt .

RUN pip install --no-cache-dir \
        "av>=12.0.0" \
        "onnxruntime-gpu>=1.18.0" \
        "supervision>=0.29.0" \
        "trackers>=2.4.0" \
        "easyocr>=1.7.0" \
        "opencv-python-headless>=4.10.0" \
        "scipy>=1.13.0" \
        "tqdm>=4.66.0" \
        "pyyaml>=6.0.2" \
        "loguru>=0.7.0"

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Smoke-test each import separately for clear failure diagnosis
RUN python -c "import numpy; print('numpy', numpy.__version__)"
RUN python -c "import torch; print('torch', torch.__version__)"
RUN python -c "import torchvision; print('torchvision', torchvision.__version__)"
RUN python -c "import cv2; print('cv2', cv2.__version__)"
RUN python -c "import av; print('av ok')"
RUN python -c "import supervision; print('supervision', supervision.__version__)"
RUN python -c "import trackers; print('trackers ok')"
RUN python -c "import easyocr; print('easyocr ok')"
RUN python -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"

CMD ["bash"]
