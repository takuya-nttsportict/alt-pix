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

# Remove ALL cv2 artifacts from the base image (pip uninstall leaves .so files behind)
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true
RUN find /usr/local/lib/python3.10/dist-packages -maxdepth 1 \
        -name "cv2*" -o -name "opencv*" | xargs rm -rf || true
RUN find /usr/lib/python3 -maxdepth 3 \
        -name "cv2*" -o -name "opencv*" | xargs rm -rf || true

# Upgrade numpy to 2.x first so all subsequent packages use the correct ABI
RUN pip install --no-cache-dir --upgrade pip "numpy>=2.0.0"

WORKDIR /opt/alt-pix
COPY requirements.txt .

RUN pip install --no-cache-dir \
        "av>=12.0.0" \
        "onnxruntime-gpu>=1.18.0" \
        "boxmot>=10.0.0" \
        "easyocr>=1.7.0" \
        "opencv-python-headless>=4.10.0" \
        "scipy>=1.13.0" \
        "tqdm>=4.66.0" \
        "pyyaml>=6.0.2" \
        "loguru>=0.7.0"

COPY . .

RUN python -c "import numpy; print('numpy', numpy.__version__)"
RUN python -c "import torch; print('torch', torch.__version__)"
RUN python -c "import cv2; print('cv2', cv2.__version__)"
RUN python -c "import av; print('av ok')"
RUN python -c "import easyocr; print('easyocr ok')"
RUN python -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"

CMD ["bash"]
