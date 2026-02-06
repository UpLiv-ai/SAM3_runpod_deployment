# Use RunPod's bleeding-edge PyTorch 2.8 image (matches your screenshot requirements)
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. System Dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
# RunPod image includes torch, so we just need the extras
RUN pip install --no-cache-dir \
    opencv-python \
    pycocotools \
    matplotlib \
    runpod \
    scipy

# 3. Install SAM 3 from Source
WORKDIR /app
RUN git clone https://github.com/facebookresearch/sam3.git
WORKDIR /app/sam3
# Install in editable mode to ensure all deps are resolved
RUN pip install -e.

# 4. Setup Handler
COPY handler.py /app/handler.py

# 5. Runtime Configuration
# We assume weights are in the Network Volume mounted at /runpod-volume
ENV SAM3_CHECKPOINT_DIR="/runpod-volume/sam3/checkpoints"

WORKDIR /app
CMD [ "python", "-u", "handler.py" ]