# 1. Base Image
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Use non-interactive mode to prevent apt-get from hanging on prompts
ENV DEBIAN_FRONTEND=noninteractive

# 2. Set the working directory
WORKDIR /root/SAM3_runpod_deployment

# 3. System Dependencies
# FIX: 'libgl1-mesa-glx' is removed in Ubuntu 24.04. Replaced with 'libgl1'.
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your code into the container
COPY . .

# 5. Python Dependencies
RUN pip install --no-cache-dir \
    opencv-python \
    pycocotools \
    matplotlib \
    runpod \
    scipy \
    einops \
    decord

# 6. Install the local SAM 3 library
WORKDIR /root/SAM3_runpod_deployment/sam3
RUN pip install -e .

# 7. Return to the root of the deployment folder for execution
WORKDIR /root/SAM3_runpod_deployment

# 8. Start the Handler
CMD [ "python", "-u", "handler.py" ]