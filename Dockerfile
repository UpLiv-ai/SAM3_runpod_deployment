# 1. Base Image
# We use the exact image you requested which matches your testing pod
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Use non-interactive mode to prevent apt-get from hanging on prompts
ENV DEBIAN_FRONTEND=noninteractive

# 2. Set the working directory
# We will mimic the path structure you used: /root/SAM3_runpod_deployment
WORKDIR /root/SAM3_runpod_deployment

# 3. System Dependencies
# Installing the specific libraries you identified as necessary
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your code into the container
# This takes all files in your current repo folder and puts them in /root/SAM3_runpod_deployment
COPY . .

# 5. Python Dependencies
# Install the external libraries first
RUN pip install --no-cache-dir \
    opencv-python \
    pycocotools \
    matplotlib \
    runpod \
    scipy \
    einops \
    decord

# 6. Install the local SAM 3 library
# We change directory to 'sam3' (inside the repo) and run the install
WORKDIR /root/SAM3_runpod_deployment/sam3
RUN pip install -e .

# 7. Return to the root of the deployment folder for execution
WORKDIR /root/SAM3_runpod_deployment

# 8. Start the Handler
# "-u" forces unbuffered stdout, so your print statements show up immediately in RunPod logs
CMD [ "python", "-u", "handler.py" ]