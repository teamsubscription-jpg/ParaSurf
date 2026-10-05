# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    nano \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Set Conda environment path (the ParaSurf env comes first so `python` and `obabel` resolve to it)
ENV PATH="/opt/conda/envs/ParaSurf/bin:/opt/conda/bin:$PATH"
SHELL ["/bin/bash", "-c"]

# Accept Anaconda channel Terms of Service (required by recent Miniconda for non-interactive builds)
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Set workspace directory
WORKDIR /workspace/ParaSurf

# Create Conda environment and install dependencies from requirements.txt
COPY requirements.txt .
RUN conda create -n ParaSurf python=3.10 -y && \
    conda run -n ParaSurf pip install --no-cache-dir -r requirements.txt && \
    conda run -n ParaSurf conda install -c conda-forge openbabel -y && \
    conda run -n ParaSurf pip install --no-cache-dir gdown runpod && \
    conda clean --all -y

# Download model weights (see the weights table in README.md)
RUN mkdir -p weights && \
    gdown 1nd3npYK303e8owDBvW8Ygd5m9SD1puhR -O weights/Paragraph_expanded_best.pth && \
    gdown 1vZGH-T6K5_ShVma3dwLkLdkoivs09rSP -O weights/Pecan_best.pth && \
    gdown 16LA99tPYP7vkKpc-ycn98esEUhXUDc-n -O weights/Paragraph_expanded_heavy_best.pth && \
    gdown 1mEBLPKi1sny-inr1ogdYWoo44XgbH8db -O weights/Paragraph_expanded_light_best.pth && \
    ls -la weights

# Copy this repository (the fork) into the image
COPY . .

# Install DMS (which is inside the ParaSurf directory)
RUN cd dms && make install && \
    chmod +x /workspace/ParaSurf/pdb2pqr-linux-bin64-2.1.1/pdb2pqr

# Set PYTHONPATH for ParaSurf
ENV PYTHONPATH=/workspace/ParaSurf

# Start the RunPod serverless worker
CMD ["python", "-u", "handler.py"]
