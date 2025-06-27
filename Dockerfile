# Use a base image with CUDA 12.6
FROM nvidia/cuda:12.6.0-runtime-ubuntu20.04

LABEL maintainer="XXXXRT"
LABEL version="V4"
LABEL description="Docker image for GPT-SoVITS with FastAPI"

ARG CUDA_VERSION=12.6
ENV CUDA_VERSION=${CUDA_VERSION}
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH="/workspace/GPT-SoVITS-latest"

# Install system dependencies and Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python3-dev \
    wget \
    unzip \
    ffmpeg \
    libsox-dev \
    libsndfile1 \
    build-essential \
    cmake \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/* && \
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /home/ec2-user/miniconda3 && \
    rm /tmp/miniconda.sh && \
    /home/ec2-user/miniconda3/bin/conda init bash && \
    echo "conda activate base" >> /root/.bashrc

# Set working directory
WORKDIR /workspace/GPT-SoVITS-latest

# Copy requirements and install scripts
COPY requirements.txt extra-req.txt install.sh install_wrapper.sh /workspace/GPT-SoVITS-latest/

# Create Conda environment and run install_wrapper.sh
RUN . /home/ec2-user/miniconda3/etc/profile.d/conda.sh && \
    conda create -n GPTSoVITS python=3.9 -y && \
    conda activate GPTSoVITS && \
    pip install --upgrade pip && \
    pip install ipykernel uvicorn fastapi && \
    bash install_wrapper.sh && \
    pip cache purge && \
    rm -rf /tmp/* /var/tmp/* /home/ec2-user/miniconda3/pkgs /root/.conda /root/.cache

# Copy the GPT-SoVITS application
COPY . /workspace/GPT-SoVITS-latest

# Create model directories (in case /workspace/models is not mounted)
RUN mkdir -p /workspace/models/pretrained_models /workspace/models/G2PWModel \
    /workspace/models/asr_models /workspace/models/uvr5_weights

# Expose the FastAPI port
EXPOSE 9880

# Command to run the FastAPI application
CMD ["/home/ec2-user/miniconda3/envs/GPTSoVITS/bin/python3", "api_v2.py", "-a", "0.0.0.0", "-p", "9880"]