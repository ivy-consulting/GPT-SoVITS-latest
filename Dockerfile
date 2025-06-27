FROM nvidia/cuda:12.6.0-runtime-ubuntu20.04

LABEL maintainer="XXXXRT"
LABEL version="V4"
LABEL description="Docker image for GPT-SoVITS"

ARG CUDA_VERSION=12.6
ENV CUDA_VERSION=${CUDA_VERSION}
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/workspace/GPT-SoVITS"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-c"]

# Install system dependencies and Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3-pip python3-dev wget unzip ffmpeg libsox-dev libsndfile1 build-essential cmake git git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/* && \
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /root/miniconda3 && \
    rm /tmp/miniconda.sh && \
    /root/miniconda3/bin/conda init bash

WORKDIR /workspace/GPT-SoVITS

COPY extra-req.txt requirements.txt install.sh /workspace/GPT-SoVITS/

# Create install_wrapper.sh
RUN echo '#!/bin/bash' > install_wrapper.sh && \
    echo 'set -e' >> install_wrapper.sh && \
    echo 'source "/root/miniconda3/etc/profile.d/conda.sh"' >> install_wrapper.sh && \
    echo 'conda create -n GPTSoVITS python=3.9 -y' >> install_wrapper.sh && \
    echo 'conda activate GPTSoVITS' >> install_wrapper.sh && \
    echo 'pip install --upgrade pip' >> install_wrapper.sh && \
    echo 'pip install ipykernel uvicorn fastapi' >> install_wrapper.sh && \
    echo 'mkdir -p GPT_SoVITS/text tools/asr tools/uvr5' >> install_wrapper.sh && \
    echo 'ln -sf /workspace/models/pretrained_models /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models' >> install_wrapper.sh && \
    echo 'ln -sf /workspace/models/G2PWModel /workspace/GPT-SoVITS/GPT_SoVITS/text/G2PWModel' >> install_wrapper.sh && \
    echo 'ln -sf /workspace/models/asr_models /workspace/GPT-SoVITS/tools/asr/models' >> install_wrapper.sh && \
    echo 'ln -sf /workspace/models/uvr5_weights /workspace/GPT-SoVITS/tools/uvr5/uvr5_weights' >> install_wrapper.sh && \
    echo 'bash install.sh --device "CU${CUDA_VERSION//./}" --source HF' >> install_wrapper.sh && \
    echo 'pip cache purge' >> install_wrapper.sh && \
    echo 'pip show torch' >> install_wrapper.sh && \
    echo 'rm -rf /tmp/* /var/tmp/*' >> install_wrapper.sh && \
    echo 'rm -rf "/root/miniconda3/pkgs"' >> install_wrapper.sh && \
    echo 'mkdir -p "/root/miniconda3/pkgs"' >> install_wrapper.sh && \
    echo 'rm -rf /root/.conda /root/.cache' >> install_wrapper.sh && \
    chmod +x install_wrapper.sh && \
    bash install_wrapper.sh

EXPOSE 9871 9872 9873 9874 9880

ENV PYTHONPATH="/workspace/GPT-SoVITS"

RUN conda init bash && echo "conda activate base" >> ~/.bashrc

WORKDIR /workspace

RUN rm -rf /workspace/GPT-SoVITS

WORKDIR /workspace/GPT-SoVITS

COPY . /workspace/GPT-SoVITS

# Command to run the FastAPI application
CMD ["/root/miniconda3/envs/GPTSoVITS/bin/python3", "api_v2.py", "-a", "0.0.0.0", "-p", "9880"]