ARG CUDA_VERSION=12.6
ARG TORCH_BASE=full

FROM xxxxrt666/torch-base:cu${CUDA_VERSION}-${TORCH_BASE}

LABEL maintainer="XXXXRT"
LABEL version="V4"
LABEL description="Docker image for GPT-SoVITS with FastAPI"

ARG CUDA_VERSION=12.6
ENV CUDA_VERSION=${CUDA_VERSION}
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/workspace/GPT-SoVITS-latest"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace/GPT-SoVITS-latest

COPY Docker /workspace/GPT-SoVITS-latest/Docker/

ARG LITE=false
ENV LITE=${LITE}

ARG WORKFLOW=false
ENV WORKFLOW=${WORKFLOW}

ARG TARGETPLATFORM
ENV TARGETPLATFORM=${TARGETPLATFORM}

RUN bash Docker/miniconda_install.sh

COPY extra-req.txt requirements.txt install.sh install_wrapper.sh /workspace/GPT-SoVITS-latest/

RUN . /home/ec2-user/miniconda3/etc/profile.d/conda.sh && \
    conda create -n GPTSoVITS python=3.9 -y && \
    conda activate GPTSoVITS && \
    pip install --upgrade pip && \
    pip install ipykernel uvicorn fastapi && \
    bash install_wrapper.sh

COPY . /workspace/GPT-SoVITS-latest

# Create model directories in case /workspace/models is not mounted
RUN mkdir -p /workspace/models/pretrained_models /workspace/models/G2PWModel \
    /workspace/models/asr_models /workspace/models/uvr5_weights

# Expose the FastAPI port
EXPOSE 9880

# Command to run the FastAPI application
CMD ["/home/ec2-user/miniconda3/envs/GPTSoVITS/bin/python3", "api_v2.py", "-a", "0.0.0.0", "-p", "9880"]