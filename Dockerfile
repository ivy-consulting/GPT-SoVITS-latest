ARG CUDA_VERSION=12.6
ARG TORCH_BASE=full

FROM xxxxrt666/torch-base:cu${CUDA_VERSION}-${TORCH_BASE}

LABEL maintainer="XXXXRT"
LABEL version="V4"
LABEL description="Docker image for GPT-SoVITS with FastAPI"

ARG CUDA_VERSION=12.6
ENV CUDA_VERSION=${CUDA_VERSION}
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/workspace/GPT-SoVITS"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace/GPT-SoVITS

COPY Docker /workspace/GPT-SoVITS/Docker/

ARG LITE=false
ENV LITE=${LITE}

ARG WORKFLOW=false
ENV WORKFLOW=${WORKFLOW}

ARG TARGETPLATFORM
ENV TARGETPLATFORM=${TARGETPLATFORM}

# Run miniconda_install.sh and ensure Conda is initialized
RUN bash Docker/miniconda_install.sh && \
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then \
        echo "source /root/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc; \
    elif [ -f "/home/ec2-user/miniconda3/etc/profile.d/conda.sh" ]; then \
        echo "source /home/ec2-user/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc; \
    else \
        echo "Error: conda.sh not found" && exit 1; \
    fi && \
    conda init bash

COPY extra-req.txt requirements.txt install.sh /workspace/GPT-SoVITS/

# Create install_wrapper.sh
RUN echo '#!/bin/bash' > install_wrapper.sh && \
    echo 'set -e' >> install_wrapper.sh && \
    echo 'if [ -f "/home/ec2-user/miniconda3/etc/profile.d/conda.sh" ]; then' >> install_wrapper.sh && \
    echo '    source "/home/ec2-user/miniconda3/etc/profile.d/conda.sh"' >> install_wrapper.sh && \
    echo 'elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then' >> install_wrapper.sh && \
    echo '    source "/root/miniconda3/etc/profile.d/conda.sh"' >> install_wrapper.sh && \
    echo 'else' >> install_wrapper.sh && \
    echo '    echo "Error: conda.sh not found" && exit 1' >> install_wrapper.sh && \
    echo 'fi' >> install_wrapper.sh && \
    echo 'conda create -n GPTSoVITS python=3.9 -y' >> install_wrapper.sh && \
    echo 'conda activate GPTSoVITS' >> install_wrapper.sh && \
    echo 'pip install --upgrade pip' >> install_wrapper.sh && \
    echo 'pip install ipykernel uvicorn fastapi' >> install_wrapper.sh && \
    echo 'mkdir -p GPT_SoVITS/text' >> install_wrapper.sh && \
    echo 'mkdir -p tools/asr' >> install_wrapper.sh && \
    echo 'mkdir -p tools/uvr5' >> install_wrapper.sh && \
    echo 'ln -sf /workspace/models/pretrained_models /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models' >> install_wrapper.sh && \
    echo 'ln -sf /workspace/models/G2PWModel /workspace/GPT-SoVITS/GPT_SoVITS/text/G2PWModel' >> install_wrapper.sh && \
    echo 'ln -sf /workspace/models/asr_models /workspace/GPT-SoVITS/tools/asr/models' >> install_wrapper.sh && \
    echo 'ln -sf /workspace/models/uvr5_weights /workspace/GPT-SoVITS/tools/uvr5/uvr5_weights' >> install_wrapper.sh && \
    echo 'bash install.sh --device "CU${CUDA_VERSION//./}" --source HF --skip-download' >> install_wrapper.sh && \
    echo 'pip cache purge' >> install_wrapper.sh && \
    echo 'pip show torch' >> install_wrapper.sh && \
    echo 'rm -rf /tmp/* /var/tmp/*' >> install_wrapper.sh && \
    echo 'rm -rf "/home/ec2-user/miniconda3/pkgs" || rm -rf "/root/miniconda3/pkgs"' >> install_wrapper.sh && \
    echo 'mkdir -p "/home/ec2-user/miniconda3/pkgs" || mkdir -p "/root/miniconda3/pkgs"' >> install_wrapper.sh && \
    echo 'rm -rf /root/.conda /root/.cache' >> install_wrapper.sh && \
    chmod +x install_wrapper.sh

RUN bash install_wrapper.sh

COPY . /workspace/GPT-SoVITS

EXPOSE 9871 9872 9873 9874 9880

ENV PYTHONPATH="/workspace/GPT-SoVITS"

WORKDIR /workspace

RUN rm -rf /workspace/GPT-SoVITS

WORKDIR /workspace/GPT-SoVITS

COPY . /workspace/GPT-SoVITS

# Create model directories in case /workspace/models is not mounted
RUN mkdir -p /workspace/models/pretrained_models /workspace/models/G2PWModel \
    /workspace/models/asr_models /workspace/models/uvr5_weights

# Command to run the FastAPI application
CMD ["/home/ec2-user/miniconda3/envs/GPTSoVITS/bin/python3", "api_v2.py", "-a", "0.0.0.0", "-p", "9880"]