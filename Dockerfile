# ARG CUDA_VERSION=12.6
# ARG TORCH_BASE=full

# FROM xxxxrt666/torch-base:cu${CUDA_VERSION}-${TORCH_BASE}

# LABEL maintainer="XXXXRT"
# LABEL version="V4"
# LABEL description="Docker image for GPT-SoVITS"

# ARG CUDA_VERSION=12.6

# ENV CUDA_VERSION=${CUDA_VERSION}

# SHELL ["/bin/bash", "-c"]

# WORKDIR /workspace/GPT-SoVITS

# COPY Docker /workspace/GPT-SoVITS/Docker/

# ARG LITE=false
# ENV LITE=${LITE}

# ARG WORKFLOW=false
# ENV WORKFLOW=${WORKFLOW}

# ARG TARGETPLATFORM
# ENV TARGETPLATFORM=${TARGETPLATFORM}

# RUN bash Docker/miniconda_install.sh

# COPY extra-req.txt /workspace/GPT-SoVITS/

# COPY requirements.txt /workspace/GPT-SoVITS/

# COPY install.sh /workspace/GPT-SoVITS/

# RUN bash Docker/install_wrapper.sh

# EXPOSE 9871 9872 9873 9874 9880

# ENV PYTHONPATH="/workspace/GPT-SoVITS"

# RUN conda init bash && echo "conda activate base" >> ~/.bashrc

# WORKDIR /workspace

# RUN rm -rf /workspace/GPT-SoVITS

# WORKDIR /workspace/GPT-SoVITS

# COPY . /workspace/GPT-SoVITS

# CMD ["/bin/bash", "-c", "\
#   rm -rf /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models && \
#   rm -rf /workspace/GPT-SoVITS/GPT_SoVITS/text/G2PWModel && \
#   rm -rf /workspace/GPT-SoVITS/tools/asr/models && \
#   rm -rf /workspace/GPT-SoVITS/tools/uvr5/uvr5_weights && \
#   ln -s /workspace/models/pretrained_models /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models && \
#   ln -s /workspace/models/G2PWModel /workspace/GPT-SoVITS/GPT_SoVITS/text/G2PWModel && \
#   ln -s /workspace/models/asr_models /workspace/GPT-SoVITS/tools/asr/models && \
#   ln -s /workspace/models/uvr5_weights /workspace/GPT-SoVITS/tools/uvr5/uvr5_weights && \
#   exec bash"]

# Use a base image with Conda and CUDA support
FROM continuumio/anaconda3:2024.10-1

# Set working directory to GPT-SoVITS
WORKDIR /app

# Copy the entire GPT-SoVITS repository (already cloned) into the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create Conda environment named GPTSoVITS with Python 3.10
RUN conda create -n GPTSoVITS python=3.10 -y

# Activate Conda environment and install dependencies
SHELL ["/bin/bash", "-c"]
RUN conda run -n GPTSoVITS pip install ipykernel uvicorn fastapi && \
    conda run -n GPTSoVITS bash install.sh --device CU126 --source HF --download-uvr5

# Expose the FastAPI port
EXPOSE 9880

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI application
CMD ["conda", "run", "-n", "GPTSoVITS", "python", "api_v2.py", "-a", "0.0.0.0", "-p", "9880"]