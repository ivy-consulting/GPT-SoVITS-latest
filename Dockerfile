ARG CUDA_VERSION=12.6
ARG TORCH_BASE=full

FROM xxxxrt666/torch-base:cu${CUDA_VERSION}-${TORCH_BASE}

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

WORKDIR /workspace/GPT-SoVITS

COPY Docker /workspace/GPT-SoVITS/Docker/

ARG LITE=false
ENV LITE=${LITE}

ARG WORKFLOW=false
ENV WORKFLOW=${WORKFLOW}

ARG TARGETPLATFORM
ENV TARGETPLATFORM=${TARGETPLATFORM}

RUN bash Docker/miniconda_install.sh

COPY extra-req.txt requirements.txt install.sh /workspace/GPT-SoVITS/

# Create install_wrapper.sh
RUN echo '#!/bin/bash' > install_wrapper.sh && \
    echo 'set -e' >> install_wrapper.sh && \
    echo 'source "/home/ec2-user/miniconda3/etc/profile.d/conda.sh"' >> install_wrapper.sh && \
    echo 'mkdir -p GPT_SoVITS' >> install_wrapper.sh && \
    echo 'mkdir -p GPT_SoVITS/text' >> install_wrapper.sh && \
    echo 'ln -s /workspace/models/pretrained_models /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models' >> install_wrapper.sh && \
    echo 'ln -s /workspace/models/G2PWModel /workspace/GPT-SoVITS/GPT_SoVITS/text/G2PWModel' >> install_wrapper.sh && \
    echo 'bash install.sh --device "CU${CUDA_VERSION//./}" --source HF' >> install_wrapper.sh && \
    echo 'pip cache purge' >> install_wrapper.sh && \
    echo 'pip show torch' >> install_wrapper.sh && \
    echo 'rm -rf /tmp/* /var/tmp/*' >> install_wrapper.sh && \
    echo 'rm -rf "/home/ec2-user/miniconda3/pkgs"' >> install_wrapper.sh && \
    echo 'mkdir -p "/home/ec2-user/miniconda3/pkgs"' >> install_wrapper.sh && \
    echo 'rm -rf /root/.conda /root/.cache' >> install_wrapper.sh && \
    chmod +x install_wrapper.sh

RUN bash install_wrapper.sh

COPY . /workspace/GPT-SoVITS

EXPOSE 9871 9872 9873 9874 9880

ENV PYTHONPATH="/workspace/GPT-SoVITS"

RUN conda init bash && echo "conda activate base" >> ~/.bashrc

WORKDIR /workspace

RUN rm -rf /workspace/GPT-SoVITS

WORKDIR /workspace/GPT-SoVITS

COPY . /workspace/GPT-SoVITS

# Create model directories in case /workspace/models is not mounted
RUN mkdir -p /workspace/models/pretrained_models /workspace/models/G2PWModel \
    /workspace/models/asr_models /workspace/models/uvr5_weights

# Command to run the FastAPI application
CMD ["/home/ec2-user/miniconda3/envs/GPTSoVITS/bin/python3", "api_v2.py", "-a", "0.0.0.0", "-p", "9880"]
