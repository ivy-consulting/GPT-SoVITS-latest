ARG CUDA_VERSION=12.6
ARG TORCH_BASE=full

FROM xxxxrt666/torch-base:cu${CUDA_VERSION}-${TORCH_BASE}

LABEL maintainer="XXXXRT"
LABEL version="V4"
LABEL description="Docker image for GPT-SoVITS with FastAPI"

ARG CUDA_VERSION=12.6
ENV CUDA_VERSION=${CUDA_VERSION}

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

COPY extra-req.txt /workspace/GPT-SoVITS/
COPY requirements.txt /workspace/GPT-SoVITS/
COPY install.sh /workspace/GPT-SoVITS/

RUN bash Docker/install_wrapper.sh

# Install Uvicorn (still needed for api_v2.py's internal uvicorn.run())
RUN pip install uvicorn

EXPOSE 9871 9872 9873 9874 9880

ENV PYTHONPATH="/workspace/GPT-SoVITS:/workspace/GPT-SoVITS/GPT_SoVITS:/workspace/GPT-SoVITS/GPT_SoVITS/eres2net"

RUN conda init bash && echo "conda activate base" >> ~/.bashrc

WORKDIR /workspace

RUN rm -rf /workspace/GPT-SoVITS

WORKDIR /workspace/GPT-SoVITS

COPY . /workspace/GPT-SoVITS

# Set up model directories with symbolic links
RUN rm -rf /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models && \
    rm -rf /workspace/GPT-SoVITS/GPT_SoVITS/text/G2PWModel && \
    rm -rf /workspace/GPT-SoVITS/tools/asr/models && \
    rm -rf /workspace/GPT-SoVITS/tools/uvr5/uvr5_weights && \
    ln -s /workspace/models/pretrained_models /workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models && \
    ln -s /workspace/models/G2PWModel /workspace/GPT-SoVITS/GPT_SoVITS/text/G2PWModel && \
    ln -s /workspace/models/asr_models /workspace/GPT-SoVITS/tools/asr/models && \
    ln -s /workspace/models/uvr5_weights /workspace/GPT-SoVITS/tools/uvr5/uvr5_weights

CMD ["/bin/bash", "-c", "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && export PYTHONPATH=/workspace/GPT-SoVITS:/workspace/GPT-SoVITS/GPT_SoVITS:/workspace/GPT-SoVITS/GPT_SoVITS/eres2net:$PYTHONPATH && python api_v2.py"]