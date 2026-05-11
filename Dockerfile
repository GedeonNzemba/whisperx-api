# Dockerfile for RunPod GPU deployment.
# Mounts /models as a Network Volume (50 GB) for persistent model caching.

############################
# Stage 1 — builder
############################
FROM --platform=linux/amd64 nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        build-essential git ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3

RUN python -m pip install --upgrade pip wheel setuptools

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Install torch 2.6.0 with CUDA 12.4 wheels first (matches base image + chatterbox-tts requirement).
RUN pip install torch==2.6.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

# Install the rest of the Python stack. torch/torchaudio are already present so pip
# will not re-download them. chatterbox-tts is installed here with full deps resolved.
RUN pip install -r /app/requirements.txt && \
    pip install chatterbox-tts

# We need TWO cuDNN versions simultaneously:
#   cuDNN 8 (libcudnn_ops_infer.so.8) — required by CTranslate2/WhisperX
#   cuDNN 9 (libcudnn.so.9)           — required by PyTorch 2.6
#
# Strategy: install cuDNN 8 first, copy its .so files to /usr/local/lib/cudnn8/,
# then reinstall cuDNN 9 (which torch already declared as a dep). Register both
# directories with ldconfig so the system linker finds both at runtime.
RUN pip install nvidia-cudnn-cu12==8.9.7.29 && \
    mkdir -p /usr/local/lib/cudnn8 && \
    cp /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn*.so* \
       /usr/local/lib/cudnn8/ && \
    pip install nvidia-cudnn-cu12==9.1.0.70 && \
    printf '/usr/local/lib/cudnn8\n/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib\n' \
        > /etc/ld.so.conf.d/zzz-nvidia-cudnn.conf && \
    ldconfig

############################
# Stage 2 — runtime
############################
FROM --platform=linux/amd64 nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    COMPUTE_TYPE=float16 \
    # /models is the RunPod Network Volume mount point.
    MODEL_DIR=/models \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        ffmpeg libsndfile1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3

# Copy installed Python packages from builder.
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
# cuDNN 8 .so files saved separately so both cuDNN 8 and 9 are available.
COPY --from=builder /usr/local/lib/cudnn8 /usr/local/lib/cudnn8
# Register both cuDNN versions with the runtime linker.
COPY --from=builder /etc/ld.so.conf.d/zzz-nvidia-cudnn.conf /etc/ld.so.conf.d/zzz-nvidia-cudnn.conf
RUN ldconfig

WORKDIR /app
COPY server.py /app/server.py
COPY static /app/static
COPY vbx_diarize.py /app/vbx_diarize.py
COPY vibevoice_client.py /app/vibevoice_client.py
COPY vibevoice/ /app/vibevoice/
COPY s2s/ /app/s2s/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# /models is expected to be the mounted Network Volume; create a fallback
# in case the container runs without it (e.g. local testing).
RUN mkdir -p /models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://127.0.0.1:${PORT}/health || exit 1

CMD ["/app/start.sh"]
