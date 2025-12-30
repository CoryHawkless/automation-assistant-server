FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip3 install --no-cache-dir \
    websockets \
    numpy \
    openai-whisper \
    openwakeword \
    onnxruntime

# Download wake word models
RUN python3 -c "from openwakeword import utils; utils.download_models()"

COPY jeeves_server.py .

EXPOSE 8765

CMD ["python3", "jeeves_server.py"]