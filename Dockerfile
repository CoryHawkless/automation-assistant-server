FROM nvcr.io/nvidia/pytorch:25.02-py3

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git



# Python deps
RUN pip3 install --no-cache-dir \
    websockets \
    numpy \
    openai-whisper \
    openwakeword \
    onnxruntime

RUN apt-get update
RUN apt-get install -y portaudio19-dev python3-pyaudio
RUN pip3 install pyaudio

# Download wake word models
RUN python3 -c "import openwakeword; openwakeword.Model()"

WORKDIR /code