# Jeeves - Wake Word Speech-to-Text Server

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![GPU](https://img.shields.io/badge/GPU-NVIDIA-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Jeeves** is an open-source, real-time speech-to-text server with integrated wake word detection. Say a wake word like "Alexa" and start speaking—Jeeves will transcribe your speech using OpenAI's Whisper model.

[Features](#features) • [Quick Start](#quick-start) • [Architecture](#architecture) • [Contributing](#contributing)

</div>

---

## What is Jeeves?

Jeeves is a lightweight, self-hosted speech recognition server designed for:

- **Smart home integrations** - Add voice commands to your automation
- **Assistants** - Build your own voice assistant with wake word activation
- **Accessibility** - Voice-to-text for applications
- **Prototyping** - Quick speech recognition for your projects

No cloud services required. Everything runs locally on your machine with GPU acceleration.

## Features

- 🎤 **Wake Word Detection** - Listens for wake words (Alexa, Hey Siri, etc.) using openWakeWord
- 📝 **Real-time Transcription** - Converts speech to text using OpenAI Whisper
- ⚡ **GPU Accelerated** - Runs on NVIDIA GPUs via CUDA for fast inference
- 🔌 **WebSocket API** - Simple JSON-based protocol for easy integration
- 🐳 **Docker Ready** - Deploy anywhere with Docker/docker-compose
- 🎛️ **Configurable** - Adjust thresholds, silence detection, and more

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support
- Docker & Docker Compose
- NVIDIA Container Toolkit

### Running with Docker

```bash
# Clone the repository
git clone https://github.com/yourusername/jeeves.git
cd jeeves

# Start the server
docker compose up --build
```

The server will be available at `ws://localhost:8765`

### Running Locally

```bash
# Install dependencies
pip install websockets numpy openai-whisper openwakeword onnxruntime

# Download wake word models
python -c "from openwakeword import utils; utils.download_models()"

# Start the server
python jeeves_server.py
```

## WebSocket API

Connect to `ws://localhost:8765` and send audio data (16-bit PCM, 16kHz mono).

### Client → Server

| Message | Description |
|---------|-------------|
| Raw audio bytes | Audio data for processing |
| `TRIGGER:` | Manually start recording |
| `CONFIG:<json>` | Optional configuration |

### Server → Client

| Message | Description |
|---------|-------------|
| `WAKE:<word>` | Wake word detected |
| `TRANSCRIPT:<text>` | Transcription result |
| `STATUS:<message>` | Current status |

### Example Session

```
→ (audio stream)
← WAKE:alexa
← STATUS:Recording...
→ (more audio)
← STATUS:Transcribing...
← TRANSCRIPT:turn on the living room lights
← STATUS:Listening...
```

## Architecture

```
┌─────────────┐     WebSocket      ┌──────────────┐
│   Client    │ ◄────────────────► │   Jeeves     │
│ ( microphone)│                   │   Server     │
└─────────────┘                    └──────┬───────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     │                    │                    │
              ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐
              │ openWakeWord│      │   VAD/SIL   │      │   Whisper   │
              │ Wake Detector│      │  Detection  │      │Transcriber  │
              └─────────────┘      └─────────────┘      └─────────────┘
```

1. **Wake Word Detection** - Continuously monitors audio for wake words
2. **Voice Activity Detection** - Detects speech vs silence
3. **Transcription** - Converts speech segments to text

## Configuration

Edit [`jeeves_server.py`](jeeves_server.py) to customize:

```python
# Wake word sensitivity
WAKE_THRESHOLD = 0.5

# Silence detection (seconds)
INITIAL_SILENCE_DURATION = 1.5  # Patience at start
SHORT_SILENCE_DURATION = 0.7    # After speech detected

# Recording limits
MAX_RECORDING = 15.0  # Max seconds per recording
```

## Performance Tips

- **GPU Memory**: Whisper small model uses ~1GB VRAM
- **CPU Mode**: Set `device="cpu"` in code for CPU-only inference
- **Model Size**: Change `"small"` to `"tiny"` or `"base"` for faster inference

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Ways to Contribute

- 🐛 **Bug Reports** - Found an issue? Let us know
- 💡 **Feature Requests** - Have an idea? Share it
- 📖 **Documentation** - Improve the docs
- 🔧 **Code** - Fix bugs, add features, optimize performance
- 🌍 **Translation** - Help with internationalization

### Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b my-feature`
3. Commit your changes: `git commit -am 'Add awesome feature'`
4. Push to the branch: `git push origin my-feature`
5. Submit a Pull Request

## Built With

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [openWakeWord](https://github.com/dscripka/openWakeWord) - Wake word detection
- [WebSockets](https://websockets.readthedocs.io/) - Real-time communication
- [PyTorch](https://pytorch.org/) - ML framework

## Acknowledgments

- OpenAI for releasing Whisper as open source
- The openWakeWord team for the wake word models
- The KiloCode team for AI-assisted development

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Jeeves** - Your local speech recognition assistant

[GitHub](https://github.com/yourusername/jeeves) • [Issues](https://github.com/yourusername/jeeves/issues)

</div>
