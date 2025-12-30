#!/usr/bin/env python3
"""
Jeeves STT Server
Wake word detection + real-time speech-to-text
"""

import asyncio
import websockets
import numpy as np
import collections
import time
from typing import Optional
import whisper
import torch
from openwakeword.model import Model as WakeWordModel

# Audio config (must match client)
SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # int16

# Wake word settings
WAKE_THRESHOLD = 0.5
POST_WAKE_DELAY = 0.3  # Seconds to discard after wake word (removes "alexa" from transcript)

# VAD / Recording settings
SILENCE_THRESHOLD = 0.01  # RMS threshold for silence
MIN_SPEECH_DURATION = 0.3  # Minimum speech before considering silence
INITIAL_SILENCE_DURATION = 1.5  # Longer patience at start
SHORT_SILENCE_DURATION = 0.7  # Shorter silence after speech detected
MAX_RECORDING = 15.0  # Max seconds to record

DEBUG = False


def log(msg: str, level: str = "INFO"):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)


def debug(msg: str):
    if DEBUG:
        log(msg, "DEBUG")


class JeevesServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 8765):
        self.host = host
        self.port = port
        
        log("Checking CUDA availability...")
        log(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        
        log("Loading models...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"  Loading Whisper model (small) on {device}...")
        load_start = time.time()
        self.whisper_model = whisper.load_model("small", device=device)
        log(f"  Whisper loaded in {time.time() - load_start:.1f}s")
        
        log("  Loading openWakeWord...")
        load_start = time.time()
        self.wake_model = WakeWordModel(inference_framework="onnx")
        self.wake_word_names = list(self.wake_model.models.keys())
        log(f"  Wake words available: {self.wake_word_names}")
        log(f"  openWakeWord loaded in {time.time() - load_start:.1f}s")
        
        log("All models loaded!")

    def get_audio_level(self, audio_bytes: bytes) -> float:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return np.sqrt(np.mean(audio ** 2))

    def is_silence(self, audio_bytes: bytes) -> bool:
        return self.get_audio_level(audio_bytes) < SILENCE_THRESHOLD

    def detect_wake_word(self, audio_bytes: bytes) -> tuple[bool, float, str]:
        """Returns (detected, score, wake_word_name)"""
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        prediction = self.wake_model.predict(audio)
        
        max_score = 0.0
        detected_word = None
        
        for name in self.wake_word_names:
            score = float(prediction.get(name, 0))
            if score > max_score:
                max_score = score
                if score > WAKE_THRESHOLD:
                    detected_word = name
        
        return detected_word is not None, max_score, detected_word

    def transcribe(self, audio_data: bytes) -> str:
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        duration = len(audio) / SAMPLE_RATE
        log(f"Transcribing {duration:.1f}s of audio...")
        start_time = time.time()
        
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
        
        options = whisper.DecodingOptions(language="en", fp16=torch.cuda.is_available())
        result = whisper.decode(self.whisper_model, mel, options)
        
        text = result.text.strip()
        elapsed = time.time() - start_time
        
        log(f"Transcription complete in {elapsed:.2f}s")
        return text

    async def handle_client(self, websocket, path=None):
        client_addr = websocket.remote_address
        log(f"Client connected: {client_addr}")
        
        self.wake_model.reset()
        
        state = "LISTENING"
        recording_buffer = bytearray()
        silence_start: Optional[float] = None
        recording_start: Optional[float] = None
        speech_detected = False
        post_wake_discard_until: Optional[float] = None
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    if message.startswith("CONFIG:"):
                        log(f"Client config: {message[7:]}")
                        await websocket.send("STATUS:Ready - say 'Alexa' to activate")
                        continue
                    elif message.startswith("TRIGGER:"):
                        log("Manual trigger received")
                        await websocket.send("STATUS:Recording...")
                        recording_buffer = bytearray()
                        recording_start = time.time()
                        silence_start = None
                        speech_detected = False
                        post_wake_discard_until = None
                        state = "RECORDING"
                        continue
                
                audio_chunk = message
                audio_level = self.get_audio_level(audio_chunk)
                
                if state == "LISTENING":
                    detected, score, wake_word = self.detect_wake_word(audio_chunk)
                    
                    if detected:
                        log(f"Wake word '{wake_word}' detected (score: {score:.2f})")
                        await websocket.send(f"WAKE:{wake_word}")
                        
                        # Start recording but discard audio for a short period
                        # to avoid capturing the wake word itself
                        recording_buffer = bytearray()
                        recording_start = time.time()
                        post_wake_discard_until = time.time() + POST_WAKE_DELAY
                        silence_start = None
                        speech_detected = False
                        state = "RECORDING"
                
                elif state == "RECORDING":
                    now = time.time()
                    elapsed = now - recording_start
                    
                    # Discard audio immediately after wake word
                    if post_wake_discard_until and now < post_wake_discard_until:
                        debug(f"Discarding post-wake audio ({post_wake_discard_until - now:.2f}s remaining)")
                        continue
                    
                    post_wake_discard_until = None
                    recording_buffer.extend(audio_chunk)
                    
                    # Track if we've heard any speech
                    if not self.is_silence(audio_chunk):
                        if not speech_detected:
                            debug("Speech detected")
                        speech_detected = True
                        silence_start = None
                    else:
                        # Silence detected
                        if silence_start is None:
                            silence_start = now
                        
                        # Variable silence duration based on whether we've heard speech
                        if speech_detected:
                            # Already heard speech - use shorter silence threshold
                            silence_duration = SHORT_SILENCE_DURATION
                        else:
                            # Haven't heard speech yet - be more patient
                            silence_duration = INITIAL_SILENCE_DURATION
                        
                        if now - silence_start >= silence_duration:
                            state = "TRANSCRIBING"
                    
                    # Max recording limit
                    if elapsed >= MAX_RECORDING:
                        log("Max recording duration reached")
                        state = "TRANSCRIBING"
                    
                    if state == "TRANSCRIBING":
                        # Check if we have enough audio
                        buffer_duration = len(recording_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                        
                        if buffer_duration < MIN_SPEECH_DURATION:
                            log(f"Recording too short ({buffer_duration:.2f}s), discarding")
                            await websocket.send("STATUS:No command detected")
                        else:
                            await websocket.send("STATUS:Transcribing...")
                            text = self.transcribe(bytes(recording_buffer))
                            
                            if text.strip():
                                log(f"Transcript: '{text}'")
                                await websocket.send(f"TRANSCRIPT:{text}")
                            else:
                                await websocket.send("STATUS:No speech detected")
                        
                        # Reset
                        recording_buffer.clear()
                        self.wake_model.reset()
                        state = "LISTENING"
                        await websocket.send("STATUS:Listening...")
                        
        except websockets.exceptions.ConnectionClosed:
            log(f"Client disconnected: {client_addr}")
        except Exception as e:
            log(f"Error: {e}", "ERROR")
            import traceback
            traceback.print_exc()

    async def run(self):
        log(f"Jeeves server starting on ws://{self.host}:{self.port}")
        log(f"Wake words: {', '.join(self.wake_word_names)}")
        log(f"Threshold: {WAKE_THRESHOLD}")
        log(f"Post-wake discard: {POST_WAKE_DELAY}s")
        log(f"Silence detection: {INITIAL_SILENCE_DURATION}s initial, {SHORT_SILENCE_DURATION}s after speech")
        
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=None
        ):
            await asyncio.Future()


def main():
    server = JeevesServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        log("Shutting down...")


if __name__ == '__main__':
    main()