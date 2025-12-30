#!/usr/bin/env python3
"""
Jeeves STT Server
Multi-mode speech-to-text: Command mode (Whisper) + Transcription mode
"""

import asyncio
import websockets
import numpy as np
import time
import re
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import whisper
import torch
from openwakeword.model import Model as WakeWordModel

# Audio config
SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2

# Wake word settings
WAKE_THRESHOLD = 0.5
POST_WAKE_DELAY = 0.3
POST_COMMAND_COOLDOWN = 0.5  # Ignore wake words for 0.5s after command completes

# Command mode settings
SILENCE_THRESHOLD = 0.01
MIN_SPEECH_DURATION = 0.3
INITIAL_SILENCE_DURATION = 1.5
SHORT_SILENCE_DURATION = 0.7
MAX_COMMAND_DURATION = 15.0

# Transcription mode settings
TRANSCRIPTION_CHUNK_DURATION = 5.0
MAX_TRANSCRIPTION_DURATION = 300.0
STOP_PHRASES = [
    "stop transcription",
    "end transcription",
    "stop dictation",
    "end dictation",
    "that's all",
    "that is all",
    "finish transcription",
    "done transcribing",
]

TRANSCRIPTION_TRIGGERS = [
    "enter transcription mode",
    "start transcription",
    "transcription mode",
    "start dictating",
    "dictation mode",
]

DEBUG = True


def log(msg: str, level: str = "INFO"):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)


def debug(msg: str):
    if DEBUG:
        log(msg, "DEBUG")


class Mode(Enum):
    LISTENING = "listening"
    COMMAND = "command"
    TRANSCRIPTION = "transcription"


class JeevesServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 8765):
        self.host = host
        self.port = port
        
        log("Checking CUDA availability...")
        log(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        
        log("Loading models...")
        
        # Whisper for command mode
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"  Loading Whisper model (small) on {device}...")
        load_start = time.time()
        self.whisper_model = whisper.load_model("small", device=device)
        log(f"  Whisper loaded in {time.time() - load_start:.1f}s")
        
        # TODO: Add Parakeet for transcription mode
        log("  Parakeet not yet implemented - using Whisper for transcription mode")
        
        # openWakeWord - no inference_framework argument in newer versions
        log("  Loading openWakeWord...")
        load_start = time.time()
        self.wake_model = WakeWordModel()
        self.wake_word_names = list(self.wake_model.models.keys())
        log(f"  Wake words: {self.wake_word_names}")
        log(f"  openWakeWord loaded in {time.time() - load_start:.1f}s")
        
        log("All models loaded!")

    def get_audio_level(self, audio_bytes: bytes) -> float:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return np.sqrt(np.mean(audio ** 2))

    def is_silence(self, audio_bytes: bytes) -> bool:
        return self.get_audio_level(audio_bytes) < SILENCE_THRESHOLD

    def detect_wake_word(self, audio_bytes: bytes) -> tuple[bool, float, str, dict]:
        """Detect wake word and return prediction scores for all models"""
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
        
        return detected_word is not None, max_score, detected_word, prediction

    def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio using Whisper"""
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        if len(audio) < SAMPLE_RATE * 0.1:
            return ""
        
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
        
        options = whisper.DecodingOptions(language="en", fp16=torch.cuda.is_available())
        result = whisper.decode(self.whisper_model, mel, options)
        
        return result.text.strip()

    def check_transcription_trigger(self, text: str) -> bool:
        text_lower = text.lower()
        for trigger in TRANSCRIPTION_TRIGGERS:
            if trigger in text_lower:
                return True
        return False

    def check_stop_phrase(self, text: str) -> tuple[bool, str]:
        text_lower = text.lower()
        for phrase in STOP_PHRASES:
            if phrase in text_lower:
                cleaned = re.sub(re.escape(phrase), '', text_lower, flags=re.IGNORECASE).strip()
                return True, cleaned
        return False, text

    async def handle_client(self, websocket, path=None):
        client_addr = websocket.remote_address
        log(f"Client connected: {client_addr}")
        
        self.wake_model.reset()
        
        # Session diagnostics for bug isolation
        session_wake_count = 0
        session_transcription_count = 0
        consecutive_empty_transcriptions = 0
        last_transcription_text = ""
        consecutive_silence_chunks = 0  # Track consecutive silent chunks
        
        mode = Mode.LISTENING
        audio_buffer = bytearray()
        transcription_buffer = bytearray()
        full_transcription = []
        
        silence_start: Optional[float] = None
        recording_start: Optional[float] = None
        last_chunk_time: Optional[float] = None
        speech_detected = False
        post_wake_discard_until: Optional[float] = None
        post_command_cooldown_until: Optional[float] = None
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    if message.startswith("CONFIG:"):
                        log(f"Client config: {message[7:]}")
                        await websocket.send("STATUS:Ready - say 'Alexa' to activate")
                        continue
                    elif message.startswith("TRIGGER:"):
                        log("Manual trigger received")
                        await websocket.send("STATUS:Recording command...")
                        audio_buffer = bytearray()
                        recording_start = time.time()
                        silence_start = None
                        speech_detected = False
                        post_wake_discard_until = None
                        mode = Mode.COMMAND
                        continue
                
                # Make a copy of audio chunk to avoid BufferError from numpy views
                audio_chunk = bytes(message)
                now = time.time()
                
                # === DIAGNOSTIC: Audio level logging ===
                chunk_level = self.get_audio_level(audio_chunk)
                chunk_is_silence = self.is_silence(audio_chunk)
                chunk_duration = len(audio_chunk) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                chunk_db = 20 * np.log10(chunk_level + 1e-10)  # Convert to dB
                
                if chunk_is_silence:
                    consecutive_silence_chunks += 1
                else:
                    consecutive_silence_chunks = 0
                
                debug(f"AUDIO_LEVEL: {chunk_db:.1f}dB chunk={len(audio_chunk)} bytes "
                      f"{'SILENCE' if chunk_is_silence else 'SPEECH'} "
                      f"(consecutive={consecutive_silence_chunks})")
                
                # === LISTENING MODE ===
                if mode == Mode.LISTENING:
                    # Check if we're in post-command cooldown period
                    if post_command_cooldown_until and now < post_command_cooldown_until:
                        debug(f"POST_COMMAND_COOLDOWN: ignoring wake detection for {post_command_cooldown_until - now:.2f}s")
                        # Still feed audio to wake model to keep it updated, but ignore detections
                        audio_for_wake = np.frombuffer(audio_chunk, dtype=np.int16)
                        self.wake_model.predict(audio_for_wake)
                        continue
                    
                    post_command_cooldown_until = None
                    
                    # Only scan for wake words in LISTENING mode (call predict() only once)
                    detected, score, wake_word, wake_prediction = self.detect_wake_word(audio_chunk)
                    
                    # DIAGNOSTIC: Log all wake word scores (only in LISTENING mode)
                    for wname in self.wake_word_names:
                        wscore = float(wake_prediction.get(wname, 0))
                        debug(f"WAKE_SCAN: {wname}={wscore:.2f}, threshold={WAKE_THRESHOLD} "
                              f"{'→ DETECTED' if wscore > WAKE_THRESHOLD else ''}")
                    
                    if detected:
                        session_wake_count += 1
                        log(f"Wake word '{wake_word}' detected (score: {score:.2f}) [wake #{session_wake_count}]")
                        await websocket.send(f"WAKE:{wake_word}")
                        
                        audio_buffer = bytearray()
                        recording_start = now
                        post_wake_discard_until = now + POST_WAKE_DELAY
                        silence_start = None
                        speech_detected = False
                        mode = Mode.COMMAND
                        continue
                
                # === COMMAND MODE ===
                elif mode == Mode.COMMAND:
                    elapsed = now - recording_start
                    
                    if post_wake_discard_until and now < post_wake_discard_until:
                        continue
                    post_wake_discard_until = None
                    
                    audio_buffer.extend(audio_chunk)
                    
                    if not chunk_is_silence:
                        if not speech_detected:
                            debug("Speech detected - first non-silence chunk")
                        speech_detected = True
                        silence_start = None
                    else:
                        if silence_start is None:
                            silence_start = now
                            debug(f"Silence started at {now:.3f}")
                        
                        silence_duration = INITIAL_SILENCE_DURATION if not speech_detected else SHORT_SILENCE_DURATION
                        current_silence_duration = now - silence_start
                        
                        # DIAGNOSTIC: Log silence state
                        debug(f"SILENCE_WAIT: duration={current_silence_duration:.2f}s, "
                              f"threshold={silence_duration:.2f}s, speech_detected={speech_detected}")
                        
                        if now - silence_start >= silence_duration:
                            await websocket.send("STATUS:Processing command...")
                            
                            buffer_duration = len(audio_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                            
                            # DIAGNOSTIC: Detailed buffer state before transcription
                            debug(f"BUFFER_STATE: duration={buffer_duration:.2f}s, "
                                  f"speech_detected={speech_detected}, "
                                  f"silence_duration={current_silence_duration:.2f}s, "
                                  f"consecutive_empty={consecutive_empty_transcriptions}")
                            
                            if buffer_duration < MIN_SPEECH_DURATION:
                                debug(f"BUFFER_TOO_SHORT: {buffer_duration:.2f}s < {MIN_SPEECH_DURATION}s minimum")
                                await websocket.send("STATUS:No command detected")
                            else:
                                log(f"Transcribing command ({buffer_duration:.1f}s, speech={speech_detected})...")
                                start = time.time()
                                text = self.transcribe_audio(bytes(audio_buffer))
                                transcribe_time = time.time() - start
                                log(f"Command transcribed in {transcribe_time:.2f}s: '{text}'")
                                
                                # DIAGNOSTIC: Track consecutive empty transcriptions
                                if not text or text.strip() == "":
                                    consecutive_empty_transcriptions += 1
                                    debug(f"EMPTY_TRANSCRIPTION: #{consecutive_empty_transcriptions} in this session")
                                    if consecutive_empty_transcriptions > 3:
                                        log(f"WARN: {consecutive_empty_transcriptions} consecutive empty transcriptions - possible loop", "WARN")
                                else:
                                    consecutive_empty_transcriptions = 0
                                
                                if text:
                                    # Check for suspicious silence-only transcription
                                    if not speech_detected and consecutive_empty_transcriptions > 1:
                                        log("WARN: Empty transcription with no speech detected - check audio source", "WARN")
                                    
                                    if self.check_transcription_trigger(text):
                                        session_transcription_count += 1
                                        log(f"Entering transcription mode [session trans #{session_transcription_count}]")
                                        await websocket.send("MODE:transcription")
                                        await websocket.send("STATUS:Transcription mode - say 'stop transcription' when done")
                                        
                                        transcription_buffer = bytearray()
                                        full_transcription = []
                                        recording_start = now
                                        last_chunk_time = now
                                        silence_start = None
                                        speech_detected = False
                                        mode = Mode.TRANSCRIPTION
                                        continue
                                    
                                    await websocket.send(f"COMMAND:{text}")
                                else:
                                    await websocket.send("STATUS:No speech detected")
                            
                            audio_buffer.clear()
                            self.wake_model.reset()
                            post_command_cooldown_until = now + POST_COMMAND_COOLDOWN
                            mode = Mode.LISTENING
                            await websocket.send("STATUS:Listening...")
                            continue
                    
                    if elapsed >= MAX_COMMAND_DURATION:
                        log(f"Max command duration reached ({elapsed:.1f}s)")
                        await websocket.send("STATUS:Processing command...")
                        
                        # DIAGNOSTIC: Log final buffer state
                        final_buffer_duration = len(audio_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                        debug(f"MAX_DURATION_BUFFER: {final_buffer_duration:.2f}s, speech_detected={speech_detected}")
                        
                        text = self.transcribe_audio(bytes(audio_buffer))
                        
                        if text:
                            if self.check_transcription_trigger(text):
                                session_transcription_count += 1
                                log(f"Entering transcription mode [session trans #{session_transcription_count}]")
                                await websocket.send("MODE:transcription")
                                await websocket.send("STATUS:Transcription mode - say 'stop transcription' when done")
                                
                                transcription_buffer = bytearray()
                                full_transcription = []
                                recording_start = now
                                last_chunk_time = now
                                mode = Mode.TRANSCRIPTION
                                continue
                            
                            await websocket.send(f"COMMAND:{text}")
                        
                        audio_buffer.clear()
                        self.wake_model.reset()
                        post_command_cooldown_until = now + POST_COMMAND_COOLDOWN
                        mode = Mode.LISTENING
                        await websocket.send("STATUS:Listening...")
                
                # === TRANSCRIPTION MODE ===
                elif mode == Mode.TRANSCRIPTION:
                    elapsed = now - recording_start
                    transcription_buffer.extend(audio_chunk)
                    
                    buffer_duration = len(transcription_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                    
                    if buffer_duration >= TRANSCRIPTION_CHUNK_DURATION:
                        log(f"Processing transcription chunk ({buffer_duration:.1f}s)...")
                        start = time.time()
                        chunk_text = self.transcribe_audio(bytes(transcription_buffer))
                        log(f"Chunk transcribed in {time.time() - start:.2f}s")
                        
                        transcription_buffer.clear()
                        last_chunk_time = now
                        
                        if chunk_text:
                            stopped, cleaned_text = self.check_stop_phrase(chunk_text)
                            
                            if cleaned_text:
                                full_transcription.append(cleaned_text)
                                await websocket.send(f"PARTIAL:{cleaned_text}")
                            
                            if stopped:
                                log("Stop phrase detected - exiting transcription mode")
                                
                                final_text = " ".join(full_transcription)
                                await websocket.send(f"TRANSCRIPTION:{final_text}")
                                await websocket.send("MODE:command")
                                
                                full_transcription = []
                                self.wake_model.reset()
                                post_command_cooldown_until = now + POST_COMMAND_COOLDOWN
                                mode = Mode.LISTENING
                                await websocket.send("STATUS:Listening...")
                                continue
                    
                    if elapsed >= MAX_TRANSCRIPTION_DURATION:
                        log("Max transcription duration reached")
                        
                        if len(transcription_buffer) > SAMPLE_RATE * BYTES_PER_SAMPLE * 0.5:
                            chunk_text = self.transcribe_audio(bytes(transcription_buffer))
                            if chunk_text:
                                _, cleaned_text = self.check_stop_phrase(chunk_text)
                                if cleaned_text:
                                    full_transcription.append(cleaned_text)
                        
                        final_text = " ".join(full_transcription)
                        await websocket.send(f"TRANSCRIPTION:{final_text}")
                        await websocket.send("MODE:command")
                        
                        full_transcription = []
                        transcription_buffer.clear()
                        self.wake_model.reset()
                        post_command_cooldown_until = now + POST_COMMAND_COOLDOWN
                        mode = Mode.LISTENING
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
        log(f"Modes: Command (Whisper), Transcription (chunked)")
        log(f"Transcription triggers: {TRANSCRIPTION_TRIGGERS}")
        log(f"Stop phrases: {STOP_PHRASES}")
        
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
