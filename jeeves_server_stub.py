#!/usr/bin/env python3
"""
Jeeves Server Stub
Simple echo server to test audio streaming pipeline
"""

import asyncio
import websockets
import time


class AudioServerStub:
    def __init__(self, host: str = '0.0.0.0', port: int = 8765):
        self.host = host
        self.port = port
        self.bytes_received = 0
        self.last_report = time.time()

    async def handle_client(self, websocket, path=None):
        """Handle incoming audio stream"""
        client_addr = websocket.remote_address
        print(f"Client connected: {client_addr}")
        
        config = None
        self.bytes_received = 0
        self.last_report = time.time()
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    # Config or control message
                    if message.startswith("CONFIG:"):
                        config = message[7:]
                        print(f"Audio config: {config}")
                        await websocket.send("STATUS:Config received, ready for audio")
                    else:
                        print(f"Control message: {message}")
                else:
                    # Binary audio data
                    self.bytes_received += len(message)
                    
                    # Report stats every second
                    now = time.time()
                    if now - self.last_report >= 1.0:
                        kb_per_sec = (self.bytes_received / 1024) / (now - self.last_report)
                        await websocket.send(f"STATUS:Receiving {kb_per_sec:.1f} KB/s")
                        self.bytes_received = 0
                        self.last_report = now
                        
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {client_addr}")
        except Exception as e:
            print(f"Error handling client: {e}")

    async def run(self):
        """Start the WebSocket server"""
        print(f"Starting Jeeves stub server on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever


def main():
    server = AudioServerStub()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
