import asyncio
import websockets
import numpy as np
import torch
import whisper
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load Whisper model on the selected device
model = whisper.load_model("large", device=device)

async def audio_handler(websocket, path):
    logging.info("Client connected.")
    buffer = np.array([], dtype=np.int16)
    CHUNK_DURATION_MS = 500       # Chunk duration in milliseconds
    CHUNK_SIZE = int(16000 * CHUNK_DURATION_MS / 1000)  # 16kHz audio

    try:
        async for message in websocket:
            # Receive audio data from the client
            audio_data = np.frombuffer(message, dtype=np.int16)
            buffer = np.concatenate((buffer, audio_data))
            logging.info(f"Received audio data: {len(audio_data)} samples")

            # Process buffer when it reaches CHUNK_SIZE
            while len(buffer) >= CHUNK_SIZE:
                chunk = buffer[:CHUNK_SIZE]
                buffer = buffer[CHUNK_SIZE:]

                # Convert int16 to float32
                audio_float32 = chunk.astype(np.float32) / 32768.0
                logging.info(f"Processing chunk of size: {CHUNK_SIZE}")

                # Perform transcription asynchronously to avoid blocking
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, transcribe_chunk, audio_float32)
                text = result.strip()
                logging.info(f"Transcription result: {text}")

                # Send transcription back to client
                await websocket.send(text)
    except websockets.exceptions.ConnectionClosed as e:
        logging.info("Client disconnected.")

def transcribe_chunk(audio_float32):
    # Adjust fp16 parameter based on the device
    fp16 = device == "cuda"

    # Perform transcription
    result = model.transcribe(audio_float32, fp16=fp16, language='en')
    logging.info(f"Transcription result: {result}")
    return result['text']

async def main():
    async with websockets.serve(audio_handler, "localhost", 8000, max_size=2**25):
        logging.info("Server started.")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
