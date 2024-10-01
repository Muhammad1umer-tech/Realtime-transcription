import asyncio
import websockets
import numpy as np
import torch
import whisper
import logging
import os
from threading import Lock

# Import Silero VAD
import torch.nn.functional as F
from scipy.signal import resample

# Configure logging
logging.basicConfig(level=logging.INFO)

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load a smaller Whisper model for faster inference
model = whisper.load_model("small", device=device)

# Predefine decoding options for faster execution
decoding_options = whisper.DecodingOptions(
    language='en',
    fp16=device == "cuda",
    temperature=0.0,
    beam_size=1,
    without_timestamps=True,
)

# Load Silero VAD model
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False)
(get_speech_timestamps, _, _, _, _) = utils

# Lock for thread-safe model inference
model_lock = Lock()

async def audio_handler(websocket, path):
    logging.info("Client connected.")
    buffer = np.array([], dtype=np.int16)
    CHUNK_DURATION_MS = 1000       # Chunk duration in milliseconds
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
                logging.info(f"Processing chunk of size: {len(chunk)}")

                # Perform VAD to detect speech segments
                speech_timestamps = await asyncio.get_event_loop().run_in_executor(
                    None, vad, audio_float32
                )

                # If speech is detected, transcribe the speech segments
                if speech_timestamps:
                    for segment in speech_timestamps:
                        start = segment['start']
                        end = segment['end']
                        # Extract the speech segment
                        speech_segment = audio_float32[int(start * 16000):int(end * 16000)]
                        # Transcribe the speech segment
                        loop = asyncio.get_running_loop()
                        text = await loop.run_in_executor(None, transcribe_chunk, speech_segment)
                        text = text.strip()
                        logging.info(f"Transcription result: {text}")

                        # Send transcription back to client
                        await websocket.send(text)
                else:
                    logging.info("No speech detected in the chunk.")
    except websockets.exceptions.ConnectionClosed as e:
        logging.info("Client disconnected.")

def vad(audio_float32):
    # Resample audio to 16000 Hz if necessary
    if audio_float32.shape[0] != 16000:
        audio_float32 = resample(audio_float32, int(len(audio_float32) * 16000 / 16000))

    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio_float32).to(device)

    # Apply VAD model
    with torch.no_grad():
        speech_timestamps = get_speech_timestamps(
            audio_tensor, vad_model, sampling_rate=16000
        )

    return speech_timestamps

def transcribe_chunk(audio_float32):
    # Convert numpy array to torch tensor
    audio_tensor = torch.from_numpy(audio_float32).to(device)

    # Compute the mel spectrogram
    mel = whisper.log_mel_spectrogram(audio_tensor)

    # Perform decoding
    with model_lock:
        result = whisper.decode(model, mel, decoding_options)

    logging.info(f"Transcription result: {result.text}")
    return result.text

async def main():
    port = int(os.getenv("PORT", 8900))
    async with websockets.serve(audio_handler, "localhost", port, max_size=2**25):
        logging.info("Server started.")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
