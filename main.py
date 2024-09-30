import torch
import whisper

import warnings
warnings.filterwarnings(
    "ignore",
    message=(
        "You are using `torch.load` with `weights_only=False`.*"
    ),
    category=FutureWarning,
)

# Force the use of CPU
device = torch.device('cpu')
print("Using CPU")

model = whisper.load_model("tiny", device=device)
result = model.transcribe("Power_English_Update.mp3", fp16=False)
print("Transcription: \n", result["text"])
