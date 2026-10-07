"""Transcribe WAVs with the cached faster-whisper model, to check intelligibility without listening."""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

model = WhisperModel("Systran/faster-distil-whisper-medium.en", device="cpu", compute_type="int8", local_files_only=True)
for path in sys.argv[1:]:
    audio, sr = sf.read(path, dtype="float32")
    audio = resample_poly(audio, 16000, sr).astype(np.float32)
    segments, _ = model.transcribe(audio, language="en", beam_size=5)
    print(f"{path}: {' '.join(s.text.strip() for s in segments)}")
