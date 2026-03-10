#!/usr/bin/env python3
"""
LUCI Whisper Server — faster-whisper + large-v3-turbo
Persistent GPU server so model loads once, transcribes fast.
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

MODEL_SIZE   = os.getenv("LUCI_WHISPER_MODEL", "large-v3-turbo")
DEVICE       = os.getenv("LUCI_WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("LUCI_WHISPER_COMPUTE", "float16")
PORT         = int(os.getenv("LUCI_WHISPER_PORT", "8765"))

INITIAL_PROMPT = (
    "LUCI is pronounced 'Lucy'. LUCI is a personal AI assistant. "
    "Chip and Edward are the same person, the user. "
    "Ogechi is his wife. "
    "Names: LUCI, Chip, Edward, Ogechi, Andrew, Christopher, Athena. "
    "Common topics: medication, Zoloft, Lyrica, health, code, GitHub, projects."
)

print(f"Loading faster-whisper {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})...", flush=True)
from faster_whisper import WhisperModel
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print(f"Model loaded.", flush=True)

app = FastAPI()

def _transcribe(audio_path: str) -> tuple[str, float]:
    """Run transcription, return (text, duration)."""
    segments, info = model.transcribe(
        audio_path,
        language="en",
        initial_prompt=INITIAL_PROMPT,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.duration

@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
    })

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        try:
            text, duration = _transcribe(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return JSONResponse({"text": text, "duration": duration})
    except Exception as e:
        return JSONResponse({"text": "", "error": str(e)}, status_code=500)

@app.post("/transcribe_path")
async def transcribe_path(body: dict):
    try:
        path = body.get("path", "")
        if not path or not Path(path).exists():
            return JSONResponse({"text": "", "error": "file not found"}, status_code=400)
        text, duration = _transcribe(path)
        return JSONResponse({"text": text, "duration": duration})
    except Exception as e:
        return JSONResponse({"text": "", "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
