#!/usr/bin/env python3
"""
LUCI Whisper STT Server
Loads Whisper once into GPU VRAM at startup.
Serves transcription requests at ~300ms instead of 10-30s cold load.
"""
import json
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path.home() / "beast" / "workspace" / ".env")

import torch
import whisper
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
import uvicorn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WHISPER_MODEL = os.getenv("LUCI_WHISPER_MODEL", "base.en")
WHISPER_PORT  = int(os.getenv("LUCI_WHISPER_PORT", "8765"))
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Load model ONCE at startup
# ---------------------------------------------------------------------------
print(f"[whisper-server] Loading {WHISPER_MODEL} on {DEVICE}...", flush=True)
_t0 = time.time()
model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
print(f"[whisper-server] ✅ Ready on {DEVICE} ({time.time()-_t0:.1f}s load time)", flush=True)

_FP16 = DEVICE == "cuda"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="LUCI Whisper Server")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "model": WHISPER_MODEL, "device": DEVICE})


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> JSONResponse:
    """Accept any audio file upload and return transcription."""
    suffix = ".webm"
    ct = audio.content_type or ""
    if "wav" in ct:
        suffix = ".wav"
    elif "ogg" in ct:
        suffix = ".ogg"
    elif "mp4" in ct or "m4a" in ct:
        suffix = ".m4a"

    tmp_path = None
    try:
        data = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        result = model.transcribe(
            tmp_path, fp16=_FP16, language="en",
            initial_prompt=(
                "LUCI is a personal AI assistant. "
                "Chip and Edward are the same person, the user. "
                "Ogechi is his wife. "
                "Names: LUCI, Chip, Edward, Ogechi, Andrew, Christopher, Athena. "
                "Common topics: medication, Zoloft, Lyrica, health, code, GitHub, projects."
            ),
        )
        text = (result.get("text") or "").strip()
        return JSONResponse({"text": text, "duration": result.get("duration", 0)})
    except Exception as e:
        return JSONResponse({"text": "", "error": str(e)}, status_code=500)
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/transcribe_path")
async def transcribe_path(request: Request) -> JSONResponse:
    """Accept JSON {"path": "/abs/path/to/file.wav"} and return transcription."""
    try:
        body = await request.json()
        path = body.get("path", "")
        if not path or not Path(path).exists():
            return JSONResponse({"text": "", "error": "file not found"}, status_code=400)
        result = model.transcribe(
            path, fp16=_FP16, language="en",
            initial_prompt=(
                "LUCI is a personal AI assistant. "
                "Chip and Edward are the same person, the user. "
                "Ogechi is his wife. "
                "Names: LUCI, Chip, Edward, Ogechi, Andrew, Christopher, Athena. "
                "Common topics: medication, Zoloft, Lyrica, health, code, GitHub, projects."
            ),
        )
        text = (result.get("text") or "").strip()
        return JSONResponse({"text": text})
    except Exception as e:
        return JSONResponse({"text": "", "error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=WHISPER_PORT, log_level="warning")
