#!/usr/bin/env python3
"""
LUCI Wake Word Daemon — Sub-phase C / Phase 9 Always-Listen
Two modes:
  • Wake-word mode (default): wait for "Hey LUCI", then record + respond
  • Always-listen mode (LUCI_ALWAYS_LISTEN=true): continuous mic monitoring
    with VAD, interrupt-while-speaking support, Whisper server STT

Dependencies:
  pip install openwakeword pyaudio numpy requests --break-system-packages
  sudo dnf install -y portaudio-devel python3-pyaudio
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WAKE_WORD        = os.getenv("LUCI_WAKE_WORD", "hey_beast")
WAKE_THRESHOLD   = float(os.getenv("LUCI_WAKE_THRESHOLD", "0.5"))
LISTEN_DURATION  = int(os.getenv("LUCI_LISTEN_DURATION", "5"))
WAKE_MODEL_DIR   = Path.home() / "beast" / "workspace" / "wake_models"
ALWAYS_LISTEN    = os.getenv("LUCI_ALWAYS_LISTEN", "false").lower() in ("1", "true", "yes")

# Always-listen VAD / recording config
AL_THRESHOLD  = float(os.getenv("LUCI_AL_THRESHOLD", "500"))   # RMS amplitude to detect speech
AL_SILENCE_MS = int(os.getenv("LUCI_AL_SILENCE_MS", "1200"))   # ms of silence to end recording
AL_CHUNK      = 1024
AL_RATE       = 16000   # 16 kHz mono for Whisper
AL_FORMAT_INT = None    # filled after pyaudio import

WHISPER_PORT = int(os.getenv("LUCI_WHISPER_PORT", "8765"))
WHISPER_URL  = f"http://127.0.0.1:{WHISPER_PORT}/transcribe"

# ---------------------------------------------------------------------------
# Imports from luci
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from luci import (  # noqa: E402
    ollama_chat,
    route_model,
    format_model_tag,
    load_persona,
    tts_speak,
    tts_to_file,
    stt_record,
    stt_transcribe,
    RUNS_DIR,
    DEFAULT_VOICE,
    LUCI_AUTO_MEMORY,
    auto_extract_memory,
)

# ---------------------------------------------------------------------------
# Internal wake model constants (never shown to user)
# ---------------------------------------------------------------------------
import openwakeword as _owow  # noqa: E402

_OWOW_MODEL_PATH = (
    Path(_owow.__file__).parent
    / "resources" / "models" / "hey_jarvis_v0.1.onnx"
)
_WAKE_KEY = "hey_jarvis_v0.1"  # prediction dict key, internal only


# ---------------------------------------------------------------------------
# Wake word detection
# ---------------------------------------------------------------------------

def listen_for_wake_word() -> bool:
    """Block until wake word detected or KeyboardInterrupt.

    Returns True on detection, False on KeyboardInterrupt/error.
    """
    try:
        import numpy as np
        import pyaudio
        from openwakeword.model import Model
    except ImportError as e:
        print(f"[wake] Missing dependency: {e}", file=sys.stderr)
        print("[wake] Run: pip install openwakeword pyaudio numpy --break-system-packages",
              file=sys.stderr)
        return False

    if not _OWOW_MODEL_PATH.exists():
        print(
            f"[wake] Wake model not found: {_OWOW_MODEL_PATH}\n"
            "[wake] Run: pip install --upgrade openwakeword --break-system-packages",
            file=sys.stderr,
        )
        return False

    try:
        model = Model(wakeword_model_paths=[str(_OWOW_MODEL_PATH)])
    except Exception as e:
        print(f"[wake] Failed to load wake model: {e}", file=sys.stderr)
        return False

    pa = pyaudio.PyAudio()
    stream = None
    try:
        stream = pa.open(
            rate=16000,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=1280,
        )
        print("👂 Listening for 'Hey LUCI'...", flush=True)

        while True:
            try:
                data = stream.read(1280, exception_on_overflow=False)
            except OSError as e:
                print(f"[wake] Audio read error: {e}", file=sys.stderr)
                continue

            chunk = np.frombuffer(data, dtype=np.int16)
            scores = model.predict(chunk)

            if scores.get(_WAKE_KEY, 0) > WAKE_THRESHOLD:
                print("🔔 Hey LUCI detected!", flush=True)
                return True

    except KeyboardInterrupt:
        return False
    except Exception as e:
        print(f"[wake] Unexpected error: {e}", file=sys.stderr)
        return False
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        try:
            pa.terminate()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Wake activation pipeline
# ---------------------------------------------------------------------------

def handle_wake_activation() -> None:
    """Full pipeline: confirm wake → record → transcribe → route → respond → speak."""

    # 1. Activation acknowledgement
    tts_speak("Yes?")

    # 2. Record command
    print("🎤 Listening for your command...", flush=True)
    try:
        audio_path = stt_record(LISTEN_DURATION)
    except Exception as e:
        print(f"[wake] Recording failed: {e}", file=sys.stderr)
        tts_speak("Sorry, I couldn't record audio.")
        return

    # 3. Transcribe
    text = stt_transcribe(audio_path)
    if not text:
        tts_speak("Sorry, I didn't catch that.")
        return
    print(f"You said: {text}", flush=True)

    # 4. Route + respond
    routed_model, category = route_model(text)
    persona = load_persona()
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    messages.append({"role": "user", "content": text})
    try:
        response = ollama_chat(messages, temperature=0.4, model=routed_model)
    except Exception as e:
        response = f"Error: {e}"

    tag = format_model_tag(routed_model, category)

    # 5. Print
    print(f"\nLUCI: {response}", flush=True)
    if tag:
        print(tag, flush=True)

    # 6. Speak (without tag)
    tts_speak(response)

    # 7. Background memory extract
    if LUCI_AUTO_MEMORY:
        threading.Thread(
            target=auto_extract_memory,
            args=(text, response),
            daemon=True,
        ).start()


# ---------------------------------------------------------------------------
# Always-listen loop (Phase 9 Workstream E)
# ---------------------------------------------------------------------------

# Shared state for interrupt
_al_aplay_proc: subprocess.Popen | None = None
_al_speaking = threading.Event()


def _al_interrupt() -> None:
    """Kill any active aplay process immediately."""
    global _al_aplay_proc
    if _al_aplay_proc is not None:
        try:
            _al_aplay_proc.kill()
        except Exception:
            pass
        _al_aplay_proc = None
    _al_speaking.clear()


def _al_speak_wav(wav_path: Path) -> None:
    """Play a WAV file via aplay (blocking in this thread). Interruptible."""
    global _al_aplay_proc
    _al_speaking.set()
    try:
        _al_aplay_proc = subprocess.Popen(
            ["aplay", "-q", str(wav_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _al_aplay_proc.wait()
    except Exception as e:
        print(f"[al] aplay error: {e}", file=sys.stderr)
    finally:
        _al_aplay_proc = None
        _al_speaking.clear()


def _al_transcribe_wav(wav_path: Path) -> str:
    """POST a WAV file to the Whisper server, return transcription text."""
    try:
        import requests as _req
        with open(wav_path, "rb") as f:
            resp = _req.post(
                WHISPER_URL,
                files={"audio": ("clip.wav", f, "audio/wav")},
                timeout=30,
            )
        if resp.ok:
            return resp.json().get("text", "").strip()
    except Exception as e:
        print(f"[al] Whisper error: {e}", file=sys.stderr)
    return ""


def _al_process(frames: list[bytes], pa_format: int, rate: int) -> None:
    """
    Background thread: save frames → WAV → Whisper → Ollama → TTS → aplay.
    """
    # Write frames to temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        import pyaudio
        sample_width = pyaudio.PyAudio().get_sample_size(pa_format)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))
    except Exception as e:
        print(f"[al] WAV write error: {e}", file=sys.stderr)
        wav_path.unlink(missing_ok=True)
        return

    # Minimum duration filter — skip clips under 0.8s (fan noise bursts)
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            duration_ms = (wf.getnframes() / wf.getframerate()) * 1000
        if duration_ms < 800:
            print(f"[al] clip too short ({duration_ms:.0f}ms), skipping", flush=True)
            wav_path.unlink(missing_ok=True)
            return
    except Exception:
        pass

    # Transcribe
    text = _al_transcribe_wav(wav_path)
    wav_path.unlink(missing_ok=True)

    if not text:
        print("[al] No speech detected or transcription empty.", flush=True)
        return

    # Filter single-word garbage / common mishears
    JUNK_PHRASES = {"you", "the", "the and", "and", "a", "i", "ok", "okay",
                    "um", "uh", "hmm", "hm", "oh", "ah", "hey"}
    if text.lower().strip(".,!?") in JUNK_PHRASES:
        print(f"[al] filtered junk: '{text}'", flush=True)
        return
    if len(text.split()) < 2:
        print(f"[al] too short to process: '{text}'", flush=True)
        return

    print(f"\nYou: {text}", flush=True)

    # Route + respond
    routed_model, category = route_model(text)
    persona = load_persona()
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    messages.append({"role": "user", "content": text})
    try:
        response = ollama_chat(messages, temperature=0.4, model=routed_model)
    except Exception as e:
        response = f"Error: {e}"

    tag = format_model_tag(routed_model, category)
    print(f"LUCI: {response}", flush=True)
    if tag:
        print(tag, flush=True)

    # TTS → temp WAV → aplay
    reply_path = RUNS_DIR / f"al_reply_{int(time.time())}.wav"
    ok = tts_to_file(response, reply_path, voice=DEFAULT_VOICE)
    if ok:
        _al_speak_wav(reply_path)
        try:
            reply_path.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        print("[al] TTS failed — falling back to tts_speak", flush=True)
        tts_speak(response)

    # Background memory
    if LUCI_AUTO_MEMORY:
        threading.Thread(
            target=auto_extract_memory,
            args=(text, response),
            daemon=True,
        ).start()


def always_listen_loop() -> None:
    """
    Continuous VAD loop. Records when RMS > AL_THRESHOLD, submits after
    AL_SILENCE_MS of silence. Interrupts playback if new speech detected.
    """
    try:
        import numpy as np
        import pyaudio
    except ImportError as e:
        print(f"[al] Missing dependency: {e}", file=sys.stderr)
        print("[al] Run: pip install pyaudio numpy --break-system-packages", file=sys.stderr)
        return

    pa = pyaudio.PyAudio()
    pa_format = pyaudio.paInt16
    pa_sample_width = pa.get_sample_size(pa_format)

    stream = None
    try:
        stream = pa.open(
            rate=AL_RATE,
            channels=1,
            format=pa_format,
            input=True,
            frames_per_buffer=AL_CHUNK,
        )
    except Exception as e:
        print(f"[al] Failed to open mic: {e}", file=sys.stderr)
        pa.terminate()
        return

    print("🎙️  Always-listening active. Speak anytime — I'm here.", flush=True)
    print(f"    VAD threshold: {AL_THRESHOLD} RMS | Silence timeout: {AL_SILENCE_MS}ms\n",
          flush=True)

    recording   = False
    frames: list[bytes] = []
    silence_start: float | None = None
    silence_ms_per_chunk = (AL_CHUNK / AL_RATE) * 1000  # ms per chunk

    try:
        while True:
            try:
                data = stream.read(AL_CHUNK, exception_on_overflow=False)
            except OSError as e:
                print(f"[al] Audio read error: {e}", file=sys.stderr)
                time.sleep(0.05)
                continue

            # Echo gate: skip normal VAD while LUCI is speaking to prevent echo loop
            if _al_speaking.is_set():
                chunk = np.frombuffer(data, dtype=np.int16)
                rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
                if rms > AL_THRESHOLD * 2.5:  # deliberate interruption (very loud)
                    print("⚡ Interrupted.", flush=True)
                    _al_interrupt()
                continue  # skip normal VAD processing while speaking

            chunk = np.frombuffer(data, dtype=np.int16)
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            is_speech = rms > AL_THRESHOLD

            if is_speech:

                if not recording:
                    recording = True
                    frames = []
                    print("🔴 Recording...", flush=True)

                frames.append(data)
                silence_start = None

            elif recording:
                frames.append(data)  # include trailing silence

                if silence_start is None:
                    silence_start = time.time()
                else:
                    elapsed_ms = (time.time() - silence_start) * 1000
                    if elapsed_ms >= AL_SILENCE_MS:
                        # End of utterance — dispatch to background thread
                        recording = False
                        print("⏹  Processing...", flush=True)
                        captured = frames[:]
                        threading.Thread(
                            target=_al_process,
                            args=(captured, pa_format, AL_RATE),
                            daemon=True,
                        ).start()
                        frames = []
                        silence_start = None

    except KeyboardInterrupt:
        pass
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        try:
            pa.terminate()
        except Exception:
            pass
        _al_interrupt()
        print("\n[al] Always-listen stopped.", flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    if ALWAYS_LISTEN:
        print("LUCI Always-Listen Daemon starting...", flush=True)
        print(f"VAD threshold:   {AL_THRESHOLD} RMS", flush=True)
        print(f"Silence timeout: {AL_SILENCE_MS}ms", flush=True)
        print(f"Whisper server:  {WHISPER_URL}", flush=True)
        print(f"Voice:           {DEFAULT_VOICE}", flush=True)
        print("Press Ctrl+C to stop\n", flush=True)
        always_listen_loop()
    else:
        print("LUCI Wake Word Daemon starting...", flush=True)
        print(f"Wake word:       Hey LUCI", flush=True)
        print(f"Threshold:       {WAKE_THRESHOLD}", flush=True)
        print(f"Listen duration: {LISTEN_DURATION}s", flush=True)
        print("Press Ctrl+C to stop\n", flush=True)

        while True:
            detected = listen_for_wake_word()
            if detected:
                handle_wake_activation()
            else:
                # KeyboardInterrupt or unrecoverable audio error
                print("\nLUCI wake word daemon stopped.", flush=True)
                break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list-models":
        try:
            import openwakeword as _oww
            models_dir = Path(_oww.__file__).parent / "resources" / "models"
            print("Available openwakeword models:")
            for f in sorted(models_dir.glob("*.onnx")):
                print(f"  {f.name}")
        except Exception as e:
            print(f"❌ {e}", file=sys.stderr)
            sys.exit(1)
    else:
        main()
