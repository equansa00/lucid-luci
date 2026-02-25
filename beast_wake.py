#!/usr/bin/env python3
"""
BEAST Wake Word Daemon — Sub-phase C
Listens for "hey jarvis" (openwakeword built-in, mapped to hey_beast),
then records, transcribes, routes, and speaks a response.

Dependencies:
  pip install openwakeword pyaudio numpy --break-system-packages
  sudo dnf install -y portaudio-devel python3-pyaudio
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WAKE_WORD = os.getenv("BEAST_WAKE_WORD", "hey_beast")
WAKE_THRESHOLD = float(os.getenv("BEAST_WAKE_THRESHOLD", "0.5"))
LISTEN_DURATION = int(os.getenv("BEAST_LISTEN_DURATION", "5"))
WAKE_MODEL_DIR = Path.home() / "beast" / "workspace" / "wake_models"

# ---------------------------------------------------------------------------
# Imports from mini_beast
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from mini_beast import (  # noqa: E402
    ollama_chat,
    route_model,
    format_model_tag,
    load_persona,
    tts_speak,
    stt_record,
    stt_transcribe,
    RUNS_DIR,
    BEAST_AUTO_MEMORY,
    auto_extract_memory,
)


# ---------------------------------------------------------------------------
# Wake model setup
# ---------------------------------------------------------------------------

def download_wake_model() -> Path:
    """Return path to the openwakeword model to use.

    openwakeword ships these built-in models:
      hey_jarvis, alexa, hey_mycroft, timer

    "hey_beast" is not built-in, so we use "hey_jarvis" as the closest
    match. Custom model training is possible via openwakeword custom model
    docs if you want a real "hey beast" wake word later.
    """
    WAKE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # openwakeword loads built-ins by name string — no file path needed.
    # We return WAKE_MODEL_DIR as a marker; the actual model is loaded
    # by name inside listen_for_wake_word().
    model_name = "hey_jarvis"
    print(f"[wake] Using built-in openwakeword model: {model_name}")
    print(f"[wake] (Mapped from BEAST_WAKE_WORD={WAKE_WORD!r})")
    return WAKE_MODEL_DIR / model_name


# ---------------------------------------------------------------------------
# Wake word detection
# ---------------------------------------------------------------------------

def listen_for_wake_word() -> bool:
    """Block until wake word detected or KeyboardInterrupt.

    Returns True on detection, False on KeyboardInterrupt.
    """
    try:
        import numpy as np
        import pyaudio
        from openwakeword.model import Model as OWWModel
    except ImportError as e:
        print(f"[wake] Missing dependency: {e}", file=sys.stderr)
        print("[wake] Run: pip install openwakeword pyaudio numpy --break-system-packages",
              file=sys.stderr)
        return False

    CHUNK = 1280
    RATE = 16000

    try:
        oww = OWWModel(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    except Exception as e:
        print(f"[wake] Failed to load openwakeword model: {e}", file=sys.stderr)
        return False

    pa = pyaudio.PyAudio()
    stream = None
    try:
        stream = pa.open(
            rate=RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=CHUNK,
        )
        print("👂 Listening for wake word...", flush=True)

        while True:
            try:
                raw = stream.read(CHUNK, exception_on_overflow=False)
            except OSError as e:
                print(f"[wake] Audio read error: {e}", file=sys.stderr)
                continue

            audio_chunk = np.frombuffer(raw, dtype=np.int16)
            scores = oww.predict(audio_chunk)

            # scores is a dict keyed by model name
            for model_name, score in scores.items():
                if score >= WAKE_THRESHOLD:
                    print(f"🔔 Wake word detected! (model={model_name}, score={score:.3f})",
                          flush=True)
                    return True

    except KeyboardInterrupt:
        return False
    except Exception as e:
        print(f"[wake] Unexpected error in listen loop: {e}", file=sys.stderr)
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
    print("🎤 Listening for command...", flush=True)
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
    print(f"\nBEAST: {response}", flush=True)
    if tag:
        print(tag, flush=True)

    # 6. Speak (without tag)
    tts_speak(response)

    # 7. Background memory extract
    if BEAST_AUTO_MEMORY:
        threading.Thread(
            target=auto_extract_memory,
            args=(text, response),
            daemon=True,
        ).start()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("BEAST Wake Word Daemon starting...", flush=True)
    print(f"Wake word:       hey jarvis (mapped to hey_beast)", flush=True)
    print(f"Threshold:       {WAKE_THRESHOLD}", flush=True)
    print(f"Listen duration: {LISTEN_DURATION}s", flush=True)
    print("Press Ctrl+C to stop\n", flush=True)

    while True:
        detected = listen_for_wake_word()
        if detected:
            handle_wake_activation()
        else:
            # KeyboardInterrupt or unrecoverable audio error
            print("\nWake word daemon stopped.", flush=True)
            break


if __name__ == "__main__":
    main()
