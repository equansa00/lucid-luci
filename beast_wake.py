#!/usr/bin/env python3
"""
BEAST Wake Word Daemon — Sub-phase C
Listens for "Hey BEAST", then records, transcribes, routes, and speaks
a response. Internally uses the hey_jarvis openwakeword model; that
detail is never surfaced to the user.

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
        print("👂 Listening for 'Hey BEAST'...", flush=True)

        while True:
            try:
                data = stream.read(1280, exception_on_overflow=False)
            except OSError as e:
                print(f"[wake] Audio read error: {e}", file=sys.stderr)
                continue

            chunk = np.frombuffer(data, dtype=np.int16)
            scores = model.predict(chunk)

            if scores.get(_WAKE_KEY, 0) > WAKE_THRESHOLD:
                print("🔔 Hey BEAST detected!", flush=True)
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
    print(f"Wake word:       Hey BEAST", flush=True)
    print(f"Threshold:       {WAKE_THRESHOLD}", flush=True)
    print(f"Listen duration: {LISTEN_DURATION}s", flush=True)
    print("Press Ctrl+C to stop\n", flush=True)

    while True:
        detected = listen_for_wake_word()
        if detected:
            handle_wake_activation()
        else:
            # KeyboardInterrupt or unrecoverable audio error
            print("\nBEAST wake word daemon stopped.", flush=True)
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
