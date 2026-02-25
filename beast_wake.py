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

def _find_builtin_model(name_fragment: str = "hey_jarvis") -> Path:
    """Locate a built-in openwakeword .onnx model by name fragment.

    openwakeword 0.4 ships models inside its package resources:
      hey_jarvis_v0.1.onnx, alexa_v0.1.onnx, hey_mycroft_v0.1.onnx,
      hey_marvin_v0.1.onnx, timer_v0.1.onnx, weather_v0.1.onnx

    Returns the first matching Path, or raises FileNotFoundError.
    """
    import openwakeword as _oww
    pkg_dir = Path(_oww.__file__).parent
    matches = sorted(pkg_dir.rglob(f"{name_fragment}*.onnx"))
    # Exclude the shared embedding / melspectrogram models
    matches = [p for p in matches if "embedding" not in p.name and "melspectrogram" not in p.name]
    if not matches:
        raise FileNotFoundError(
            f"No built-in openwakeword model matching {name_fragment!r}. "
            f"Available: {[p.name for p in pkg_dir.rglob('*.onnx')]}"
        )
    return matches[0]


def download_wake_model() -> Path:
    """Return path to the hey_jarvis built-in model (proxy for hey_beast).

    openwakeword 0.4 API requires wakeword_model_paths=[path_to_onnx].
    "hey_beast" is not built-in; hey_jarvis is the closest match.
    """
    WAKE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _find_builtin_model("hey_jarvis")
    print(f"[wake] Using built-in model: {path.name}")
    print(f"[wake] (Mapped from BEAST_WAKE_WORD={WAKE_WORD!r})")
    return path


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

    # Locate the built-in hey_jarvis .onnx file
    try:
        model_path = _find_builtin_model("hey_jarvis")
    except FileNotFoundError as e:
        print(f"[wake] {e}", file=sys.stderr)
        return False

    # openwakeword 0.4 API: pass explicit file paths
    try:
        oww = OWWModel(wakeword_model_paths=[str(model_path)])
    except Exception as e:
        print(f"[wake] Failed to load openwakeword model: {e}", file=sys.stderr)
        return False

    # Discover the prediction key on the first chunk
    _wake_key: str = ""

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

        first_pred = True
        while True:
            try:
                raw = stream.read(CHUNK, exception_on_overflow=False)
            except OSError as e:
                print(f"[wake] Audio read error: {e}", file=sys.stderr)
                continue

            audio_chunk = np.frombuffer(raw, dtype=np.int16)
            scores = oww.predict(audio_chunk)

            # On first prediction: print available keys and pick best match
            nonlocal_key = _wake_key  # capture for closure
            if first_pred:
                first_pred = False
                keys = list(scores.keys())
                print(f"[wake] Available model keys: {keys}", flush=True)
                # Prefer key containing "jarvis", else first key
                jarvis_keys = [k for k in keys if "jarvis" in k.lower()]
                nonlocal_key = jarvis_keys[0] if jarvis_keys else (keys[0] if keys else "")
                print(f"[wake] Using key: {nonlocal_key!r}", flush=True)

            # Check all keys; use discovered key for threshold
            for key, score in scores.items():
                if key == nonlocal_key and score >= WAKE_THRESHOLD:
                    print(f"🔔 Wake word detected! (key={key}, score={score:.3f})",
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
    if len(sys.argv) > 1 and sys.argv[1] == "--list-models":
        try:
            import numpy as np
            from openwakeword.model import Model as OWWModel
            import openwakeword as _oww
            from pathlib import Path as _Path

            pkg_dir = _Path(_oww.__file__).parent
            all_onnx = sorted(
                p for p in pkg_dir.rglob("*.onnx")
                if "embedding" not in p.name and "melspectrogram" not in p.name
                and "silero" not in p.name
            )
            print("Available built-in openwakeword models:")
            for p in all_onnx:
                print(f"  {p.name}  ({p})")

            # Load hey_jarvis and do one dummy prediction to show keys
            model_path = _find_builtin_model("hey_jarvis")
            m = OWWModel(wakeword_model_paths=[str(model_path)])
            dummy = np.zeros(1280, dtype=np.int16)
            scores = m.predict(dummy)
            print(f"\npredict() keys for {model_path.name}:")
            for k, v in scores.items():
                print(f"  {k!r}  (score={v:.4f})")
        except Exception as e:
            print(f"❌ {e}", file=sys.stderr)
            sys.exit(1)
    else:
        main()
