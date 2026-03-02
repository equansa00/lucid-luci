#!/usr/bin/env python3
"""
LUCI Streaming TTS — sentence-level streaming via Piper pipes.
Eliminates the 4-12 second delay of file-based TTS.

Architecture:
  LLM token stream → SentenceBuffer → synthesize_raw() → AudioPlayer queue → aplay
  First audio plays in ~400ms instead of 4-12 seconds.
"""
from __future__ import annotations

import os
import re
import sys
import queue
import threading
import subprocess
import time
from pathlib import Path
from typing import Iterator, Optional, Callable

# ── Paths ────────────────────────────────────────────────────────────────────
_WS = Path(__file__).parent

PIPER_BIN = str(_WS / "piper" / "piper" / "piper")

PIPER_VOICES: dict[str, str] = {
    "luci-male":    str(_WS / "piper" / "en_US-lessac-medium.onnx"),
    "luci-female":  str(_WS / "piper" / "en_US-amy-medium.onnx"),
    "luci-crisp":   str(_WS / "piper" / "en_US-ljspeech-high.onnx"),
    "luci-british": str(_WS / "piper" / "en_GB-alba-medium.onnx"),
}
DEFAULT_VOICE = os.getenv("LUCI_DEFAULT_VOICE", "luci-male")
VOICE_MODEL   = PIPER_VOICES.get(DEFAULT_VOICE, PIPER_VOICES["luci-male"])

SAMPLE_RATE = 22050
CHANNELS    = 1
FORMAT      = "S16_LE"

# ── Text cleaning ─────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    """Strip markdown/code/URLs — keep plain spoken text."""
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ── Sentence splitting ────────────────────────────────────────────────────────
# Split after .!? followed by whitespace + capital letter or closing quote
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z\"\'])')
_MIN_SENT   = 6
_MAX_SENT   = 400


class SentenceBuffer:
    """Accumulates LLM tokens and emits complete sentences."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, token: str) -> list[str]:
        """Add token, return any complete sentences ready to speak."""
        self._buf += token
        out: list[str] = []
        while True:
            m = _SENT_SPLIT.search(self._buf)
            if not m:
                break
            sent = self._buf[: m.start() + 1].strip()
            self._buf = self._buf[m.end():]
            if len(sent) >= _MIN_SENT:
                out.extend(_split_long(sent))
        return out

    def flush(self) -> list[str]:
        """Return any remaining buffered text."""
        rem = self._buf.strip()
        self._buf = ""
        if len(rem) >= _MIN_SENT:
            return _split_long(rem)
        return []


def _split_long(text: str) -> list[str]:
    """Break sentences longer than _MAX_SENT at comma/semicolon boundaries."""
    if len(text) <= _MAX_SENT:
        return [text]
    parts = re.split(r'(?<=[,;:])\s+', text)
    result: list[str] = []
    cur = ""
    for p in parts:
        if len(cur) + 1 + len(p) <= _MAX_SENT:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                result.append(cur)
            cur = p
    if cur:
        result.append(cur)
    return result or [text]


# ── Piper synthesis (in-memory pipe) ─────────────────────────────────────────
def synthesize_raw(text: str, model: str = "") -> bytes:
    """
    Synthesize text → raw S16_LE PCM bytes via Piper pipe mode.
    No temp files written. stdin → stdout, all in-memory.
    """
    model = model or VOICE_MODEL
    proc = subprocess.Popen(
        [PIPER_BIN, "--model", model, "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    raw, _ = proc.communicate(input=text.encode("utf-8"), timeout=20)
    if proc.returncode != 0:
        raise RuntimeError(f"Piper exited {proc.returncode} for text={text[:60]!r}")
    return raw


# ── Audio player (serialized aplay queue) ────────────────────────────────────
class AudioPlayer:
    """
    Background worker that plays PCM chunks serially through aplay.
    Synthesis threads enqueue chunks; the worker plays them in order.
    Fully interruptible — interrupt() kills current aplay and drains queue.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self.speaking = threading.Event()  # public — used by luci_wake echo gate
        threading.Thread(target=self._worker, daemon=True, name="luci-audio").start()

    def _worker(self) -> None:
        while True:
            chunk = self._q.get()
            if chunk is None:
                self._q.task_done()
                continue
            self.speaking.set()
            try:
                with self._proc_lock:
                    self._proc = subprocess.Popen(
                        ["aplay", "-r", str(SAMPLE_RATE), "-f", FORMAT,
                         "-c", str(CHANNELS), "-q"],
                        stdin=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                self._proc.communicate(input=chunk)
            except Exception as e:
                print(f"[tts] aplay error: {e}", file=sys.stderr)
            finally:
                with self._proc_lock:
                    self._proc = None
                if self._q.empty():
                    self.speaking.clear()
                self._q.task_done()

    def enqueue(self, raw: bytes) -> None:
        self._q.put(raw)

    def interrupt(self) -> None:
        """Stop current playback, drain pending queue."""
        # Drain first so worker doesn't pick up more
        while not self._q.empty():
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                break
        # Kill aplay
        with self._proc_lock:
            if self._proc is not None:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=1)
                except Exception:
                    pass
                self._proc = None
        self.speaking.clear()

    def wait(self, timeout: float = 60.0) -> None:
        """Block until all queued audio finishes."""
        self._q.join()

    @property
    def is_playing(self) -> bool:
        return self.speaking.is_set() or not self._q.empty()


# Module-level singleton
_player = AudioPlayer()


# ── Public API ────────────────────────────────────────────────────────────────
def speak(text: str, interrupt_first: bool = False) -> None:
    """
    Synthesize and enqueue text for playback. Non-blocking.
    Audio plays in background; call wait_done() to block until finished.
    """
    if interrupt_first:
        _player.interrupt()
    text = _clean(text)
    if not text:
        return
    try:
        raw = synthesize_raw(text)
        _player.enqueue(raw)
    except Exception as e:
        print(f"[tts] speak error: {e}", file=sys.stderr)


def speak_streaming(
    token_iter: Iterator[str],
    on_sentence: Optional[Callable[[str], None]] = None,
    interrupt_first: bool = False,
) -> str:
    """
    Speak while LLM is still generating. First audio plays in ~400ms.

    token_iter   — yields string tokens from a streaming LLM response
    on_sentence  — optional callback each time a sentence is dispatched
    returns      — full concatenated text

    Example:
        def llm_tokens():
            for chunk in requests.post(url, json={..., "stream": True}, stream=True):
                yield json.loads(chunk)["message"]["content"]

        full_text = speak_streaming(llm_tokens())
        wait_done()
    """
    if interrupt_first:
        _player.interrupt()

    buf = SentenceBuffer()
    full = ""
    threads: list[threading.Thread] = []

    def _synth_enqueue(sent: str) -> None:
        try:
            raw = synthesize_raw(sent)
            _player.enqueue(raw)
            if on_sentence:
                on_sentence(sent)
        except Exception as e:
            print(f"[tts] synth error: {e}", file=sys.stderr)

    for token in token_iter:
        full += token
        for sent in buf.feed(token):
            sent = _clean(sent)
            if sent:
                t = threading.Thread(target=_synth_enqueue, args=(sent,), daemon=True)
                t.start()
                threads.append(t)

    for sent in buf.flush():
        sent = _clean(sent)
        if sent:
            t = threading.Thread(target=_synth_enqueue, args=(sent,), daemon=True)
            t.start()
            threads.append(t)

    # Wait for all synthesis (not playback) to finish enqueueing
    for t in threads:
        t.join(timeout=30)

    return full


def interrupt() -> None:
    """Stop all current and pending speech immediately."""
    _player.interrupt()


def wait_done(timeout: float = 60.0) -> None:
    """Block until all queued audio has played."""
    _player.wait(timeout)


def is_speaking() -> bool:
    return _player.is_playing


# Public event for external echo-gate checks (used by luci_wake.py)
speaking_event: threading.Event = _player.speaking


def check_config() -> dict:
    return {
        "piper_bin":   PIPER_BIN,
        "voice_model": VOICE_MODEL,
        "piper_ok":    Path(PIPER_BIN).exists(),
        "model_ok":    Path(VOICE_MODEL).exists(),
        "sample_rate": SAMPLE_RATE,
        "voices":      list(PIPER_VOICES.keys()),
    }


# ── CLI / self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = check_config()
    print("TTS Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    if not cfg["piper_ok"] or not cfg["model_ok"]:
        print("ERROR: Piper not configured correctly", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1 and sys.argv[1] != "--demo":
        text = " ".join(sys.argv[1:])
        print(f"\nSpeaking: {text!r}")
        t0 = time.time()
        speak(text)
        wait_done()
        print(f"Done in {time.time() - t0:.3f}s")
    else:
        def _fake_llm() -> Iterator[str]:
            words = (
                "Hello Chip. I am LUCI, your personal AI agent. "
                "I am now speaking with streaming TTS. "
                "Each sentence plays as soon as it is ready, "
                "which means you hear me much faster than before."
            ).split()
            for w in words:
                yield w + " "
                time.sleep(0.04)

        print("\nStreaming demo (simulated LLM at 25 tokens/sec)...")
        t0 = time.time()
        full = speak_streaming(_fake_llm())
        wait_done()
        print(f"\nFull text: {full!r}")
        print(f"Total time: {time.time() - t0:.3f}s")
