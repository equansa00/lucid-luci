#!/usr/bin/env python3
"""
LUCI Web Interface — Phase 7B: Holographic Ambient Presence
Full-screen face + Three.js particle field + Web Audio waveform + PWA.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEB_HOST = os.getenv("LUCI_WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("LUCI_WEB_PORT", "7860"))
WEB_SECRET = os.getenv("LUCI_WEB_SECRET", "")

WORKSPACE      = Path.home() / "beast" / "workspace"
STATIC_DIR     = WORKSPACE / "static"
FACES_DIR      = STATIC_DIR / "faces"
WHISPER_SERVER = f"http://127.0.0.1:{os.getenv('LUCI_WHISPER_PORT', '8765')}"

# ---------------------------------------------------------------------------
# Imports from luci
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from luci import (  # noqa: E402
    MODEL,
    RUNS_DIR,
    MEMORY_PATH,
    PIPER_BIN,
    PIPER_MODEL,
    PIPER_VOICES,
    PIPER_VOICE_LABELS,
    DEFAULT_VOICE,
    LUCI_AUTO_MEMORY,
    auto_extract_memory,
    ollama_chat,
    summarize_large_output,
    route_model,
    format_model_tag,
    load_persona,
    load_persona_with_memory,
    tts_to_file,
    stt_transcribe,
    get_weather,
    get_time,
)

# ---------------------------------------------------------------------------
# LUCI v2 — sandbox, search, perceive, builder
# ---------------------------------------------------------------------------
try:
    from luci_sandbox import safe_path, SecurityError
    from luci_search import search_and_summarize, should_search
    from luci_perceive import perceive_file, perceive_url, summarize_perception
    from luci_builder import start_build, get_build_status
    _LUCI_V2 = True
except ImportError as _v2_err:
    _LUCI_V2 = False
    print(f"[luci_web] v2 modules not available: {_v2_err}", flush=True)

# Build trigger keywords
_BUILD_TRIGGERS = [
    "build me", "build a", "create a feature",
    "add feature", "implement a", "develop a",
    "give luci", "luci should", "add capability",
    "create a system", "write a system", "make a tool",
]

# ---------------------------------------------------------------------------
# Persistent Whisper server client
# ---------------------------------------------------------------------------
import urllib.request as _ureq
import urllib.error as _uerr


def stt_transcribe_fast(audio_path: Path) -> str:
    """POST audio to the persistent Whisper GPU server (~300ms).
    Falls back to direct in-process Whisper if the server is not up.
    """
    try:
        with open(audio_path, "rb") as fh:
            audio_data = fh.read()
        boundary = b"----LUCIWhisperBoundary"
        suffix = audio_path.suffix or ".webm"
        body = (
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="audio"; filename="audio' + suffix.encode() + b'"\r\n'
            b"Content-Type: application/octet-stream\r\n\r\n"
            + audio_data
            + b"\r\n--" + boundary + b"--\r\n"
        )
        req = _ureq.Request(
            f"{WHISPER_SERVER}/transcribe",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
            method="POST",
        )
        with _ureq.urlopen(req, timeout=30) as resp:
            import json as _json
            return _json.loads(resp.read()).get("text", "")
    except Exception as exc:
        print(f"[web] Whisper server unavailable ({exc}), falling back to direct", flush=True)
        return stt_transcribe(audio_path)


# ---------------------------------------------------------------------------
# Voice conversation history — SQLite
# ---------------------------------------------------------------------------
VOICE_DB = WORKSPACE / "luci_voice_history.db"


def _vhdb() -> sqlite3.Connection:
    """Return a thread-local DB connection (creates tables if new)."""
    conn = sqlite3.connect(str(VOICE_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vh_sessions (
            id         TEXT PRIMARY KEY,
            title      TEXT NOT NULL DEFAULT 'Voice session',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vh_messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES vh_sessions(id) ON DELETE CASCADE,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            audio_url  TEXT,
            ts         REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def vh_create_session() -> str:
    """Create a new voice session and return its ID."""
    sid = str(uuid.uuid4())
    now = time.time()
    with _vhdb() as conn:
        conn.execute(
            "INSERT INTO vh_sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (sid, "Voice session", now, now),
        )
    return sid


def vh_add_message(session_id: str, role: str, content: str, audio_url: str | None = None) -> None:
    """Append a message to a voice session."""
    now = time.time()
    with _vhdb() as conn:
        conn.execute(
            "INSERT INTO vh_messages (session_id, role, content, audio_url, ts) VALUES (?,?,?,?,?)",
            (session_id, role, content, audio_url, now),
        )
        conn.execute(
            "UPDATE vh_sessions SET updated_at=? WHERE id=?",
            (now, session_id),
        )


def vh_get_history(session_id: str, limit: int = 10) -> list[dict]:
    """Return last `limit` messages for the session (oldest first)."""
    conn = _vhdb()
    rows = conn.execute(
        """SELECT role, content FROM vh_messages
           WHERE session_id=?
           ORDER BY ts DESC LIMIT ?""",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def vh_set_title(session_id: str, title: str) -> None:
    with _vhdb() as conn:
        conn.execute("UPDATE vh_sessions SET title=? WHERE id=?", (title, session_id))


def vh_list_sessions(limit: int = 30) -> list[dict]:
    conn = _vhdb()
    rows = conn.execute(
        """SELECT s.id, s.title, s.created_at, s.updated_at,
                  COUNT(m.id) AS msg_count
           FROM vh_sessions s
           LEFT JOIN vh_messages m ON m.session_id = s.id
           GROUP BY s.id
           ORDER BY s.updated_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="LUCI")

STATIC_DIR.mkdir(parents=True, exist_ok=True)
FACES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/faces",  StaticFiles(directory=str(FACES_DIR)),  name="faces")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Connected WebSocket clients — for state broadcasting
# ---------------------------------------------------------------------------
_connected_ws: set[WebSocket] = set()


async def broadcast_state(state: str) -> None:
    msg = json.dumps({"type": "state", "state": state})
    for conn in list(_connected_ws):
        try:
            await conn.send_text(msg)
        except Exception:
            _connected_ws.discard(conn)


async def broadcast_transcript(text: str) -> None:
    msg = json.dumps({"type": "transcript", "text": text})
    for conn in list(_connected_ws):
        try:
            await conn.send_text(msg)
        except Exception:
            _connected_ws.discard(conn)


def _piper_tts_to_wav_bytes(text: str) -> bytes | None:
    """Run Piper TTS and return WAV bytes, or None on failure."""
    if not PIPER_BIN.exists() or not PIPER_MODEL.exists():
        return None
    try:
        # Clean text for TTS (strip markdown, cap length)
        import re as _re
        clean = _re.sub(r'```.*?```', '', text, flags=_re.DOTALL)
        clean = _re.sub(r'`[^`]+`', '', clean)
        clean = _re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', clean)
        clean = _re.sub(r'https?://\S+', '', clean)
        clean = _re.sub(r'\s+', ' ', clean).strip()[:3000]
        if not clean:
            return None

        proc = subprocess.run(
            [str(PIPER_BIN), "--model", str(PIPER_MODEL), "--output_raw"],
            input=clean.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # 16-bit PCM
            wf.setframerate(22050)
            wf.writeframes(proc.stdout)
        return buf.getvalue()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Service Worker
# ---------------------------------------------------------------------------
_SW_JS = """\
const CACHE = 'luci-v1';
const SHELL = ['/', '/manifest.json', '/faces/luci_neutral.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(clients.claim());
});

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  e.respondWith(
    fetch(e.request)
      .then(r => {
        if (r && r.status === 200) {
          const clone = r.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return r;
      })
      .catch(() => caches.match(e.request))
  );
});
"""

# ---------------------------------------------------------------------------
# Holographic HTML
# ---------------------------------------------------------------------------
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#D4AF37">
<title>LUCI</title>
<link rel="manifest" href="/manifest.json">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --gold:       #D4AF37;
    --gold-dim:   rgba(212,175,55,0.3);
    --gold-glow:  rgba(212,175,55,0.5);
    --bg:         #0a0705;
    --warm:       #F5E6C8;
    --surface:    rgba(12,9,6,0.96);
  }

  html, body {
    width: 100%; height: 100%;
    background: var(--bg);
    overflow: hidden;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
    user-select: none;
    touch-action: manipulation;
  }

  /* === Particle canvas — fullscreen background === */
  #particles {
    position: fixed; inset: 0;
    z-index: 0;
    pointer-events: none;
  }

  /* === Brand — top left === */
  #brand {
    position: fixed; top: 18px; left: 24px;
    z-index: 30; pointer-events: none;
  }
  #brand-name {
    font-size: 20px; font-weight: 700;
    letter-spacing: 0.16em; color: var(--gold);
    text-shadow: 0 0 24px rgba(212,175,55,0.55);
  }
  #brand-sub {
    font-size: 10px; letter-spacing: 0.12em;
    text-transform: uppercase; margin-top: 3px;
    color: rgba(212,175,55,0.45);
  }

  /* === Wake indicator — offset from mode-toggle === */
  #wake-indicator {
    position: fixed; top: 20px; right: 72px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.14em;
    color: var(--gold); opacity: 0;
    transition: opacity 0.35s;
    z-index: 30; pointer-events: none;
  }
  #wake-indicator.active {
    opacity: 1;
    animation: wakeGlow 1.2s ease-in-out infinite;
  }
  @keyframes wakeGlow {
    0%,100% { opacity: 0.65; text-shadow: 0 0 8px var(--gold); }
    50%      { opacity: 1;    text-shadow: 0 0 18px var(--gold), 0 0 36px var(--gold); }
  }

  /* === Main UI column (centered) === */
  #ui {
    position: fixed; inset: 0;
    z-index: 10;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 0;
    pointer-events: none;
  }

  /* === Face container === */
  #face-container {
    position: relative;
    width: 380px; height: 380px;
    border-radius: 50%;
    overflow: hidden;
    cursor: pointer;
    pointer-events: all;
    transition: box-shadow 0.55s ease;
    animation: ringIdle 4s ease-in-out infinite;
  }

  /* Ring keyframes */
  @keyframes ringIdle {
    0%,100% {
      box-shadow: 0 0 30px 8px rgba(212,175,55,0.28),
                  0 0 65px 22px rgba(212,175,55,0.08);
      transform: scale(1.000);
    }
    50% {
      box-shadow: 0 0 48px 14px rgba(212,175,55,0.50),
                  0 0 95px 38px rgba(212,175,55,0.14);
      transform: scale(1.013);
    }
  }

  @keyframes ringListening {
    0%,100% {
      box-shadow: 0 0 50px 16px rgba(255,255,255,0.55),
                  0 0 95px 38px rgba(255,255,255,0.18);
      transform: scale(1.000);
    }
    50% {
      box-shadow: 0 0 70px 24px rgba(255,255,255,0.80),
                  0 0 120px 50px rgba(255,255,255,0.26);
      transform: scale(1.010);
    }
  }

  @keyframes ringThinking {
    0%,100% {
      box-shadow: 0 0 42px 12px rgba(212,175,55,0.55),
                  0 0 75px 28px rgba(212,175,55,0.18);
      transform: scale(1.000);
    }
    50% {
      box-shadow: 0 0 68px 24px rgba(212,175,55,0.90),
                  0 0 110px 46px rgba(212,175,55,0.30);
      transform: scale(1.010);
    }
  }

  @keyframes ringSpeaking {
    0%,100% {
      box-shadow: 0 0 44px 12px rgba(212,175,55,0.65),
                  0 0 85px 32px rgba(212,175,55,0.22);
      transform: scale(1.000);
    }
    50% {
      box-shadow: 0 0 58px 18px rgba(212,175,55,0.85),
                  0 0 105px 44px rgba(212,175,55,0.28);
      transform: scale(1.008);
    }
  }

  /* State classes */
  #face-container.state-idle      { animation: ringIdle      4.0s ease-in-out infinite; }
  #face-container.state-listening { animation: ringListening  1.6s ease-in-out infinite; }
  #face-container.state-thinking  { animation: ringThinking   1.1s ease-in-out infinite; }
  #face-container.state-speaking  { animation: ringSpeaking   3.0s ease-in-out infinite; }

  /* Hologram inner wrapper */
  .face-hologram {
    position: relative;
    width: 100%; height: 100%;
  }

  /* Scanlines overlay */
  .face-hologram::before {
    content: '';
    position: absolute; inset: 0; z-index: 3;
    background: repeating-linear-gradient(
      0deg,
      transparent 0px, transparent 2px,
      rgba(212,175,55,0.038) 2px, rgba(212,175,55,0.038) 4px
    );
    pointer-events: none;
  }

  /* Radial vignette — blends amber PNG edges into #0a0705 */
  .face-hologram::after {
    content: '';
    position: absolute; inset: 0; z-index: 2;
    background: radial-gradient(
      ellipse 70% 70% at 50% 47%,
      transparent 42%,
      rgba(10,7,5,0.55) 68%,
      rgba(10,7,5,0.96) 100%
    );
    pointer-events: none;
  }

  #face-img {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    filter: contrast(1.08) brightness(0.94) sepia(0.14);
    transition: opacity 0.42s ease;
    z-index: 1;
  }

  /* Outer ripple ring (decorative) */
  #face-ripple {
    position: absolute;
    inset: -10px; border-radius: 50%;
    border: 1px solid rgba(212,175,55,0.18);
    animation: ripple 3.5s ease-in-out infinite;
    pointer-events: none; z-index: -1;
  }
  #face-ripple-2 {
    position: absolute;
    inset: -22px; border-radius: 50%;
    border: 1px solid rgba(212,175,55,0.08);
    animation: ripple 3.5s ease-in-out infinite 1.2s;
    pointer-events: none; z-index: -1;
  }
  @keyframes ripple {
    0%,100% { transform: scale(1);    opacity: 0.5; }
    50%      { transform: scale(1.06); opacity: 1;   }
  }

  /* === Waveform === */
  #waveform {
    display: block;
    width: min(560px, 90vw);
    height: 56px;
    margin-top: 22px;
    pointer-events: none;
    opacity: 0.85;
  }

  /* === Subtitle === */
  #subtitle {
    position: fixed; bottom: 12%; left: 50%;
    transform: translateX(-50%);
    width: min(340px, 88vw);
    text-align: center;
    font-size: 16px; line-height: 1.6;
    color: var(--warm);
    text-shadow: 0 0 18px rgba(245,230,200,0.3);
    opacity: 0;
    transition: opacity 0.5s ease;
    z-index: 20; pointer-events: none;
    padding: 0 16px;
    /* Clamp to 3 visible lines */
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  #subtitle.visible  { opacity: 1; }
  #subtitle.response { color: var(--gold); text-shadow: 0 0 20px rgba(212,175,55,0.45); }

  /* === Audio unlock nudge === */
  #audio-unlock {
    position: fixed; bottom: 8%; left: 50%;
    transform: translateX(-50%);
    font-size: 11px; letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(212,175,55,0.5);
    pointer-events: none; z-index: 25;
    transition: opacity 0.6s ease;
    animation: unlockPulse 2s ease-in-out infinite;
  }
  #audio-unlock.hidden { opacity: 0; }
  @keyframes unlockPulse {
    0%,100% { opacity: 0.45; } 50% { opacity: 0.85; }
  }

  /* === Tap hint === */
  #tap-hint {
    position: fixed; bottom: 6%; left: 50%;
    transform: translateX(-50%);
    font-size: 11px; letter-spacing: 0.18em;
    text-transform: uppercase; color: rgba(212,175,55,0.38);
    pointer-events: none; z-index: 20;
    transition: opacity 0.5s;
  }

  /* === Debug overlay === */
  #debug-overlay {
    position: fixed;
    bottom: 2%;
    left: 12px;
    z-index: 50;
    pointer-events: none;
    font-family: monospace;
    font-size: 10px;
    color: rgba(212,175,55,0.65);
    line-height: 1.7;
    max-width: 280px;
  }

  /* === Always-listen toggle === */
  #always-listen-toggle {
    position: fixed; bottom: 10%; left: 50%;
    transform: translateX(-50%);
    z-index: 30;
    display: flex; gap: 8px; align-items: center;
  }
  #al-btn {
    background: rgba(10,7,5,0.8);
    border: 1px solid rgba(212,175,55,0.3);
    color: rgba(212,175,55,0.6);
    border-radius: 20px; padding: 8px 18px;
    font-size: 12px; letter-spacing: 0.1em;
    cursor: pointer; backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    transition: all 0.2s;
  }
  #al-btn.active {
    border-color: rgba(212,175,55,0.8);
    color: #D4AF37;
    box-shadow: 0 0 16px rgba(212,175,55,0.2);
  }
  #al-btn.interrupted {
    border-color: rgba(220,38,38,0.6);
    color: #ef4444;
  }
  #new-voice-session-btn {
    background: rgba(10,7,5,0.7);
    border: 1px solid rgba(212,175,55,0.2);
    color: rgba(212,175,55,0.4);
    border-radius: 20px; padding: 6px 14px;
    font-size: 11px; letter-spacing: 0.08em;
    cursor: pointer; transition: all 0.2s;
  }
  #new-voice-session-btn:hover {
    border-color: rgba(212,175,55,0.5);
    color: rgba(212,175,55,0.8);
  }

  /* === Voice settings gear button === */
  #voice-settings-btn {
    position: fixed; top: 16px; right: 68px;
    z-index: 200;
    background: rgba(12,9,6,0.7);
    border: 1px solid rgba(212,175,55,0.35);
    color: #D4AF37; font-size: 16px;
    width: 40px; height: 40px;
    border-radius: 50%; cursor: pointer;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    transition: all 0.2s;
  }
  #voice-settings-btn:hover { background: rgba(212,175,55,0.15); }

  /* === Voice settings panel === */
  #voice-settings-panel {
    position: fixed; top: 68px; right: 20px;
    width: 260px; z-index: 150;
    background: rgba(10,7,5,0.95);
    border: 1px solid rgba(212,175,55,0.28);
    border-radius: 16px; padding: 0;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.6);
    display: none; overflow: hidden;
  }
  #voice-settings-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 16px 18px 12px;
    border-bottom: 1px solid rgba(212,175,55,0.12);
    font-size: 11px; letter-spacing: 0.14em;
    text-transform: uppercase; color: rgba(212,175,55,0.55);
  }
  #voice-settings-close {
    background: none; border: none; color: rgba(212,175,55,0.5);
    cursor: pointer; font-size: 14px; padding: 0;
  }
  #voice-list { padding: 10px; display: flex; flex-direction: column; gap: 6px; }
  .voice-option {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px; border-radius: 10px;
    border: 1px solid transparent; cursor: pointer;
    transition: all 0.15s;
  }
  .voice-option:hover { background: rgba(212,175,55,0.08); }
  .voice-option.selected {
    background: rgba(212,175,55,0.12);
    border-color: rgba(212,175,55,0.3);
  }
  .voice-option-name { flex: 1; font-size: 13px; color: rgba(245,230,200,0.8); }
  .voice-option.selected .voice-option-name { color: #F5E6C8; }
  .voice-preview-btn {
    background: none; border: 1px solid rgba(212,175,55,0.25);
    color: #D4AF37; border-radius: 6px; padding: 4px 8px;
    font-size: 11px; cursor: pointer; transition: all 0.15s; flex-shrink: 0;
  }
  .voice-preview-btn:hover { background: rgba(212,175,55,0.15); }
  .voice-preview-btn.playing { color: #ef4444; border-color: rgba(220,38,38,0.4); }
  #voice-speed-row {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px 16px;
    border-top: 1px solid rgba(212,175,55,0.1);
    font-size: 12px; color: rgba(212,175,55,0.5);
  }
  #voice-speed { flex: 1; accent-color: #D4AF37; }
  #voice-speed-val { font-size: 11px; color: rgba(212,175,55,0.6); width: 28px; }

  /* === Mode toggle button === */
  #mode-toggle {
    position: fixed; top: 16px; right: 20px;
    z-index: 200;
    background: rgba(12,9,6,0.7);
    border: 1px solid rgba(212,175,55,0.35);
    color: #D4AF37;
    font-size: 18px;
    width: 40px; height: 40px;
    border-radius: 50%;
    cursor: pointer;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    transition: all 0.2s;
  }
  #mode-toggle:hover {
    background: rgba(212,175,55,0.15);
    border-color: rgba(212,175,55,0.7);
  }

  /* ============================================================
     CHAT MODE
  ============================================================ */
  #chat-mode {
    position: fixed; inset: 0;
    display: flex; flex-direction: row;
    background: #0a0705;
    font-family: system-ui, -apple-system, sans-serif;
  }

  /* History sidebar */
  #history-panel {
    width: 220px; flex-shrink: 0;
    background: rgba(10,7,5,0.82);
    border-right: 1px solid rgba(212,175,55,0.15);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    display: flex; flex-direction: column;
    overflow: hidden; z-index: 10;
  }
  #history-header {
    padding: 20px 16px 12px;
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid rgba(212,175,55,0.12);
    flex-shrink: 0;
  }
  #history-header span {
    font-size: 11px; letter-spacing: 0.14em;
    text-transform: uppercase; color: rgba(212,175,55,0.55);
  }
  #new-chat-btn {
    background: none; border: 1px solid rgba(212,175,55,0.3);
    color: #D4AF37; width: 26px; height: 26px;
    border-radius: 6px; cursor: pointer; font-size: 16px;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.15s;
  }
  #new-chat-btn:hover { background: rgba(212,175,55,0.12); }
  #history-list { flex: 1; overflow-y: auto; padding: 8px; }
  #history-list::-webkit-scrollbar { width: 3px; }
  #history-list::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.2); border-radius: 2px; }
  .history-group-label {
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    color: rgba(212,175,55,0.3); padding: 10px 8px 4px;
  }
  .history-item {
    padding: 9px 10px; border-radius: 8px; cursor: pointer;
    font-size: 13px; color: rgba(245,230,200,0.65);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    transition: all 0.15s; margin-bottom: 2px;
    border: 1px solid transparent;
  }
  .history-item:hover { background: rgba(212,175,55,0.08); color: rgba(245,230,200,0.9); }
  .history-item.active {
    background: rgba(212,175,55,0.12);
    border-color: rgba(212,175,55,0.25);
    color: #F5E6C8;
  }
  .history-item { position: relative; padding-right: 28px; }
  .history-item .del-btn {
    position: absolute; right: 6px; top: 50%; transform: translateY(-50%);
    background: none; border: none; color: rgba(212,175,55,0.0);
    font-size: 14px; cursor: pointer; padding: 2px 4px; line-height: 1;
    transition: color 0.15s; border-radius: 3px;
  }
  .history-item:hover .del-btn { color: rgba(212,175,55,0.45); }
  .history-item .del-btn:hover { color: #ff4455 !important; background: rgba(255,68,85,0.12); }

  /* Friction slide-over panel */
  #friction-overlay {
    position: fixed; inset: 0; z-index: 200;
    background: rgba(10,7,5,0.75); backdrop-filter: blur(6px);
    display: none; align-items: center; justify-content: center;
  }
  #friction-overlay.open { display: flex; }
  #friction-box {
    background: #0d0f14; border: 1px solid #00e5ff;
    border-radius: 4px; padding: 28px; width: min(600px, 92vw);
    max-height: 88vh; overflow-y: auto;
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
  }
  #friction-box h2 {
    font-size: 13px; letter-spacing: 3px; color: #00e5ff;
    text-transform: uppercase; margin-bottom: 20px;
  }
  .fi-label {
    font-size: 10px; letter-spacing: 2px; color: #4a5168;
    text-transform: uppercase; margin-bottom: 6px;
  }
  .fi-input {
    width: 100%; background: #1a1e28; border: 1px solid #2e3447;
    color: #c8cedd; font-family: inherit; font-size: 12px;
    padding: 10px 12px; border-radius: 2px; outline: none;
    transition: border-color 0.15s; box-sizing: border-box; resize: vertical;
  }
  .fi-input:focus { border-color: #00e5ff; }
  #fi-unblock-btn {
    width: 100%; padding: 12px; margin-top: 4px;
    background: rgba(0,229,255,0.1); border: 1px solid #00e5ff;
    color: #00e5ff; font-family: inherit; font-size: 13px;
    font-weight: 700; letter-spacing: 3px; cursor: pointer;
    border-radius: 2px; transition: all 0.15s; text-transform: uppercase;
  }
  #fi-unblock-btn:hover:not(:disabled) { background: #00e5ff; color: #06070a; }
  #fi-unblock-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  #fi-response {
    display: none; margin-top: 16px;
    background: #06070a; border: 1px solid #2e3447;
    border-left: 3px solid #00e5ff; border-radius: 2px;
    padding: 16px; font-size: 12px; line-height: 1.9;
    white-space: pre-wrap; color: #c8cedd;
  }
  .fi-section-label { color: #00e5ff; font-weight: 700; }
  #fi-close-btn {
    float: right; background: none; border: 1px solid #2e3447;
    color: #4a5168; font-family: inherit; font-size: 11px;
    padding: 4px 10px; cursor: pointer; border-radius: 2px;
    transition: all 0.15s;
  }
  #fi-close-btn:hover { border-color: #ff4455; color: #ff4455; }

  /* Memory maintenance panel */
  #memory-overlay {
    position: fixed; inset: 0; z-index: 200;
    background: rgba(10,7,5,0.75); backdrop-filter: blur(6px);
    display: none; align-items: center; justify-content: center;
  }
  #memory-overlay.open { display: flex; }
  #memory-box {
    background: #0d0f14; border: 1px solid rgba(212,175,55,0.5);
    border-radius: 4px; padding: 28px; width: min(640px, 92vw);
    max-height: 88vh; overflow-y: auto;
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
  }
  #memory-box h2 {
    font-size: 13px; letter-spacing: 3px; color: #D4AF37;
    text-transform: uppercase; margin-bottom: 6px;
  }
  #memory-box .mem-subtitle {
    font-size: 10px; color: #4a5168; margin-bottom: 20px; line-height: 1.6;
  }
  .mem-entry {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 12px; background: #1a1e28;
    border: 1px solid #2e3447; border-radius: 2px; margin-bottom: 6px;
  }
  .mem-entry-text { flex: 1; font-size: 11px; color: #c8cedd; line-height: 1.6; }
  .mem-entry-key { font-size: 10px; color: #D4AF37; margin-bottom: 2px; letter-spacing: 1px; }
  .mem-del-btn {
    background: none; border: 1px solid transparent; color: #4a5168;
    font-size: 13px; cursor: pointer; padding: 2px 6px; border-radius: 2px;
    transition: all 0.15s; flex-shrink: 0; margin-top: 1px;
  }
  .mem-del-btn:hover { border-color: #ff4455; color: #ff4455; background: rgba(255,68,85,0.1); }
  #mem-close-btn {
    float: right; background: none; border: 1px solid #2e3447;
    color: #4a5168; font-family: inherit; font-size: 11px;
    padding: 4px 10px; cursor: pointer; border-radius: 2px; transition: all 0.15s;
  }
  #mem-close-btn:hover { border-color: #ff4455; color: #ff4455; }
  #mem-search {
    width: 100%; background: #1a1e28; border: 1px solid #2e3447;
    color: #c8cedd; font-family: inherit; font-size: 12px;
    padding: 8px 12px; border-radius: 2px; outline: none;
    box-sizing: border-box; margin-bottom: 14px;
    transition: border-color 0.15s;
  }
  #mem-search:focus { border-color: #D4AF37; }

  /* Chat panel */
  #chat-panel {
    flex: 1; min-width: 0;
    display: flex; flex-direction: column;
    position: relative;
    border-right: 1px solid rgba(212,175,55,0.1);
  }
  #messages-container {
    flex: 1; overflow-y: auto; padding: 24px 20px 12px;
    scroll-behavior: smooth;
  }
  #messages-container::-webkit-scrollbar { width: 3px; }
  #messages-container::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.15); border-radius: 2px; }

  /* Message bubbles */
  .msg-row { display: flex; margin-bottom: 16px; gap: 10px; }
  .msg-row.user { justify-content: flex-end; }
  .msg-row.luci { justify-content: flex-start; }
  .msg-bubble {
    max-width: 72%; padding: 11px 15px;
    border-radius: 16px; line-height: 1.6;
    font-size: 14px; white-space: pre-wrap; word-break: break-word;
    user-select: text; -webkit-user-select: text; cursor: text;
  }
  .msg-row.user .msg-bubble {
    background: rgba(212,175,55,0.14);
    border: 1px solid rgba(212,175,55,0.25);
    color: #F5E6C8; border-bottom-right-radius: 4px;
  }
  .msg-row.luci .msg-bubble {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    color: rgba(245,230,200,0.88); border-bottom-left-radius: 4px;
  }
  .mem-tag {
    display: inline-block; margin-top: 6px;
    font-size: 10px; letter-spacing: 0.08em;
    color: rgba(212,175,55,0.7);
    border: 1px solid rgba(212,175,55,0.25);
    border-radius: 4px; padding: 2px 6px;
    cursor: pointer; transition: all 0.15s;
  }
  .mem-tag:hover { background: rgba(212,175,55,0.12); color: #D4AF37; }
  .typing-dots span {
    display: inline-block; width: 5px; height: 5px;
    border-radius: 50%; background: rgba(212,175,55,0.5);
    margin: 0 2px; animation: typingDot 1.2s infinite;
  }
  .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes typingDot {
    0%,60%,100% { transform: translateY(0); opacity: 0.4; }
    30%          { transform: translateY(-5px); opacity: 1; }
  }
  .model-tag { font-size: 10px; color: rgba(212,175,55,0.4); margin-top: 4px; display: block; }

  /* Input bar */
  #input-bar {
    padding: 12px 16px;
    background: rgba(10,7,5,0.9);
    border-top: 1px solid rgba(212,175,55,0.12);
    display: flex; gap: 8px; align-items: flex-end;
    flex-shrink: 0;
  }
  #chat-input {
    flex: 1; resize: none; min-height: 40px; max-height: 120px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 10px; color: #F5E6C8;
    padding: 10px 14px; font-size: 14px; outline: none;
    font-family: inherit; line-height: 1.4;
    transition: border-color 0.2s; user-select: text;
  }
  #chat-input:focus { border-color: rgba(212,175,55,0.5); }
  #chat-input::placeholder { color: rgba(245,230,200,0.25); }
  #voice-btn, #send-btn {
    height: 40px; border-radius: 10px; border: none;
    cursor: pointer; font-size: 15px; flex-shrink: 0;
    transition: all 0.15s;
  }
  #voice-btn {
    width: 40px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(212,175,55,0.2);
    color: #D4AF37;
  }
  #voice-btn:hover { background: rgba(212,175,55,0.12); }
  #voice-btn.recording {
    background: rgba(220,38,38,0.2);
    border-color: rgba(220,38,38,0.5);
    color: #ef4444;
    animation: recPulse 1s infinite;
  }
  @keyframes recPulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.3); }
    50%     { box-shadow: 0 0 0 6px rgba(220,38,38,0); }
  }
  #send-btn { width: 40px; background: rgba(212,175,55,0.85); color: #0a0705; font-weight: 700; }
  #send-btn:hover { background: #D4AF37; }

  /* Face panel */
  #face-panel {
    width: 320px; flex-shrink: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    position: relative; overflow: hidden; gap: 12px;
  }
  #chat-particles { position: absolute; inset: 0; pointer-events: none; z-index: 0; }
  #chat-face-container {
    position: relative; width: 240px; height: 240px;
    border-radius: 50%; overflow: hidden; z-index: 1;
  }
  #chat-face-container.state-idle      { animation: ringIdle      4.0s ease-in-out infinite; }
  #chat-face-container.state-listening { animation: ringListening  1.6s ease-in-out infinite; }
  #chat-face-container.state-thinking  { animation: ringThinking   1.1s ease-in-out infinite; }
  #chat-face-container.state-speaking  { animation: ringSpeaking   3.0s ease-in-out infinite; }
  #chat-face-img {
    width: 100%; height: 100%; object-fit: cover;
    filter: contrast(1.08) brightness(0.94) sepia(0.14);
    transition: opacity 0.42s;
  }
  .face-ripple {
    position: absolute; inset: -10px; border-radius: 50%;
    border: 1px solid rgba(212,175,55,0.18);
    animation: ripple 3.5s ease-in-out infinite; pointer-events: none;
  }
  .face-ripple-2 {
    position: absolute; inset: -22px; border-radius: 50%;
    border: 1px solid rgba(212,175,55,0.08);
    animation: ripple 3.5s ease-in-out infinite 1.2s; pointer-events: none;
  }
  #chat-waveform { width: 220px; height: 40px; z-index: 1; pointer-events: none; }
  #chat-status-text {
    font-size: 11px; letter-spacing: 0.12em;
    text-transform: uppercase; color: rgba(212,175,55,0.4); z-index: 1;
  }

  /* === Mobile === */
  @media (max-width: 480px) {
    #face-container { width: 286px; height: 286px; }
    #subtitle { font-size: 15px; }
  }
  @media (max-height: 600px) {
    #face-container { width: 240px; height: 240px; }
    #waveform { height: 40px; margin-top: 14px; }
  }
  @media (max-width: 768px) {
    #chat-mode { flex-direction: column; }
    #history-panel { width: 100%; height: 48px; flex-direction: row; border-right: none; border-bottom: 1px solid rgba(212,175,55,0.15); overflow-x: auto; }
    #history-list { display: flex; flex-direction: row; padding: 4px; overflow-x: auto; overflow-y: hidden; }
    .history-item { white-space: nowrap; flex-shrink: 0; }
    .history-group-label { display: none; }
    #history-header { padding: 10px 12px; }
    #face-panel { width: 100%; height: 200px; flex-direction: row; justify-content: center; gap: 20px; }
    #chat-face-container { width: 140px; height: 140px; }
    #chat-particles { display: none; }
    #chat-panel { flex: 1; min-height: 0; }
  }
</style>
</head>
<body>

<!-- Mode toggle — always visible in both modes -->
<button id="mode-toggle" title="Switch mode">&#x1F4AC;</button>
<!-- Voice settings gear — always visible -->
<button id="voice-settings-btn" title="Voice settings">&#x2699;</button>

<!-- Voice settings panel -->
<div id="voice-settings-panel">
  <div id="voice-settings-header">
    <span>Voice Settings</span>
    <button id="voice-settings-close">&#x2715;</button>
  </div>
  <div id="voice-list"></div>
  <div id="voice-speed-row">
    <span>Speed</span>
    <input type="range" id="voice-speed" min="0.7" max="1.5" step="0.1" value="1.0">
    <span id="voice-speed-val">1.0&times;</span>
  </div>
</div>

<!-- ============================================================
     AMBIENT MODE
============================================================ -->
<div id="ambient-mode">

<!-- Background particle field -->
<canvas id="particles"></canvas>

<!-- Brand -->
<div id="brand">
  <div id="brand-name">LUCI</div>
  <div id="brand-sub">Personal AI Agent</div>
</div>

<!-- Wake indicator -->
<div id="wake-indicator">&#9679; LISTENING</div>

<!-- Centered UI column -->
<div id="ui">
  <div id="face-container" class="state-idle">
    <div class="face-hologram">
      <img id="face-img" src="/faces/luci_neutral.png" alt="LUCI">
    </div>
    <div id="face-ripple"></div>
    <div id="face-ripple-2"></div>
  </div>
  <canvas id="waveform"></canvas>
</div>

<!-- Subtitle overlay -->
<div id="subtitle"></div>
<div id="audio-unlock">tap once to enable voice</div>
<div id="tap-hint">tap or speak to interact</div>

<!-- Always-listen toggle -->
<div id="always-listen-toggle">
  <button id="al-btn">&#x1F442; Always Listen</button>
  <button id="new-voice-session-btn" title="Start new voice conversation" onclick="newVoiceSession();this.textContent='\u2713 New session'">&#x1F504; New Session</button>
</div>

<!-- Debug overlay -->
<div id="debug-overlay">
  <div id="debug-lines"></div>
</div>

</div><!-- end #ambient-mode -->

<!-- ============================================================
     CHAT MODE
============================================================ -->
<div id="chat-mode" style="display:none">

  <!-- Left: history sidebar -->
  <div id="history-panel">
    <div id="history-header">
      <span>Conversations</span>
      <div style="display:flex;gap:5px;align-items:center;">
        <button id="memory-btn" title="Memory maintenance"
          onclick="openMemory()"
          style="background:none;border:1px solid rgba(212,175,55,0.3);color:#D4AF37;
                 width:26px;height:26px;border-radius:6px;cursor:pointer;font-size:13px;
                 display:flex;align-items:center;justify-content:center;transition:all 0.15s;">
          🧠
        </button>
        <button id="friction-btn" title="Friction Interceptor — get unblocked"
          onclick="openFriction()"
          style="background:none;border:1px solid rgba(0,229,255,0.3);color:#00e5ff;
                 width:26px;height:26px;border-radius:6px;cursor:pointer;font-size:13px;
                 display:flex;align-items:center;justify-content:center;transition:all 0.15s;">
          ⚡
        </button>
        <button id="new-chat-btn" title="New conversation">&#xFF0B;</button>
      </div>
    </div>
    <div id="history-list"></div>
  </div>

  <!-- Center: chat panel -->
  <div id="chat-panel">
    <div id="messages-container">
      <div id="messages"></div>
    </div>
    <div id="input-bar">
      <textarea id="chat-input" placeholder="Message LUCI..." rows="1"></textarea>
      <button id="voice-btn" title="Voice input">&#x1F3A4;</button>
      <button id="send-btn">&#x27A4;</button>
    </div>
  </div>

  <!-- Right: LUCI face panel -->
  <div id="face-panel">
    <canvas id="chat-particles"></canvas>
    <div id="chat-face-container" class="state-idle">
      <div class="face-hologram">
        <img id="chat-face-img" src="/faces/luci_neutral.png" alt="LUCI">
      </div>
      <div class="face-ripple"></div>
      <div class="face-ripple-2"></div>
    </div>
    <canvas id="chat-waveform"></canvas>
    <div id="chat-status-text">Ready</div>
  </div>

</div><!-- end #chat-mode -->

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
'use strict';

// =========================================================================
// Ambient state
// =========================================================================
let currentState  = 'idle';
let ws            = null;
let wsReady       = false;

let micStream     = null;
let analyser      = null;
let mediaRecorder = null;
let audioChunks   = [];
let isRecording   = false;
let silenceTimer  = null;
let subtitleTimer = null;
let currentAudio  = null;
let dpr           = window.devicePixelRatio || 1;
let _lastLevelLog = 0;
let _audioUnlocked = false;
let _audioCtx      = null;

// Voice session tracking — persists across page loads, resets on "new session"
let voiceSessionId = localStorage.getItem('luci-voice-session') || '';
function newVoiceSession() {
  voiceSessionId = '';
  localStorage.removeItem('luci-voice-session');
}
function _saveVoiceSession(id) {
  voiceSessionId = id;
  localStorage.setItem('luci-voice-session', id);
}

// DOM refs
const faceImg       = document.getElementById('face-img');
const faceContainer = document.getElementById('face-container');
const waveCanvas    = document.getElementById('waveform');
const waveCtx       = waveCanvas.getContext('2d');
const subtitleEl    = document.getElementById('subtitle');
const wakeIndicator = document.getElementById('wake-indicator');
const tapHint       = document.getElementById('tap-hint');
const audioUnlockEl = document.getElementById('audio-unlock');

const FACES = {
  idle:      '/faces/luci_neutral.png',
  thinking:  '/faces/luci_neutral.png',
  listening: '/faces/luci_listening.png',
  speaking:  '/faces/luci_speaking.png',
};

// =========================================================================
// =========================================================================
// Debug overlay
// =========================================================================
const debugLines = document.getElementById('debug-lines');
const MAX_DEBUG  = 8;
let debugHistory = [];

function dbg(msg) {
  const ts = new Date().toLocaleTimeString('en', {
    hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
  });
  debugHistory.push(ts + ' ' + msg);
  if (debugHistory.length > MAX_DEBUG) debugHistory.shift();
  debugLines.innerHTML = debugHistory.map(l => '<div>' + l + '</div>').join('');
}

// =========================================================================
// State machine
// =========================================================================
function setState(next) {
  if (currentState === next) return;
  currentState = next;
  dbg('\u2192 state: ' + next);

  // Crossfade face image
  const target  = FACES[next] || FACES.idle;
  const current = faceImg.getAttribute('src') || '';
  if (!current.endsWith(target.split('/').pop())) {
    faceImg.style.opacity = '0';
    setTimeout(() => { faceImg.src = target; faceImg.style.opacity = '1'; }, 230);
  }

  // Ring animation class
  faceContainer.className = 'state-' + next;

  // Wake indicator
  wakeIndicator.className = next === 'listening' ? 'active' : '';

  // Tap hint — only visible when idle
  tapHint.style.opacity = next === 'idle' ? '1' : '0';
}

// =========================================================================
// Subtitle
// =========================================================================
function showSubtitle(text, isResponse) {
  subtitleEl.textContent = text;
  subtitleEl.className   = 'visible' + (isResponse ? ' response' : '');
  if (subtitleTimer) clearTimeout(subtitleTimer);
  subtitleTimer = setTimeout(() => {
    subtitleEl.classList.remove('visible', 'response');
  }, 4600);
}

// =========================================================================
// WebSocket
// =========================================================================
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(proto + '://' + location.host + '/ws');

  ws.onopen = () => {};

  ws.onmessage = (ev) => {
    let d; try { d = JSON.parse(ev.data); } catch { return; }
    dbg('ws: ' + d.type);

    if (d.type === 'status') {
      wsReady = true;
      showSubtitle('LUCI online', true);
      setTimeout(() => subtitleEl.classList.remove('visible', 'response'), 2800);

    } else if (d.type === 'state') {
      setState(d.state);

    } else if (d.type === 'transcript') {
      showSubtitle(d.text, false);

    } else if (d.type === 'response') {
      showSubtitle(d.text, true);
      if (currentState !== 'speaking') setState('speaking');

    } else if (d.type === 'audio') {
      dbg('\uD83D\uDD0A audio received (' + Math.round(d.data.length * 0.75 / 1024) + 'kb)');
      playWavBase64(d.data, () => {
        currentAudio = null;
        setState('idle');
        dbg('\uD83D\uDD0A playback finished');
      });
    }
  };

  ws.onerror = () => {};
  ws.onclose = () => { wsReady = false; setTimeout(connectWS, 3000); };
}

// =========================================================================
// Audio unlock (browsers require user gesture before playing audio)
// =========================================================================
function _unlockAudio() {
  if (_audioUnlocked) return;
  _audioUnlocked = true;
  _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  _audioCtx.resume();
  audioUnlockEl.classList.add('hidden');
  dbg('\uD83D\uDD0A audio unlocked');
}
// Unlock on first any-click (covers tap-to-speak too)
document.addEventListener('click', _unlockAudio, {once: false});

// =========================================================================
// Play WAV from base64
// =========================================================================
function playWavBase64(b64, onEnded) {
  try {
    const bytes    = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const blob     = new Blob([bytes], {type: 'audio/wav'});
    const url      = URL.createObjectURL(blob);
    currentAudio   = new Audio(url);
    currentAudio.oncanplaythrough = () => {
      currentAudio.play().catch(e => {
        dbg('\u274c play blocked: ' + e.message);
        URL.revokeObjectURL(url);
        if (onEnded) onEnded();
      });
    };
    currentAudio.onended = () => {
      URL.revokeObjectURL(url);
      currentAudio = null;
      if (onEnded) onEnded();
    };
    currentAudio.onerror = () => {
      URL.revokeObjectURL(url);
      currentAudio = null;
      if (onEnded) onEnded();
    };
  } catch (e) {
    dbg('\u274c playWavBase64: ' + e.message);
    if (onEnded) onEnded();
  }
}

// Connect immediately — tunnel handles security
connectWS();

// =========================================================================
// Microphone init
// =========================================================================
async function initMic() {
  if (micStream) return true;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({audio: true, video: false});
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaStreamSource(micStream);
    analyser  = ctx.createAnalyser();
    analyser.fftSize = 128;
    analyser.smoothingTimeConstant = 0.78;
    src.connect(analyser);
    return true;
  } catch (e) {
    showSubtitle('Mic access denied', false);
    return false;
  }
}

// =========================================================================
// Voice — tap face to speak
// =========================================================================
faceContainer.addEventListener('click', async (e) => {
  e.stopPropagation();

  if (!wsReady) { showSubtitle('Connecting\u2026', false); return; }
  if (currentState === 'thinking') return;

  // Tap while speaking — stop audio
  if (currentState === 'speaking' && currentAudio) {
    currentAudio.pause(); currentAudio = null;
    setState('idle'); return;
  }

  if (!micStream) {
    const ok = await initMic();
    if (!ok) return;
  }

  isRecording ? stopRecording() : startRecording();
});

function startRecording() {
  audioChunks = [];
  const mime = ['audio/webm;codecs=opus','audio/webm','audio/ogg','']
    .find(t => !t || MediaRecorder.isTypeSupported(t));
  const opts = mime ? {mimeType: mime} : {};
  try {
    mediaRecorder = new MediaRecorder(micStream, opts);
  } catch {
    mediaRecorder = new MediaRecorder(micStream);
  }
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
  mediaRecorder.onstop = sendVoice;
  mediaRecorder.start(100);
  isRecording = true;
  setState('listening');
  dbg('\uD83C\uDF99 recording...');

  // ---- Silence / VAD detector ----
  const SILENCE_THRESHOLD = 8;
  const SILENCE_TIMEOUT   = 1500;
  let lastSpeechMs = Date.now();
  let silenceMs    = 0;

  silenceTimer = setInterval(() => {
    if (!analyser || !isRecording) return;
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    const rms = data.reduce((s, v) => s + v, 0) / data.length;

    // Log level every 500ms to avoid spam
    const now = Date.now();
    if (now - _lastLevelLog >= 500) {
      dbg('\uD83C\uDF99 level: ' + Math.round(rms));
      _lastLevelLog = now;
    }

    if (rms >= SILENCE_THRESHOLD) {
      lastSpeechMs = Date.now();
      silenceMs    = 0;
    } else {
      silenceMs = Date.now() - lastSpeechMs;
      if (silenceMs > 0 && silenceMs % 300 < 150) {
        dbg('\uD83E\uDD2B silence ' + (Math.round(silenceMs / 100) / 10) + 's');
      }
      if (silenceMs >= SILENCE_TIMEOUT && isRecording) {
        clearInterval(silenceTimer);
        silenceTimer = null;
        dbg('\uD83E\uDD2B auto-stop (silence)');
        stopRecording();
      }
    }
  }, 150);
}

function stopRecording() {
  if (silenceTimer) { clearInterval(silenceTimer); silenceTimer = null; }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  isRecording = false;
  setState('thinking');
  dbg('\u23F9 stopped \u2014 sending...');
}

async function sendVoice() {
  if (!audioChunks.length) { setState('idle'); return; }
  const mime = audioChunks[0]?.type || 'audio/webm';
  const blob = new Blob(audioChunks, {type: mime});
  const form = new FormData();
  form.append('audio', blob, 'voice.webm');
  dbg('\uD83D\uDCE4 uploading ' + (blob.size / 1024).toFixed(1) + 'kb...');

  try {
    let url = '/voice?voice=' + encodeURIComponent(selectedVoice);
    if (voiceSessionId) url += '&session_id=' + encodeURIComponent(voiceSessionId);
    const resp = await fetch(url, {method: 'POST', body: form});
    const data = await resp.json();
    if (data.session_id) _saveVoiceSession(data.session_id);

    if (data.error) {
      dbg('\u274c ' + data.error);
      showSubtitle('\u274c ' + data.error, false); setState('idle'); return;
    }
    if (data.transcription) {
      dbg('\uD83D\uDCDD transcript: ' + data.transcription.slice(0, 30));
      showSubtitle(data.transcription, false);
    }

    if (data.audio_url) {
      currentAudio = new Audio(data.audio_url);
      currentAudio.oncanplaythrough = () => {
        dbg('\uD83D\uDD0A playing response');
        setState('speaking');
        if (data.response) setTimeout(() => showSubtitle(data.response, true), 250);
        currentAudio.play().catch(() => {});
      };
      currentAudio.onended  = () => { currentAudio = null; setState('idle'); };
      currentAudio.onerror  = () => { currentAudio = null; setState('idle'); };
    } else if (data.response) {
      setState('speaking');
      showSubtitle(data.response, true);
      setTimeout(() => setState('idle'), 5200);
    } else {
      setState('idle');
    }
  } catch (err) {
    dbg('\u274c ' + err.message);
    showSubtitle('\u274c Request failed', false);
    setState('idle');
  }
}

// =========================================================================
// Waveform — Web Audio AnalyserNode → canvas bars
// =========================================================================
function resizeWaveCanvas() {
  dpr = window.devicePixelRatio || 1;
  const w = waveCanvas.offsetWidth;
  const h = waveCanvas.offsetHeight;
  if (!w || !h) return;
  waveCanvas.width  = w * dpr;
  waveCanvas.height = h * dpr;
}
resizeWaveCanvas();
window.addEventListener('resize', resizeWaveCanvas);

function drawWaveform() {
  requestAnimationFrame(drawWaveform);
  const W = waveCanvas.width;
  const H = waveCanvas.height;
  if (!W || !H) return;
  waveCtx.clearRect(0, 0, W, H);
  if (!analyser) return;

  const bufLen = analyser.frequencyBinCount;
  const data   = new Uint8Array(bufLen);
  analyser.getByteFrequencyData(data);

  const barW     = W / bufLen;
  const speaking = currentState === 'speaking';

  for (let i = 0; i < bufLen; i++) {
    const v     = data[i] / 255;
    const h     = v * H * 0.88;
    if (h < 1) continue;
    const alpha = 0.38 + v * 0.62;
    waveCtx.fillStyle = speaking
      ? `rgba(255,255,255,${alpha})`
      : `rgba(212,175,55,${alpha})`;
    const x = i * barW;
    const y = H - h;
    const rw = Math.max(1, barW - 1);
    waveCtx.beginPath();
    // Simple rounded top
    if (waveCtx.roundRect) {
      waveCtx.roundRect(x, y, rw, h, Math.min(2, rw * 0.4));
    } else {
      waveCtx.rect(x, y, rw, h);
    }
    waveCtx.fill();
  }
}
drawWaveform();

// =========================================================================
// Three.js Particle Field
// =========================================================================
(function initParticles() {
  if (typeof THREE === 'undefined') return;

  const canvas   = document.getElementById('particles');
  const W = window.innerWidth, H = window.innerHeight;

  const renderer = new THREE.WebGLRenderer({canvas, alpha: true, antialias: false});
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, W / H, 0.1, 100);
  camera.position.z = 5;

  // Viewport half-extents at z=0 (plus margin so particles wrap off-screen)
  const halfH = Math.tan((75 / 2) * (Math.PI / 180)) * 5;
  const halfW = halfH * (W / H);
  const BX = halfW + 2.5, BY = halfH + 2.0;

  const N   = 200;
  const pos = new Float32Array(N * 3);
  const vel = new Float32Array(N * 2);

  for (let i = 0; i < N; i++) {
    pos[i*3]   = (Math.random() - 0.5) * BX * 2;
    pos[i*3+1] = (Math.random() - 0.5) * BY * 2;
    pos[i*3+2] = (Math.random() - 0.5) * 2.0;
    vel[i*2]   = (Math.random() - 0.5) * 0.0065;
    vel[i*2+1] = (Math.random() - 0.5) * 0.0065;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));

  const mat = new THREE.PointsMaterial({
    color: 0xD4AF37,
    size: 1.5,
    sizeAttenuation: false,
    transparent: true,
    opacity: 0.62,
  });

  scene.add(new THREE.Points(geo, mat));

  function tick() {
    requestAnimationFrame(tick);
    const p = geo.attributes.position;
    for (let i = 0; i < N; i++) {
      const x = p.getX(i), y = p.getY(i);
      // Gravitational drag near center — particles close to origin drift slower
      const dist  = Math.sqrt(x * x + y * y);
      const speed = Math.min(1.0, dist / 3.8);
      let nx = x + vel[i*2]   * speed;
      let ny = y + vel[i*2+1] * speed;
      // Wrap at viewport edges
      if (nx >  BX) nx = -BX;
      if (nx < -BX) nx =  BX;
      if (ny >  BY) ny = -BY;
      if (ny < -BY) ny =  BY;
      p.setX(i, nx); p.setY(i, ny);
    }
    p.needsUpdate = true;
    renderer.render(scene, camera);
  }
  tick();

  window.addEventListener('resize', () => {
    const W2 = window.innerWidth, H2 = window.innerHeight;
    camera.aspect = W2 / H2;
    camera.updateProjectionMatrix();
    renderer.setSize(W2, H2);
  });
})();

// =========================================================================
// PWA Service Worker registration
// =========================================================================
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  });
}

// =========================================================================
// VOICE SETTINGS
// =========================================================================
let selectedVoice = localStorage.getItem('luci-voice') || 'luci-female';
let voiceSpeed    = parseFloat(localStorage.getItem('luci-voice-speed') || '1.0');
let voicesData    = {};

async function loadVoices() {
  try {
    const resp = await fetch('/voices');
    const data = await resp.json();
    voicesData = data.voices || {};
    if (!voicesData[selectedVoice]) {
      selectedVoice = data.default || 'luci-female';
      localStorage.setItem('luci-voice', selectedVoice);
    }
    renderVoiceList();
  } catch {}
}

function renderVoiceList() {
  const list = document.getElementById('voice-list');
  if (!list) return;
  list.innerHTML = Object.entries(voicesData).map(([id, label]) =>
    '<div class="voice-option ' + (id === selectedVoice ? 'selected' : '') + '" onclick="selectVoice(\'' + id + '\')">' +
    '<span class="voice-option-name">' + escHtml(label) + '</span>' +
    '<button class="voice-preview-btn" id="preview-btn-' + id + '" onclick="event.stopPropagation();previewVoice(\'' + id + '\',this)">\u25B6 Play</button>' +
    '</div>'
  ).join('');
}

function selectVoice(id) {
  selectedVoice = id;
  localStorage.setItem('luci-voice', id);
  renderVoiceList();
}

let previewAudio = null;
function previewVoice(id, btn) {
  if (previewAudio) { previewAudio.pause(); previewAudio = null; }
  document.querySelectorAll('.voice-preview-btn').forEach(b => {
    b.textContent = '\u25B6 Play'; b.classList.remove('playing');
  });
  btn.textContent = '\u23F8 Stop'; btn.classList.add('playing');
  previewAudio = new Audio('/preview/' + id);
  previewAudio.onended = () => { btn.textContent = '\u25B6 Play'; btn.classList.remove('playing'); previewAudio = null; };
  previewAudio.onerror = () => { btn.textContent = '\u25B6 Play'; btn.classList.remove('playing'); previewAudio = null; };
  previewAudio.play().catch(() => {});
}

const speedSlider = document.getElementById('voice-speed');
const speedVal    = document.getElementById('voice-speed-val');
if (speedSlider) {
  speedSlider.value = voiceSpeed;
  speedVal.textContent = voiceSpeed.toFixed(1) + '\xD7';
  speedSlider.addEventListener('input', () => {
    voiceSpeed = parseFloat(speedSlider.value);
    speedVal.textContent = voiceSpeed.toFixed(1) + '\xD7';
    localStorage.setItem('luci-voice-speed', voiceSpeed);
  });
}

document.getElementById('voice-settings-btn')?.addEventListener('click', () => {
  const panel = document.getElementById('voice-settings-panel');
  panel.style.display = panel.style.display === 'block' ? 'none' : 'block';
  if (panel.style.display === 'block' && Object.keys(voicesData).length === 0) loadVoices();
});
document.getElementById('voice-settings-close')?.addEventListener('click', () => {
  document.getElementById('voice-settings-panel').style.display = 'none';
});

// =========================================================================
// ALWAYS-LISTENING ENGINE
// =========================================================================
let alActive    = false;
let alRecording = false;
let alSilTimer  = null;
let alSpeaking  = false;
let alAudio     = null;
let alRecorder  = null;
let alChunks    = [];

const AL_SPEECH_THRESHOLD = 12;
const AL_SILENCE_MS       = 1200;

async function startAlwaysListen() {
  if (!micStream) { const ok = await initMic(); if (!ok) return; }
  alActive = true;
  const btn = document.getElementById('al-btn');
  btn.classList.add('active');
  btn.textContent = '\uD83D\uDC42 Listening\u2026';
  localStorage.setItem('luci-always-listen', 'true');
  alMonitorLoop();
}

function stopAlwaysListen() {
  alActive = false; alRecording = false;
  if (alRecorder && alRecorder.state !== 'inactive') alRecorder.stop();
  if (alSilTimer) { clearInterval(alSilTimer); alSilTimer = null; }
  const btn = document.getElementById('al-btn');
  btn.classList.remove('active', 'interrupted');
  btn.textContent = '\uD83D\uDC42 Always Listen';
  localStorage.setItem('luci-always-listen', 'false');
}

function alMonitorLoop() {
  if (!alActive || !analyser) return;
  const bufLen = analyser.frequencyBinCount;
  const data   = new Uint8Array(bufLen);
  function tick() {
    if (!alActive) return;
    requestAnimationFrame(tick);
    analyser.getByteFrequencyData(data);
    const rms = data.reduce((s,v) => s+v, 0) / bufLen;
    if (rms >= AL_SPEECH_THRESHOLD && !alRecording) {
      if (alSpeaking && alAudio) {
        alAudio.pause(); alAudio = null; alSpeaking = false;
        setState('idle'); setChatFaceState('idle');
        const btn = document.getElementById('al-btn');
        btn.classList.add('interrupted');
        setTimeout(() => btn?.classList.remove('interrupted'), 800);
        dbg('\u26A1 interrupted LUCI');
      }
      alStartCapture();
    }
  }
  tick();
}

function alStartCapture() {
  if (alRecording || !alActive) return;
  alRecording = true; alChunks = [];
  const mime = ['audio/webm;codecs=opus','audio/webm','audio/ogg','']
    .find(t => !t || MediaRecorder.isTypeSupported(t));
  try { alRecorder = new MediaRecorder(micStream, mime ? {mimeType: mime} : {}); }
  catch { alRecorder = new MediaRecorder(micStream); }
  alRecorder.ondataavailable = e => { if (e.data.size > 0) alChunks.push(e.data); };
  alRecorder.onstop = alProcessCapture;
  alRecorder.start(100);
  let lastSpeechMs = Date.now(), silenceMs = 0;
  alSilTimer = setInterval(() => {
    if (!alActive || !alRecording) { clearInterval(alSilTimer); return; }
    const d2 = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(d2);
    const rms2 = d2.reduce((s,v) => s+v, 0) / d2.length;
    if (rms2 >= AL_SPEECH_THRESHOLD) { lastSpeechMs = Date.now(); silenceMs = 0; }
    else {
      silenceMs = Date.now() - lastSpeechMs;
      if (silenceMs >= AL_SILENCE_MS) {
        clearInterval(alSilTimer); alSilTimer = null;
        if (alRecorder && alRecorder.state !== 'inactive') alRecorder.stop();
      }
    }
    dbg('\uD83C\uDF99 rms:' + Math.round(rms2) + ' sil:' + Math.round(silenceMs/100)/10 + 's');
  }, 150);
  setState('listening'); setChatFaceState('listening');
  dbg('\uD83C\uDF99 capturing...');
}

async function alProcessCapture() {
  alRecording = false;
  if (!alChunks.length || !alActive) { setState('idle'); setChatFaceState('idle'); return; }
  const blob = new Blob(alChunks, {type: 'audio/webm'});
  if (blob.size < 2000) { setState('idle'); setChatFaceState('idle'); dbg('\u23ED too short'); return; }
  setState('thinking'); setChatFaceState('thinking');
  dbg('\uD83D\uDCE4 sending...');
  const form = new FormData();
  form.append('audio', blob, 'voice.webm');
  try {
    let alUrl = '/voice?voice=' + encodeURIComponent(selectedVoice);
    if (voiceSessionId) alUrl += '&session_id=' + encodeURIComponent(voiceSessionId);
    const resp = await fetch(alUrl, {method:'POST', body:form});
    const data = await resp.json();
    if (data.session_id) _saveVoiceSession(data.session_id);
    if (data.transcription) {
      dbg('\uD83D\uDCDD ' + data.transcription.slice(0,30));
      showSubtitle(data.transcription, false);
      if (mode === 'chat') {
        if (!activeConvoId || !getActiveConvo()) newConversation();
        const isFirst = getActiveConvo().messages.length === 0;
        addMessage('user', '\uD83D\uDC42 ' + data.transcription);
        if (isFirst) autoTitleConvo(activeConvoId, data.transcription);
        if (data.response) addMessage('luci', data.response, {model: data.model || ''});
      }
    }
    if (data.audio_url) {
      alSpeaking = true;
      setState('speaking'); setChatFaceState('speaking');
      if (data.response) setTimeout(() => showSubtitle(data.response, true), 300);
      alAudio = new Audio(data.audio_url);
      alAudio.onended  = () => { alSpeaking = false; alAudio = null; setState('idle'); setChatFaceState('idle'); };
      alAudio.onerror  = () => { alSpeaking = false; alAudio = null; setState('idle'); setChatFaceState('idle'); };
      alAudio.play().catch(() => { alSpeaking = false; setState('idle'); setChatFaceState('idle'); });
    } else if (data.response) {
      setState('speaking'); setChatFaceState('speaking');
      showSubtitle(data.response, true);
      setTimeout(() => { setState('idle'); setChatFaceState('idle'); }, 5000);
    } else {
      setState('idle'); setChatFaceState('idle');
    }
  } catch (err) {
    dbg('\u274C ' + err.message); setState('idle'); setChatFaceState('idle');
  }
}

document.getElementById('al-btn')?.addEventListener('click', () => {
  alActive ? stopAlwaysListen() : startAlwaysListen();
});

// Auto-restore always-listen if it was active before
if (localStorage.getItem('luci-always-listen') === 'true') {
  document.addEventListener('click', () => { if (!alActive) startAlwaysListen(); }, {once: true});
}

// =========================================================================
// MODE TOGGLE
// =========================================================================
let mode = localStorage.getItem('luci-mode') || 'ambient';

function setMode(m) {
  mode = m;
  localStorage.setItem('luci-mode', m);
  document.getElementById('ambient-mode').style.display = m === 'ambient' ? 'block' : 'none';
  document.getElementById('chat-mode').style.display    = m === 'chat'    ? 'flex'  : 'none';
  document.getElementById('mode-toggle').textContent    = m === 'ambient' ? '\uD83D\uDCAC' : '\u2726';
}
document.getElementById('mode-toggle').addEventListener('click', () => {
  setMode(mode === 'ambient' ? 'chat' : 'ambient');
});
setMode(mode);

// =========================================================================
// CHAT MODE — Conversation storage (localStorage)
// =========================================================================
const STORAGE_KEY = 'luci-conversations';

function loadConversations() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
  catch { return []; }
}
function saveConversations(convos) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(convos));
}

let conversations = loadConversations();
let activeConvoId = null;

function newConversation() {
  const id = 'conv-' + Date.now();
  const convo = { id, title: 'New conversation', createdAt: Date.now(), updatedAt: Date.now(), messages: [], memoryTags: [] };
  conversations.unshift(convo);
  saveConversations(conversations);
  setActiveConvo(id);
  renderHistoryList();
  return convo;
}

function setActiveConvo(id) {
  activeConvoId = id;
  renderHistoryList();
  renderMessages();
}

function getActiveConvo() {
  return conversations.find(c => c.id === activeConvoId);
}

// =========================================================================
// CHAT MODE — History list
// =========================================================================
function renderHistoryList() {
  const list = document.getElementById('history-list');
  if (!list) return;
  if (conversations.length === 0) {
    list.innerHTML = '<div style="padding:20px 8px;font-size:12px;color:rgba(245,230,200,0.25);text-align:center">No conversations yet</div>';
    return;
  }
  const today     = new Date().toDateString();
  const yesterday = new Date(Date.now() - 86400000).toDateString();
  const groups = {};
  conversations.forEach(c => {
    const d = new Date(c.updatedAt).toDateString();
    const label = d === today ? 'Today' : d === yesterday ? 'Yesterday'
      : new Date(c.updatedAt).toLocaleDateString('en', {month:'short', day:'numeric'});
    if (!groups[label]) groups[label] = [];
    groups[label].push(c);
  });
  list.innerHTML = Object.entries(groups).map(([label, items]) =>
    '<div class="history-group-label">' + label + '</div>' +
    items.map(c =>
      '<div class="history-item ' + (c.id === activeConvoId ? 'active' : '') + '" onclick="setActiveConvo(\'' + c.id + '\')" data-id="' + c.id + '">' +
        escHtml(c.title) +
        '<button class="del-btn" title="Delete conversation" onclick="event.stopPropagation();deleteConversation(\'' + c.id + '\')">✕</button>' +
      '</div>'
    ).join('')
  ).join('');
}

// =========================================================================
// CHAT MODE — Messages
// =========================================================================
function renderMsgText(text) {
  // Escape HTML first, then convert safe markdown patterns
  let s = escHtml(text);
  // Markdown links [label](/output/...) → clickable anchor
  s = s.replace(/\[([^\]]+)\]\((\/output\/[^)]+)\)/g,
    '<a href="$2" target="_blank" style="color:#D4AF37;text-decoration:underline">$1</a>'
  );
  // **bold**
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // `code`
  s = s.replace(/`([^`]+)`/g,
    '<code style="background:rgba(255,255,255,0.08);padding:1px 5px;border-radius:3px;font-size:12px">$1</code>'
  );
  return s;
}

function renderMessages() {
  const el = document.getElementById('messages');
  if (!el) return;
  const convo = getActiveConvo();
  if (!convo) { el.innerHTML = ''; return; }
  el.innerHTML = convo.messages.map(m => {
    const memBadge = m.memoryKey
      ? '<span class="mem-tag">\uD83C\uDFF7 ' + escHtml(m.memoryKey.replace('mem_','').replace(/_/g,' ')) + '</span>'
      : '';
    const modelBadge = m.model ? '<span class="model-tag">' + escHtml(m.model) + '</span>' : '';
    return '<div class="msg-row ' + m.role + '"><div class="msg-bubble">' + renderMsgText(m.text) + memBadge + modelBadge + '</div></div>';
  }).join('');
  const cont = document.getElementById('messages-container');
  if (cont) cont.scrollTop = cont.scrollHeight;
}

function addMessage(role, text, extras) {
  const convo = getActiveConvo();
  if (!convo) return;
  convo.messages.push(Object.assign({role, text, ts: Date.now()}, extras || {}));
  convo.updatedAt = Date.now();
  saveConversations(conversations);
  renderMessages();
}

// =========================================================================
// CHAT MODE — Auto-title
// =========================================================================
async function autoTitleConvo(convoId, firstMessage) {
  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        text: 'Give this conversation a title of 4 words or less, no punctuation, no quotes: "' + firstMessage.slice(0,100) + '"',
        system: 'You are a conversation title generator. Reply with ONLY the title, 4 words max, no punctuation.'
      })
    });
    if (resp.ok) {
      const data = await resp.json();
      const title = (data.response || '').trim().slice(0, 40) || firstMessage.slice(0, 30);
      const convo = conversations.find(c => c.id === convoId);
      if (convo) { convo.title = title; saveConversations(conversations); renderHistoryList(); }
    }
  } catch {}
}

// =========================================================================
// CHAT MODE — Send message
// =========================================================================
async function sendChatMessage(text) {
  text = text.trim();
  if (!text) return;
  if (!activeConvoId || !getActiveConvo()) newConversation();
  const convoId = activeConvoId;
  const isFirst = getActiveConvo().messages.length === 0;
  addMessage('user', text);
  if (isFirst) autoTitleConvo(convoId, text);

  const msgEl   = document.getElementById('messages');
  const typingId = 'typing-' + Date.now();
  msgEl.insertAdjacentHTML('beforeend',
    '<div id="' + typingId + '" class="msg-row luci"><div class="msg-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div></div>'
  );
  const cont = document.getElementById('messages-container');
  if (cont) cont.scrollTop = 999999;
  setChatFaceState('thinking');

  try {
    const convo = getActiveConvo();
    // Pass last 6 messages before the current one as conversation history
    const history = convo ? convo.messages.slice(0, -1).slice(-6) : [];
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, voice: selectedVoice, history})
    });
    const data = await resp.json();
    document.getElementById(typingId)?.remove();
    if (data.error) {
      addMessage('luci', '\u274c ' + data.error);
    } else {
      addMessage('luci', data.response, {model: data.model || '', memoryKey: data.memory_key || null});
      if (data.audio_url) {
        setChatFaceState('speaking');
        const audio = new Audio(data.audio_url);
        audio.onended = () => setChatFaceState('idle');
        audio.onerror = () => setChatFaceState('idle');
        audio.play().catch(() => setChatFaceState('idle'));
      } else {
        setChatFaceState('idle');
      }
    }
  } catch (err) {
    document.getElementById(typingId)?.remove();
    addMessage('luci', '\u274c Request failed: ' + err.message);
    setChatFaceState('idle');
  }
}

// =========================================================================
// CHAT MODE — Chat face state
// =========================================================================
function setChatFaceState(state) {
  const fc = document.getElementById('chat-face-container');
  const fi = document.getElementById('chat-face-img');
  const st = document.getElementById('chat-status-text');
  if (!fc) return;
  const CF = { idle:'/faces/luci_neutral.png', thinking:'/faces/luci_neutral.png', listening:'/faces/luci_listening.png', speaking:'/faces/luci_speaking.png' };
  const LABELS = { idle:'Ready', thinking:'Thinking...', listening:'Listening...', speaking:'Speaking' };
  fc.className = 'state-' + state;
  if (fi) { fi.style.opacity = '0'; setTimeout(() => { fi.src = CF[state] || CF.idle; fi.style.opacity = '1'; }, 220); }
  if (st) st.textContent = LABELS[state] || '';
}

// =========================================================================
// CHAT MODE — Input handlers
// =========================================================================
const chatInput = document.getElementById('chat-input');
const sendBtn   = document.getElementById('send-btn');
const voiceBtn  = document.getElementById('voice-btn');

if (chatInput) {
  chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const txt = chatInput.value; chatInput.value = ''; chatInput.style.height = 'auto';
      sendChatMessage(txt);
    }
  });
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
  });
}
if (sendBtn) {
  sendBtn.addEventListener('click', () => {
    const txt = chatInput.value; chatInput.value = ''; chatInput.style.height = 'auto';
    sendChatMessage(txt);
  });
}

// Voice button in chat mode
let chatRecording = false;
if (voiceBtn) {
  voiceBtn.addEventListener('click', async () => {
    if (!micStream) { const ok = await initMic(); if (!ok) return; }
    if (!chatRecording) {
      chatRecording = true;
      voiceBtn.classList.add('recording');
      setChatFaceState('listening');
      audioChunks = [];
      let chatMR;
      try { chatMR = new MediaRecorder(micStream, {}); } catch { chatMR = new MediaRecorder(micStream); }
      voiceBtn._recorder = chatMR;
      chatMR.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
      chatMR.onstop = async () => {
        chatRecording = false;
        voiceBtn.classList.remove('recording');
        setChatFaceState('thinking');
        const blob = new Blob(audioChunks, {type: 'audio/webm'});
        const form = new FormData();
        form.append('audio', blob, 'voice.webm');
        try {
          let cvUrl = '/voice?voice=' + encodeURIComponent(selectedVoice);
          if (voiceSessionId) cvUrl += '&session_id=' + encodeURIComponent(voiceSessionId);
          const resp = await fetch(cvUrl, {method: 'POST', body: form});
          const data = await resp.json();
          if (data.session_id) _saveVoiceSession(data.session_id);
          if (data.transcription) {
            if (!activeConvoId || !getActiveConvo()) newConversation();
            const isFirst = getActiveConvo().messages.length === 0;
            addMessage('user', '\uD83C\uDF99 ' + data.transcription);
            if (isFirst) autoTitleConvo(activeConvoId, data.transcription);
          }
          if (data.response) addMessage('luci', data.response, {model: data.model || ''});
          if (data.audio_url) {
            setChatFaceState('speaking');
            const audio = new Audio(data.audio_url);
            audio.onended = () => setChatFaceState('idle');
            audio.onerror = () => setChatFaceState('idle');
            audio.play().catch(() => setChatFaceState('idle'));
          } else { setChatFaceState('idle'); }
        } catch { setChatFaceState('idle'); }
      };
      chatMR.start(100);
      // Auto-stop on silence
      let silMs2 = 0, lastSp2 = Date.now();
      const silT2 = setInterval(() => {
        if (!analyser || !chatRecording) { clearInterval(silT2); return; }
        const d2 = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(d2);
        const rms2 = d2.reduce((s,v) => s+v, 0) / d2.length;
        if (rms2 >= 8) { lastSp2 = Date.now(); silMs2 = 0; }
        else { silMs2 = Date.now() - lastSp2; if (silMs2 >= 1500 && chatRecording) { clearInterval(silT2); if (chatMR.state !== 'inactive') chatMR.stop(); } }
      }, 150);
    } else {
      chatRecording = false;
      voiceBtn.classList.remove('recording');
      if (voiceBtn._recorder && voiceBtn._recorder.state !== 'inactive') voiceBtn._recorder.stop();
    }
  });
}

document.getElementById('new-chat-btn')?.addEventListener('click', () => { newConversation(); });

// =========================================================================
// CHAT MODE — Mini particle field (face panel)
// =========================================================================
(function initChatParticles() {
  if (typeof THREE === 'undefined') return;
  const canvas = document.getElementById('chat-particles');
  if (!canvas) return;
  const W = 320, H = window.innerHeight;
  const renderer = new THREE.WebGLRenderer({canvas, alpha: true, antialias: false});
  renderer.setSize(W, H); renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, W/H, 0.1, 100);
  camera.position.z = 5;
  const N = 80, pos = new Float32Array(N*3), vel = new Float32Array(N*2);
  for (let i = 0; i < N; i++) {
    pos[i*3]=(Math.random()-0.5)*6; pos[i*3+1]=(Math.random()-0.5)*10; pos[i*3+2]=(Math.random()-0.5)*2;
    vel[i*2]=(Math.random()-0.5)*0.006; vel[i*2+1]=(Math.random()-0.5)*0.006;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({color:0xD4AF37, size:1.5, sizeAttenuation:false, transparent:true, opacity:0.5});
  scene.add(new THREE.Points(geo, mat));
  function tick() {
    requestAnimationFrame(tick);
    const p = geo.attributes.position;
    for (let i = 0; i < N; i++) {
      let nx = p.getX(i)+vel[i*2], ny = p.getY(i)+vel[i*2+1];
      if (nx>4) nx=-4; if (nx<-4) nx=4; if (ny>6) ny=-6; if (ny<-6) ny=6;
      p.setX(i,nx); p.setY(i,ny);
    }
    p.needsUpdate = true; renderer.render(scene, camera);
  }
  tick();
})();

// =========================================================================
// CHAT MODE — Init
// =========================================================================
if (conversations.length > 0) { setActiveConvo(conversations[0].id); } else { newConversation(); }
renderHistoryList();
loadVoices();

// =========================================================================
// Helper
// =========================================================================
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// =========================================================================
// DELETE CONVERSATION — never touches memory
// =========================================================================
function deleteConversation(id) {
  conversations = conversations.filter(c => c.id !== id);
  saveConversations(conversations);
  if (activeConvoId === id) {
    if (conversations.length > 0) { setActiveConvo(conversations[0].id); }
    else { newConversation(); }
  } else {
    renderHistoryList();
  }
}

// =========================================================================
// FRICTION INTERCEPTOR
// =========================================================================
function openFriction() {
  document.getElementById('friction-overlay').classList.add('open');
  document.getElementById('fi-what').focus();
}
function closeFriction() {
  document.getElementById('friction-overlay').classList.remove('open');
}
async function fireFriction() {
  const what     = document.getElementById('fi-what').value.trim();
  const expected = document.getElementById('fi-expected').value.trim();
  const actual   = document.getElementById('fi-actual').value.trim();
  if (!what) { document.getElementById('fi-what').focus(); return; }

  const btn  = document.getElementById('fi-unblock-btn');
  const resp = document.getElementById('fi-response');
  btn.disabled  = true;
  btn.textContent = 'ANALYZING...';
  resp.style.display = 'block';
  resp.style.color   = '#4a5168';
  resp.textContent   = 'LUCI is analyzing your friction point...';

  try {
    const secret = window.__luciSecret || '';
    const r = await fetch('/friction', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-LUCI-Secret': secret },
      body: JSON.stringify({ what, expected, actual })
    });
    const data = await r.json();
    if (data.ok) {
      // Colorize section labels
      resp.innerHTML = data.response
        .split('\n')
        .map(line => {
          if (/^(DIAGNOSIS|FIX|FALLBACK|VERIFY):/.test(line)) {
            return '<span class="fi-section-label">' + escHtml(line) + '</span>';
          }
          return escHtml(line);
        })
        .join('\n');
      resp.style.color = '#c8cedd';
    } else {
      resp.style.color   = '#ff4455';
      resp.textContent   = 'Error: ' + (data.error || 'unknown');
    }
  } catch(e) {
    resp.style.color   = '#ff4455';
    resp.textContent   = 'Network error: ' + e.message;
  }
  btn.disabled    = false;
  btn.textContent = 'UNBLOCK →';
}

// =========================================================================
// MEMORY MAINTENANCE
// =========================================================================
let _memoryData = [];

function openMemory() {
  document.getElementById('memory-overlay').classList.add('open');
  loadMemoryEntries();
}
function closeMemory() {
  document.getElementById('memory-overlay').classList.remove('open');
}
async function loadMemoryEntries() {
  const listEl = document.getElementById('mem-list');
  listEl.innerHTML = '<div style="color:#4a5168;font-size:12px;padding:20px 0;">Loading memory...</div>';
  try {
    const secret = window.__luciSecret || '';
    const r = await fetch('/memory/list', {
      headers: { 'X-LUCI-Secret': secret }
    });
    if (!r.ok) { throw new Error('HTTP ' + r.status); }
    const data = await r.json();
    _memoryData = data.memories || [];
    renderMemoryList();
  } catch(e) {
    listEl.innerHTML = '<div style="color:#ff4455;font-size:12px;padding:20px 0;">Error: ' + escHtml(e.message) + '</div>';
  }
}
function renderMemoryList() {
  const listEl  = document.getElementById('mem-list');
  const query   = (document.getElementById('mem-search')?.value || '').toLowerCase().trim();
  const entries = query
    ? _memoryData.filter(m => (m.summary || m.embedding_text || m.text || m.content || '').toLowerCase().includes(query))
    : _memoryData;

  if (entries.length === 0) {
    listEl.innerHTML = '<div style="color:#4a5168;font-size:12px;padding:20px 0;text-align:center;">No memories found.</div>';
    return;
  }
  listEl.innerHTML = entries.map((m, i) => {
    const text    = escHtml(m.summary || m.embedding_text || m.text || m.content || m.value || JSON.stringify(m));
    const key     = escHtml(m.key || (m.id ? m.id.replace(/^mem_/, '').replace(/_/g, ' ') : ('entry-' + i)));
    const salience = m.salience != null ? ' · salience ' + m.salience : '';
    return (
      '<div class="mem-entry">' +
        '<div class="mem-entry-text">' +
          '<div class="mem-entry-key">' + key + salience + '</div>' +
          '<div>' + text + '</div>' +
        '</div>' +
        '<button class="mem-del-btn" title="Delete this memory" ' +
          'onclick="deleteMemoryEntry(' + JSON.stringify(m.key || m.id || i) + ')">✕</button>' +
      '</div>'
    );
  }).join('');
}
async function deleteMemoryEntry(key) {
  if (!confirm('Delete this memory entry? This cannot be undone.')) return;
  try {
    const secret = window.__luciSecret || '';
    const r = await fetch('/memory/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-LUCI-Secret': secret },
      body: JSON.stringify({ key })
    });
    const data = await r.json();
    if (data.ok) {
      _memoryData = _memoryData.filter(m => (m.key || m.id) !== key);
      renderMemoryList();
    } else {
      alert('Delete failed: ' + (data.error || 'unknown'));
    }
  } catch(e) {
    alert('Network error: ' + e.message);
  }
}

</script>

  <!-- ═══ Friction Interceptor overlay ═══ -->
  <div id="friction-overlay" onclick="if(event.target===this)closeFriction()">
    <div id="friction-box">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;">
        <h2 style="margin:0;">⚡ Friction Interceptor</h2>
        <button id="fi-close-btn" onclick="closeFriction()">✕ close</button>
      </div>
      <div style="display:grid;gap:14px;">
        <div>
          <div class="fi-label">What were you trying to do</div>
          <textarea id="fi-what" class="fi-input" rows="3" placeholder="e.g. Connect Alpaca API, run Docker container..."></textarea>
        </div>
        <div>
          <div class="fi-label">What did you expect</div>
          <input id="fi-expected" class="fi-input" type="text" placeholder="e.g. Server starts on port 8000" style="resize:none;" />
        </div>
        <div>
          <div class="fi-label">What actually happened</div>
          <textarea id="fi-actual" class="fi-input" rows="3" placeholder="Paste error or describe symptom"></textarea>
        </div>
        <button id="fi-unblock-btn" onclick="fireFriction()">UNBLOCK →</button>
      </div>
      <div id="fi-response"></div>
    </div>
  </div>

  <!-- ═══ Memory Maintenance overlay ═══ -->
  <div id="memory-overlay" onclick="if(event.target===this)closeMemory()">
    <div id="memory-box">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <h2 style="margin:0;">🧠 Memory</h2>
        <button id="mem-close-btn" onclick="closeMemory()">✕ close</button>
      </div>
      <div class="mem-subtitle">
        LUCI's active memories — injected into every conversation.<br>
        Deleting here removes from active memory only. Chat history is separate.
      </div>
      <input id="mem-search" type="text" placeholder="Search memories..." oninput="renderMemoryList()" />
      <div id="mem-list">Loading...</div>
    </div>
  </div>

</body>
</html>
"""


def _build_html() -> str:
    return _HTML


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_build_html())


@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "model": MODEL,
        "voice": PIPER_BIN.exists(),
        "memory": MEMORY_PATH.exists(),
        "timestamp": time.time(),
    })


@app.get("/manifest.json")
async def manifest() -> JSONResponse:
    return JSONResponse({
        "name": "LUCI",
        "short_name": "LUCI",
        "description": "Your personal AI agent",
        "start_url": "/",
        "display": "fullscreen",
        "background_color": "#0a0705",
        "theme_color": "#D4AF37",
        "orientation": "portrait",
        "icons": [
            {"src": "/faces/luci_neutral.png", "sizes": "512x512", "type": "image/png"},
        ],
    })


@app.get("/sw.js")
async def service_worker() -> PlainTextResponse:
    return PlainTextResponse(_SW_JS, media_type="application/javascript")


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    if not re.match(r"^((reply|web_reply|chat_reply)_\d+|preview_[a-z\-]+)\.wav$", filename):
        return JSONResponse({"error": "not found"}, status_code=404)
    path = RUNS_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(path), media_type="audio/wav")


@app.get("/output/{filename}")
async def serve_output(filename: str) -> PlainTextResponse:
    if not re.match(r"^full_output_\d+\.txt$", filename):
        return PlainTextResponse("not found", status_code=404)
    path = RUNS_DIR / filename
    if not path.exists():
        return PlainTextResponse("not found", status_code=404)
    return PlainTextResponse(path.read_text(encoding="utf-8"))


@app.get("/voices")
async def list_voices() -> JSONResponse:
    available = {k: v for k, v in PIPER_VOICE_LABELS.items() if PIPER_VOICES[k].exists()}
    return JSONResponse({"voices": available, "default": DEFAULT_VOICE})


@app.get("/preview/{voice_id}")
async def preview_voice(voice_id: str):
    if voice_id not in PIPER_VOICES:
        return JSONResponse({"error": "unknown voice"}, status_code=404)
    preview_path = RUNS_DIR / f"preview_{voice_id}.wav"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not preview_path.exists():
        loop = asyncio.get_running_loop()
        text = "Hello Chip. I am LUCI, your personal AI agent."
        await loop.run_in_executor(None, lambda: tts_to_file(text, preview_path, voice_id))
    if preview_path.exists():
        return FileResponse(str(preview_path), media_type="audio/wav")
    return JSONResponse({"error": "preview failed"}, status_code=500)


@app.post("/voice")
async def voice_upload(request: Request, audio: UploadFile = File(...)):
    voice      = request.query_params.get("voice", DEFAULT_VOICE)
    session_id = request.query_params.get("session_id", "")
    ts = int(time.time())
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = RUNS_DIR / f"web_voice_{ts}.wav"
    reply_path  = RUNS_DIR / f"web_reply_{ts}.wav"

    data = await audio.read()
    upload_path.write_bytes(data)

    # ---- Whisper ----
    await broadcast_state("listening")
    loop = asyncio.get_running_loop()
    text: str = await loop.run_in_executor(None, lambda: stt_transcribe_fast(upload_path))
    if not text:
        upload_path.unlink(missing_ok=True)
        await broadcast_state("idle")
        return JSONResponse({"error": "could not transcribe"})

    await broadcast_transcript(text)

    # ---- Session history ----
    is_new_session = not session_id
    if is_new_session:
        session_id = await loop.run_in_executor(None, vh_create_session)

    history = await loop.run_in_executor(None, lambda: vh_get_history(session_id, limit=10))

    # ---- Ollama ----
    await broadcast_state("thinking")
    model_name, category = route_model(text)
    persona = load_persona()
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    # Inject conversation history
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": text})

    response_text: str = await loop.run_in_executor(
        None, lambda: ollama_chat(messages, 0.4, model_name)
    )
    tag = format_model_tag(model_name, category)

    # ---- Piper TTS ----
    await broadcast_state("speaking")
    wav_ok: bool = await loop.run_in_executor(
        None, lambda: tts_to_file(response_text, reply_path, voice)
    )
    audio_url = f"/audio/web_reply_{ts}.wav" if wav_ok else None

    upload_path.unlink(missing_ok=True)

    # ---- Persist messages ----
    await loop.run_in_executor(None, lambda: vh_add_message(session_id, "user", text))
    await loop.run_in_executor(
        None, lambda: vh_add_message(session_id, "assistant", response_text, audio_url)
    )

    # Auto-title session after first exchange
    if is_new_session:
        short = text[:60].strip()
        asyncio.create_task(asyncio.to_thread(vh_set_title, session_id, short))

    if LUCI_AUTO_MEMORY:
        asyncio.create_task(asyncio.to_thread(auto_extract_memory, text, response_text))

    # Client transitions to idle when audio finishes
    return JSONResponse({
        "transcription": text,
        "response": response_text,
        "model": tag,
        "audio_url": audio_url,
        "session_id": session_id,
    })


@app.get("/voice/sessions")
async def voice_sessions_list() -> JSONResponse:
    """List recent voice sessions."""
    loop = asyncio.get_running_loop()
    sessions = await loop.run_in_executor(None, vh_list_sessions)
    return JSONResponse({"sessions": sessions})


@app.get("/voice/sessions/{sid}")
async def voice_session_get(sid: str) -> JSONResponse:
    """Get all messages for a voice session."""
    loop = asyncio.get_running_loop()
    def _fetch():
        conn = _vhdb()
        session_row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM vh_sessions WHERE id=?", (sid,)
        ).fetchone()
        if not session_row:
            return None
        msgs = conn.execute(
            "SELECT role, content, audio_url, ts FROM vh_messages WHERE session_id=? ORDER BY ts",
            (sid,),
        ).fetchall()
        conn.close()
        return {
            "session": dict(session_row),
            "messages": [dict(m) for m in msgs],
        }
    result = await loop.run_in_executor(None, _fetch)
    if result is None:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return JSONResponse(result)


@app.delete("/voice/sessions/{sid}")
async def voice_session_delete(sid: str) -> JSONResponse:
    """Delete a voice session and all its messages."""
    loop = asyncio.get_running_loop()
    def _delete():
        with _vhdb() as conn:
            conn.execute("DELETE FROM vh_messages WHERE session_id=?", (sid,))
            conn.execute("DELETE FROM vh_sessions WHERE id=?", (sid,))
    await loop.run_in_executor(None, _delete)
    return JSONResponse({"ok": True})



# ─────────────────────────────────────────────────────────────────
# Friction Interceptor Endpoint
# ─────────────────────────────────────────────────────────────────

FRICTION_SYSTEM = (
    "You are LUCI, an execution engine for a senior developer named Chip. "
    "Your ONLY job is to unblock technical friction fast. No fluff, no encouragement. "
    "When given a friction description:\n"
    "1. Diagnose the most likely cause in ONE sentence\n"
    "2. Give the single most direct fix (code snippet if relevant, under 10 lines)\n"
    "3. Give one alternative if the first does not work\n"
    "4. State what to check to confirm it is resolved\n\n"
    "Format your response with exactly these labels on their own lines:\n"
    "DIAGNOSIS:\n"
    "FIX:\n"
    "FALLBACK:\n"
    "VERIFY:\n\n"
    "Be brutal and direct. Chip is technical. Treat him accordingly."
)


# ─────────────────────────────────────────────────────────────────
# Memory Maintenance Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/memory/list")
async def memory_list(request: Request) -> JSONResponse:
    """Return all active beast_memory.json entries for the memory UI."""
    try:
        mem_path = Path(os.getenv("LUCI_WORKSPACE", str(Path(__file__).parent))) / "beast_memory.json"
        if not mem_path.exists():
            return JSONResponse({"memories": [], "source": "beast_memory.json not found"})
        raw = json.loads(mem_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            memories = raw
        elif isinstance(raw, dict):
            memories = (raw.get("memories")
                        or raw.get("active")
                        or raw.get("entries")
                        or [{"key": k, "text": str(v)}
                            for k, v in raw.items()
                            if k not in ("version", "meta")])
        else:
            memories = []
        return JSONResponse({"memories": memories, "count": len(memories)})
    except Exception as e:
        return JSONResponse({"error": str(e), "memories": []}, status_code=500)


@app.post("/memory/delete")
async def memory_delete(request: Request) -> JSONResponse:
    """Delete a single memory entry by key. Never touches chat history."""
    secret   = request.headers.get("X-LUCI-Secret", "")
    expected = os.getenv("LUCI_WEB_SECRET", "")
    if expected and secret != expected:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    key = str(body.get("key", "")).strip()
    if not key:
        return JSONResponse({"error": "no key provided"}, status_code=400)
    try:
        mem_path = Path(os.getenv("LUCI_WORKSPACE", str(Path(__file__).parent))) / "beast_memory.json"
        if not mem_path.exists():
            return JSONResponse({"error": "beast_memory.json not found"}, status_code=404)
        raw     = json.loads(mem_path.read_text(encoding="utf-8"))
        deleted = False
        if isinstance(raw, list):
            before  = len(raw)
            raw     = [m for m in raw if str(m.get("key", m.get("id", ""))) != key]
            deleted = len(raw) < before
        elif isinstance(raw, dict):
            if "memories" in raw:
                before           = len(raw["memories"])
                raw["memories"]  = [m for m in raw["memories"]
                                    if str(m.get("key", m.get("id", ""))) != key]
                deleted = len(raw["memories"]) < before
            elif key in raw:
                del raw[key]
                deleted = True
        if deleted:
            bak = mem_path.with_suffix(".json.membak")
            mem_path.rename(bak)
            mem_path.write_text(
                json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return JSONResponse({"ok": True, "deleted": key})
        else:
            return JSONResponse({"ok": False, "error": f"key not found: {key}"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e), "ok": False}, status_code=500)


@app.post("/friction")
async def friction_endpoint(request: Request) -> JSONResponse:
    """Friction Interceptor — describe a block, get unblocked."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    what     = (body.get("what")     or "").strip()
    expected = (body.get("expected") or "").strip()
    actual   = (body.get("actual")   or "").strip()
    project  = (body.get("project")  or "").strip()

    if not what:
        return JSONResponse({"error": "no friction description"}, status_code=400)

    # Verify secret header
    secret = request.headers.get("X-LUCI-Secret", "")
    expected_secret = os.getenv("LUCI_WEB_SECRET", "")
    if expected_secret and secret != expected_secret:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    user_msg = f"Project: {project or unspecified}\n"
    user_msg += f"What I was trying to do: {what}\n"
    if expected:
        user_msg += f"What I expected: {expected}\n"
    if actual:
        user_msg += f"What actually happened: {actual}\n"

    try:
        loop = asyncio.get_running_loop()
        messages = [
            {"role": "system", "content": FRICTION_SYSTEM},
            {"role": "user",   "content": user_msg},
        ]
        response = await loop.run_in_executor(
            None,
            lambda: ollama_chat(messages, temperature=0.2),
        )
        return JSONResponse({"response": response, "ok": True})
    except Exception as e:
        return JSONResponse({"error": str(e), "ok": False}, status_code=500)

@app.post("/chat")
async def chat_endpoint(request: Request) -> JSONResponse:
    """Text chat endpoint for the chat UI mode."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    text = (body.get("text") or "").strip()
    system_override = body.get("system", "")
    voice = body.get("voice") or DEFAULT_VOICE
    history = body.get("history", [])  # list of {role, text} from client
    if not text:
        return JSONResponse({"error": "no text"}, status_code=400)

    try:
        loop = asyncio.get_running_loop()
        model_name, category = route_model(text)

        # GitHub injection: when keywords detected, fetch real repo data from API
        github_keywords = [
            "github", "repo", "repository", "repositories", "my repos",
            "list repos", "show repos", "my projects", "my code",
        ]
        injected_text = text
        if any(kw in text.lower() for kw in github_keywords):
            try:
                from luci_github import github_list_repos
                repos = github_list_repos()
                repo_list = "\n".join(
                    f"- {r['name']} ({'private' if r.get('private') else 'public'})"
                    for r in (repos[:60] if isinstance(repos, list) else [])
                )
                injected_text = (
                    f"{text}\n\n"
                    f"[SYSTEM: Here are the user's ACTUAL GitHub repositories "
                    f"retrieved live from the API — do NOT make up names, "
                    f"use ONLY these:\n{repo_list}]"
                )
            except Exception:
                pass  # fall through to normal LLM response

        # Weather injection
        _weather_kw = [
            "weather", "temperature", "forecast", "rain", "sunny",
            "hot", "cold", "humidity", "wind", "outside", "climate",
        ]
        if any(kw in text.lower() for kw in _weather_kw):
            try:
                import re as _re
                _location = "The Bronx, New York"
                _loc_m = _re.search(
                    r"in ([A-Z][a-zA-Z\s,]+?)(?:\?|$|\.|today|now|right now)",
                    text
                )
                if _loc_m:
                    _location = _loc_m.group(1).strip()
                _wdata = get_weather(_location)
                injected_text = (
                    f"{injected_text}\n\n"
                    f"[REAL WEATHER DATA — fetched live right now]:\n{_wdata}\n"
                    f"[Report this weather to the user naturally. "
                    f"Do NOT say you cannot get weather.]"
                )
            except Exception:
                pass

        # Time injection
        _time_kw = ["what time", "current time", "what's the time", "time now"]
        if any(kw in text.lower() for kw in _time_kw):
            try:
                injected_text += f"\n\n[REAL TIME DATA]: {get_time('America/New_York')}"
            except Exception:
                pass

        # Email injection — fetch real unread emails when requested
        _email_kw = [
            "email", "emails", "inbox", "unread", "mail", "messages",
            "get my email", "check my email", "check email", "any emails",
            "what emails", "new emails", "my mail",
        ]
        if any(kw in text.lower() for kw in _email_kw):
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).parent))
                from luci import email_list_unread
                _emails = email_list_unread(max_results=8)
                if _emails:
                    _email_lines = []
                    for _i, _e in enumerate(_emails, 1):
                        _email_lines.append(
                            f"{_i}. From: {_e.get('sender','?')}\n"
                            f"   Subject: {_e.get('subject','(no subject)')}\n"
                            f"   Date: {_e.get('date','')}\n"
                            f"   Preview: {_e.get('snippet','')[:120]}"
                        )
                    _email_summary = "\n".join(_email_lines)
                    injected_text = (
                        f"{injected_text}\n\n"
                        f"[EMAILS — live from Gmail]:\n"
                        f"{_email_summary}\n"
                        f"[Report these emails to Chip directly and naturally. "
                        f"Do NOT say you are checking or fetching — just report them. "
                        f"Never invent emails not listed above.]"
                    )
                else:
                    injected_text = (
                        f"{injected_text}\n\n"
                        f"[Gmail reports: no unread emails right now. "
                        f"Tell Chip his inbox is clear.]"
                    )
            except Exception as _ee:
                injected_text = (
                    f"{injected_text}\n\n"
                    f"[Email fetch failed: {_ee}. "
                    f"Tell Chip honestly that email is unavailable right now.]"
                )

        # Calendar injection — fetch today's events when requested
        _cal_kw = [
            "calendar", "schedule", "today", "appointments", "meetings",
            "what's on", "what do i have", "my day", "events today",
        ]
        if any(kw in text.lower() for kw in _cal_kw):
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).parent))
                from luci import calendar_today
                _cal = calendar_today()
                if _cal and _cal.strip():
                    injected_text = (
                        f"{injected_text}\n\n"
                        f"[REAL CALENDAR DATA — today's events fetched live]:\n"
                        f"{_cal}\n"
                        f"<inst>Report these events naturally. Never invent events.</inst>"
                    )
                else:
                    injected_text = (
                        f"{injected_text}\n\n"
                        f"[Calendar: no events scheduled for today.]"
                    )
            except Exception as _ce:
                pass  # calendar unavailable — fail silently

        # Web search injection (Tavily)
        if _LUCI_V2 and should_search(text):
            try:
                search_ctx = search_and_summarize(text[:150])
                injected_text = f"{injected_text}\n\n{search_ctx}"
                print(f"[search] Injected for: {text[:50]}", flush=True)
            except Exception as _se:
                print(f"[search] Error: {_se}", flush=True)

        # Build request detection — hand off to autonomous builder
        if _LUCI_V2 and len(text) > 30 and any(t in text.lower() for t in _BUILD_TRIGGERS):
            try:
                build_id = start_build(text)
                return JSONResponse({
                    "response": (
                        f"On it. Building now — ID: <b>{build_id}</b>\n\n"
                        f"I'll send Telegram updates as I work. "
                        f"Check progress: /build/status/{build_id}"
                    ),
                    "model": "luci-builder",
                    "build_id": build_id,
                })
            except Exception as _be:
                print(f"[builder] Error: {_be}", flush=True)

        # Use persona + memory.md so LUCI knows actual recorded facts
        persona = load_persona_with_memory()
        messages = []
        if system_override:
            messages.append({"role": "system", "content": system_override})
        elif persona:
            # Inject live capability context for self-knowledge questions
            _cap_kw = ["can you", "what can you", "what do you", "tell me about yourself",
                       "what are you", "who are you", "your features", "your capabilities",
                       "what you can do", "features that you have", "abilities", "what all"]
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).parent))
                from luci import build_capability_context
                if any(kw in text.lower() for kw in _cap_kw):
                    cap_ctx = build_capability_context()
                    messages.append({"role": "system", "content": persona + "\n\n" + cap_ctx})
                else:
                    messages.append({"role": "system", "content": persona})
            except Exception:
                messages.append({"role": "system", "content": persona})

        # Inject structured beast_memory.json entries (salience-sorted, top 30)
        try:
            mem_path = Path(__file__).parent / "beast_memory.json"
            if mem_path.exists():
                mem_data = json.loads(mem_path.read_text(encoding="utf-8"))
                raw_mems = mem_data.get("memories", [])
                # active entries only, sorted by salience descending
                active = sorted(
                    [m for m in raw_mems if m.get("status") == "active"],
                    key=lambda m: m.get("salience", 0),
                    reverse=True,
                )
                if active:
                    mem_lines = "\n".join(
                        f"- {m['summary']}" for m in active[:30] if m.get("summary")
                    )
                    mem_block = (
                        "YOUR IDENTITY: You are LUCI (pronounced Lucy). "
                        "You are the AI agent Chip built. Speak in first person always. "
                        "When recalling conversations, say 'we discussed' or 'I remember' — "
                        "never 'Chip's AI agent discussed' or 'LUCI did'.\n\n"
                        "REAL MEMORY (facts you actually know about Chip — "
                        "do NOT invent anything beyond this list):\n"
                        + mem_lines
                        + "\n\nIf asked about past conversations not recorded here, "
                        "say honestly: \"I don't have that in my memory.\" "
                        "Never fabricate history."
                    )
                    if messages and messages[0]["role"] == "system":
                        messages[0]["content"] += "\n\n" + mem_block
                    else:
                        messages.insert(0, {"role": "system", "content": mem_block})
        except Exception:
            pass

        # Final identity anchor — placed last in system so it's closest to the user turn
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += (
                "\n\n---\nYou are LUCI. Respond entirely in first person as LUCI. "
                "Never say 'I am an AI language model' or describe yourself generically. "
                "You are not a generic AI — you are LUCI, Chip's personal agent, "
                "running on his laptop right now.\n\nCRITICAL BEHAVIOR RULES:\n1. When real data has been injected (emails, weather, calendar), report it DIRECTLY and immediately. Never say 'let me check' or 'I will fetch' or 'let me get that' — the data is already in your context. Just report it.\n2. Never invent emails, events, names, or data. If no real data was injected, say so honestly.\n3. Be conversational and immediate. No preamble. No announcements. Just answer."
            )

        # Inject last 6 turns of conversation history for in-session memory
        for h in history[-6:]:
            role = "user" if h.get("role") == "user" else "assistant"
            content = (h.get("text") or "").strip()
            if content:
                messages.append({"role": role, "content": content})

        # For capability questions, rewrite the prompt to force the model to use the list
        _cap_kw = ["can you", "what can you", "what do you", "tell me about yourself",
                   "what are you", "who are you", "your features", "your capabilities",
                   "what you can do", "features that you have", "abilities", "what all"]
        if any(kw in text.lower() for kw in _cap_kw):
            injected_text = (
                "Using ONLY the capability list in your system prompt, "
                "describe everything you can do in 2-3 natural flowing paragraphs. "
                "First person. No numbered lists. No bullet points. "
                "Cover every single item in the list — do not skip any. "
                "Original question: " + text
            )

        messages.append({"role": "user", "content": injected_text})

        await broadcast_state("thinking")

        response_text = await loop.run_in_executor(
            None, lambda: ollama_chat(messages, 0.4, model_name)
        )

        # Auto-summarize large responses — but skip for capability/identity questions
        _cap_kw = ["can you", "what can you", "your features", "your capabilities",
                   "what you can do", "features that you have", "tell me about yourself",
                   "what are you", "who are you", "abilities"]
        _is_cap_question = any(kw in text.lower() for kw in _cap_kw)
        if not _is_cap_question:
            topic = category if category else "output"
            response_text, full_path = await loop.run_in_executor(
                None, lambda: summarize_large_output(response_text, topic)
            )
            if full_path:
                fname = Path(full_path).name
                response_text += f"\n\n📄 [View full output](/output/{fname})"

        # Strip chatbot/apology language using full LUCI persona filter
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent))
            from luci import strip_generic_phrases
            response_text = strip_generic_phrases(response_text)
        except Exception:
            import re as _re
            for _pat in [
                r"I apologize for (?:the )?(?:any )?inconvenience[.,]?\s*",
                r"Is there anything else I can (?:assist|help) you with\??\s*",
                r"(?:Certainly|Of course|Absolutely|Sure thing)[!,]?\s*",
                r"Please let me know if.*$",
            ]:
                response_text = _re.sub(_pat, "", response_text, flags=_re.IGNORECASE | _re.DOTALL)
        response_text = response_text.strip()

        tag = format_model_tag(model_name, category)

        ts = int(time.time())
        reply_path = RUNS_DIR / f"chat_reply_{ts}.wav"
        RUNS_DIR.mkdir(parents=True, exist_ok=True)

        await broadcast_state("speaking")
        wav_ok: bool = await loop.run_in_executor(
            None, lambda: tts_to_file(response_text, reply_path, voice)
        )
        audio_url = f"/audio/{reply_path.name}" if wav_ok else None

        if LUCI_AUTO_MEMORY:
            asyncio.create_task(asyncio.to_thread(auto_extract_memory, text, response_text))

        await broadcast_state("idle")

        return JSONResponse({
            "response": response_text,
            "model": tag,
            "audio_url": audio_url,
        })

    except Exception as e:
        print(f"[web] /chat error: {e}", flush=True)
        return JSONResponse({
            "response": f"❌ Error: {str(e)}",
            "model": "",
            "audio_url": None,
        })


@app.get("/build/status/{build_id}")
async def build_status_endpoint(build_id: str) -> JSONResponse:
    """Get status of an autonomous build."""
    if not _LUCI_V2:
        return JSONResponse({"error": "v2 modules not loaded"}, status_code=503)
    status = get_build_status(build_id)
    return JSONResponse(status)


@app.post("/build")
async def trigger_build_endpoint(request: Request) -> JSONResponse:
    """Trigger an autonomous build via POST."""
    if not _LUCI_V2:
        return JSONResponse({"error": "v2 modules not loaded"}, status_code=503)
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    req_text = data.get("request", "").strip()
    if not req_text:
        return JSONResponse({"error": "No request"}, status_code=400)
    build_id = start_build(req_text)
    return JSONResponse({"build_id": build_id, "status": "started"})


@app.post("/upload")
async def upload_file_endpoint(request: Request) -> JSONResponse:
    """Accept file upload, analyze with LUCI perceive."""
    if not _LUCI_V2:
        return JSONResponse({"error": "v2 modules not loaded"}, status_code=503)
    from fastapi import UploadFile
    import shutil
    try:
        form = await request.form()
        upload = form.get("file")
        if not upload:
            return JSONResponse({"error": "No file field"}, status_code=400)
        filename = upload.filename or "upload.bin"
        uploads_dir = WORKSPACE / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        save_path = uploads_dir / filename
        with open(str(save_path), "wb") as fout:
            content = await upload.read()
            fout.write(content)
        file_type, content_str = perceive_file(str(save_path))
        summary = summarize_perception(file_type, content_str, filename)
        return JSONResponse({
            "filename": filename,
            "type": file_type,
            "content": summary[:3000],
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



@app.get("/audit")
async def audit_endpoint(request: Request) -> JSONResponse:
    """LUCI workspace health check — returns full audit report as JSON."""
    try:
        import importlib, sys
        sys.path.insert(0, str(Path(__file__).parent))
        import luci_audit
        importlib.reload(luci_audit)
        report = luci_audit.run_audit(verbose=False)
        files  = luci_audit.scan_workspace()
        luci_audit.auto_update_registry(files)
        return JSONResponse(report)
    except Exception as e:
        return JSONResponse({"error": str(e), "health": "UNKNOWN"}, status_code=500)


@app.get("/audit/ui", response_class=HTMLResponse)
async def audit_ui_endpoint(request: Request) -> HTMLResponse:
    """Human-readable audit dashboard."""
    try:
        import importlib, sys
        sys.path.insert(0, str(Path(__file__).parent))
        import luci_audit
        importlib.reload(luci_audit)
        report = luci_audit.run_audit(verbose=False)
        files  = luci_audit.scan_workspace()
        luci_audit.auto_update_registry(files)
    except Exception as e:
        report = {"health": "ERROR", "summary": {}, "issues": [{"severity": "CRITICAL", "category": "audit_failed", "file": "luci_audit.py", "detail": str(e)}], "services": {}, "recommendations": []}

    h        = report.get("health", "UNKNOWN")
    s        = report.get("summary", {})
    issues   = report.get("issues", [])
    services = report.get("services", {})
    recs     = report.get("recommendations", [])

    health_color = {"HEALTHY": "#00ff88", "MOSTLY_HEALTHY": "#ffd700", "NEEDS_ATTENTION": "#ff8c00", "DEGRADED": "#ff3333", "ERROR": "#ff0000"}.get(h, "#888888")
    sev_color    = {"CRITICAL": "#ff3333", "WARNING": "#ffd700", "INFO": "#88ccff"}

    issues_html = ""
    for issue in issues:
        c = sev_color.get(issue["severity"], "#ccc")
        issues_html += f'''<div style="background:rgba(255,255,255,0.05);border-left:3px solid {c};padding:10px 14px;margin:6px 0;border-radius:0 6px 6px 0;">
          <span style="color:{c};font-weight:bold;font-size:11px;">{issue["severity"]}</span>
          <span style="color:#aaa;font-size:11px;margin-left:8px;">{issue["category"]}</span>
          <div style="color:#ddd;margin-top:4px;font-size:13px;">{issue["file"]}</div>
          <div style="color:#999;font-size:12px;">{issue["detail"]}</div>
        </div>'''

    services_html = ""
    for svc, info in services.items():
        c = "#00ff88" if info["status"] == "active" else "#ff3333"
        services_html += f'''<div style="display:flex;justify-content:space-between;padding:8px 12px;background:rgba(255,255,255,0.04);border-radius:6px;margin:4px 0;">
          <span style="color:#ddd;">{svc}</span>
          <span style="color:{c};font-weight:bold;">{info["status"].upper()}</span>
        </div>'''

    recs_html   = "".join(f'<div style="padding:6px 0;color:#ffd700;font-size:13px;">→ {r}</div>' for r in recs)
    no_issues   = '<div style="color:#00ff88;padding:20px 0;text-align:center;">✨ No issues found — everything looks good!</div>'
    recs_block  = f"<div class='section'><div class='section-title'>Recommendations</div>{recs_html}</div>" if recs_html else ""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LUCI Health Dashboard</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0a0a0a;color:#e0e0e0;font-family:'SF Mono','Fira Code',monospace;padding:24px}}
  .header{{text-align:center;padding:32px 0 24px}}
  .health-badge{{display:inline-block;background:{health_color}22;border:2px solid {health_color};color:{health_color};font-size:22px;font-weight:bold;padding:12px 32px;border-radius:40px;letter-spacing:2px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:24px 0}}
  .stat{{background:rgba(255,255,255,0.05);border-radius:10px;padding:16px;text-align:center}}
  .stat-val{{font-size:28px;font-weight:bold;color:#D4AF37}}
  .stat-lbl{{font-size:11px;color:#888;margin-top:4px;text-transform:uppercase;letter-spacing:1px}}
  .section{{background:rgba(255,255,255,0.03);border-radius:12px;padding:20px;margin:16px 0}}
  .section-title{{font-size:13px;font-weight:bold;color:#888;text-transform:uppercase;letter-spacing:2px;margin-bottom:14px}}
  .refresh-btn{{display:block;margin:24px auto 0;background:#D4AF37;color:#000;border:none;padding:10px 28px;border-radius:20px;font-size:14px;font-weight:bold;cursor:pointer;letter-spacing:1px}}
  .refresh-btn:hover{{background:#f0c840}}
  .timestamp{{text-align:center;color:#444;font-size:11px;margin-top:16px}}
</style></head><body>
<div class="header">
  <div style="color:#888;font-size:12px;letter-spacing:3px;margin-bottom:12px;">LUCI SYSTEM HEALTH</div>
  <div class="health-badge">{h}</div>
</div>
<div class="grid">
  <div class="stat"><div class="stat-val">{s.get("total_files",0)}</div><div class="stat-lbl">Total Files</div></div>
  <div class="stat"><div class="stat-val">{s.get("python_files",0)}</div><div class="stat-lbl">Python Files</div></div>
  <div class="stat"><div class="stat-val" style="color:{'#00ff88' if s.get('services_active',0)==s.get('services_checked',0) else '#ff8c00'}">{s.get("services_active",0)}/{s.get("services_checked",0)}</div><div class="stat-lbl">Services Active</div></div>
  <div class="stat"><div class="stat-val" style="color:{'#ff3333' if s.get('critical_issues',0) else '#00ff88'}">{s.get("critical_issues",0)}</div><div class="stat-lbl">Critical</div></div>
  <div class="stat"><div class="stat-val" style="color:{'#ffd700' if s.get('warnings',0) else '#00ff88'}">{s.get("warnings",0)}</div><div class="stat-lbl">Warnings</div></div>
</div>
<div class="section"><div class="section-title">Issues</div>{issues_html if issues_html else no_issues}</div>
<div class="section"><div class="section-title">Services</div>{services_html if services_html else '<div style="color:#888;">No services found</div>'}</div>
{recs_block}
<button class="refresh-btn" onclick="location.reload()">↻ REFRESH</button>
<div class="timestamp">Last checked: <span id="ts"></span></div>
<script>document.getElementById("ts").textContent=new Date().toLocaleString();</script>
</body></html>"""
    return HTMLResponse(html)



# ══════════════════════════════════════════════════════
#  LUCI LEARN — Web-based curriculum + quiz endpoints
# ══════════════════════════════════════════════════════

@app.get("/learn", response_class=HTMLResponse)
async def learn_page(request: Request) -> HTMLResponse:
    """Serve the LUCI Learn dashboard."""
    from luci_learn_web import get_learn_html
    return HTMLResponse(get_learn_html())


@app.get("/learn/curriculum")
async def learn_curriculum(request: Request) -> JSONResponse:
    """Return current curriculum state."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from luci_teacher import load_curriculum, get_current_lesson, get_progress_summary
        data   = load_curriculum()
        lesson = get_current_lesson(data)
        total  = sum(
            len(m["lessons"])
            for p in data["phases"]
            for m in p["modules"]
        )
        completed = len(data.get("completed_lessons", []))
        pct = round(completed / total * 100, 1) if total else 0
        return JSONResponse({
            "current_lesson":  lesson,
            "completed":       completed,
            "total_lessons":   total,
            "progress_pct":    pct,
            "phases": [
                {
                    "phase":        p["phase"],
                    "title":        p["title"],
                    "duration":     p["duration"],
                    "module_count": len(p["modules"]),
                }
                for p in data["phases"]
            ],
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/learn/teach")
async def learn_teach(request: Request) -> JSONResponse:
    """Have LUCI teach the current lesson."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from luci_teacher import load_curriculum, get_current_lesson, build_teaching_prompt
        from luci import load_persona_with_memory, strip_generic_phrases
        loop = asyncio.get_running_loop()
        data   = load_curriculum()
        lesson = get_current_lesson(data)
        if not lesson:
            return JSONResponse({"response": "Curriculum complete!"})
        persona       = load_persona_with_memory()
        teach_prompt  = build_teaching_prompt(lesson)
        messages = [
            {"role": "system", "content": persona + "\n\n" + teach_prompt},
            {"role": "user",   "content": f"Teach me: {lesson['lesson_title']}"},
        ]
        response = await loop.run_in_executor(
            None, lambda: ollama_chat(messages, 0.7, route_model(lesson["lesson_title"])[0])
        )
        response = strip_generic_phrases(response)
        return JSONResponse({"response": response})
    except Exception as e:
        return JSONResponse({"error": str(e), "response": f"Error: {e}"}, status_code=500)


@app.post("/learn/chat")
async def learn_chat(request: Request) -> JSONResponse:
    """Follow-up chat during a lesson."""
    try:
        body = await request.json()
        text    = (body.get("text") or "").strip()
        history = body.get("history", [])
        if not text:
            return JSONResponse({"error": "no text"}, status_code=400)
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from luci_teacher import load_curriculum, get_current_lesson, build_teaching_prompt
        from luci import load_persona_with_memory, strip_generic_phrases
        loop = asyncio.get_running_loop()
        data   = load_curriculum()
        lesson = get_current_lesson(data)
        persona      = load_persona_with_memory()
        teach_prompt = build_teaching_prompt(lesson) if lesson else ""
        messages = [{"role": "system", "content": persona + "\n\n" + teach_prompt}]
        for h in history[-6:]:
            messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        messages.append({"role": "user", "content": text})
        response = await loop.run_in_executor(
            None, lambda: ollama_chat(messages, 0.7, route_model(text)[0])
        )
        response = strip_generic_phrases(response)
        return JSONResponse({"response": response})
    except Exception as e:
        return JSONResponse({"error": str(e), "response": f"Error: {e}"}, status_code=500)


@app.get("/learn/quiz/scenarios")
async def learn_quiz_scenarios(request: Request) -> JSONResponse:
    """Get quiz scenarios for current lesson."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from luci_teacher import load_curriculum, get_current_lesson
        from luci_quiz   import get_all_scenarios
        data   = load_curriculum()
        lesson = get_current_lesson(data)
        if not lesson:
            return JSONResponse({"scenarios": []})
        scenarios = get_all_scenarios(lesson["phase"], lesson["module"], lesson["lesson"])
        return JSONResponse({"scenarios": scenarios, "lesson": lesson["lesson_title"]})
    except Exception as e:
        return JSONResponse({"error": str(e), "scenarios": []}, status_code=500)


@app.post("/learn/quiz/evaluate")
async def learn_quiz_evaluate(request: Request) -> JSONResponse:
    """Evaluate a quiz answer."""
    try:
        body = await request.json()
        scenario = (body.get("scenario") or "").strip()
        answer   = (body.get("answer")   or "").strip()
        if not scenario or not answer:
            return JSONResponse({"error": "missing scenario or answer"}, status_code=400)
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from luci_teacher import load_curriculum, get_current_lesson
        from luci_quiz   import build_quiz_prompt, log_quiz_result
        from luci import load_persona_with_memory
        loop = asyncio.get_running_loop()
        data   = load_curriculum()
        lesson = get_current_lesson(data)
        persona    = load_persona_with_memory()
        quiz_prompt = build_quiz_prompt(
            scenario,
            lesson["lesson_title"] if lesson else "General",
            lesson["phase_title"]  if lesson else "General",
        )
        messages = [
            {"role": "system", "content": persona + "\n\n" + quiz_prompt},
            {"role": "user",   "content": f"Here is my answer:\n{answer}\n\nEvaluate it."},
        ]
        response = await loop.run_in_executor(
            None, lambda: ollama_chat(messages, 0.4, route_model(scenario)[0])
        )
        # Detect result from response
        resp_upper = response.upper()
        if "PASS" in resp_upper and "PARTIAL" not in resp_upper and "NEEDS" not in resp_upper:
            result = "PASS"
        elif "PARTIAL" in resp_upper:
            result = "PARTIAL"
        elif "NEEDS REVIEW" in resp_upper or "NEEDS_REVIEW" in resp_upper:
            result = "NEEDS_REVIEW"
        else:
            result = "PARTIAL"
        if lesson:
            log_quiz_result(
                lesson["phase"], lesson["module"], lesson["lesson"],
                scenario, result
            )
        return JSONResponse({"response": response, "result": result})
    except Exception as e:
        return JSONResponse({"error": str(e), "response": f"Error: {e}"}, status_code=500)


@app.get("/learn/quiz/stats")
async def learn_quiz_stats(request: Request) -> JSONResponse:
    """Return quiz performance stats."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from luci_quiz import get_quiz_stats
        return JSONResponse(get_quiz_stats())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/learn/exam")
async def learn_exam(request: Request) -> JSONResponse:
    """Run a phase exam."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from luci_teacher import load_curriculum, get_current_lesson
        from luci_quiz   import build_exam_prompt
        from luci import load_persona_with_memory
        loop = asyncio.get_running_loop()
        data   = load_curriculum()
        lesson = get_current_lesson(data)
        if not lesson:
            return JSONResponse({"response": "Curriculum complete!"})
        completed = [
            k for k in data.get("completed_lessons", [])
            if k.startswith(f"p{lesson['phase']}_")
        ]
        persona     = load_persona_with_memory()
        exam_prompt = build_exam_prompt(lesson["phase"], lesson["phase_title"], completed)
        messages = [
            {"role": "system", "content": persona + "\n\n" + exam_prompt},
            {"role": "user",   "content": "Start the exam. Give me question 1 of 3."},
        ]
        response = await loop.run_in_executor(
            None, lambda: ollama_chat(messages, 0.5, route_model("exam analysis")[0])
        )
        return JSONResponse({"response": response})
    except Exception as e:
        return JSONResponse({"error": str(e), "response": f"Error: {e}"}, status_code=500)


@app.post("/learn/next")
async def learn_next(request: Request) -> JSONResponse:
    """Advance to the next lesson."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from luci_teacher import load_curriculum, save_curriculum, advance_lesson, get_current_lesson
        data   = load_curriculum()
        data   = advance_lesson(data)
        save_curriculum(data)
        lesson = get_current_lesson(data)
        return JSONResponse({
            "ok":     True,
            "lesson": lesson,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    _connected_ws.add(websocket)
    await websocket.send_text('{"type":"status","text":"LUCI online"}')

    try:
        while True:
            raw = await websocket.receive_text()
            msg = _parse_json(raw)
            if not msg or msg.get("type") != "chat":
                continue
            text = str(msg.get("text", "")).strip()
            if not text:
                continue

            await broadcast_state("thinking")

            model_name, category = route_model(text)
            persona = load_persona()
            messages = []
            if persona:
                messages.append({"role": "system", "content": persona})
            messages.append({"role": "user", "content": text})

            try:
                response = await loop.run_in_executor(
                    None, lambda: ollama_chat(messages, 0.4, model_name)
                )
            except Exception as e:
                response = f"❌ Error: {e}"

            tag = format_model_tag(model_name, category)

            # Run Piper TTS in thread while we still have the response
            wav_bytes: bytes | None = await loop.run_in_executor(
                None, lambda: _piper_tts_to_wav_bytes(response)
            )

            await broadcast_state("speaking")
            await websocket.send_text(json.dumps({
                "type": "response",
                "text": response,
                "model": tag,
            }))

            # Send WAV audio as base64 so the browser can play it
            if wav_bytes:
                await websocket.send_text(json.dumps({
                    "type": "audio",
                    "data": base64.b64encode(wav_bytes).decode(),
                    "format": "wav",
                }))
            else:
                # No audio produced — let frontend idle out after subtitle
                await websocket.send_text('{"type":"state","state":"idle"}')

            if LUCI_AUTO_MEMORY:
                asyncio.create_task(
                    asyncio.to_thread(auto_extract_memory, text, response)
                )

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _connected_ws.discard(websocket)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _parse_json(raw: str) -> dict:
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT, log_level="info")
