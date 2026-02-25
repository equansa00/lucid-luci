#!/usr/bin/env python3
"""
LUCI Web Interface — Phase 7B: Holographic Ambient Presence
Full-screen face + Three.js particle field + Web Audio waveform + PWA.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
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
    LUCI_AUTO_MEMORY,
    auto_extract_memory,
    ollama_chat,
    route_model,
    format_model_tag,
    load_persona,
    tts_to_file,
    stt_transcribe,
)

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

  /* === Wake indicator — top right === */
  #wake-indicator {
    position: fixed; top: 20px; right: 24px;
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
    position: fixed; bottom: 14%; left: 50%;
    transform: translateX(-50%);
    width: min(700px, 88vw);
    text-align: center;
    font-size: 17px; line-height: 1.6;
    color: var(--warm);
    text-shadow: 0 0 20px rgba(245,230,200,0.35);
    opacity: 0;
    transition: opacity 0.5s ease;
    z-index: 20; pointer-events: none;
    padding: 0 16px;
  }
  #subtitle.visible  { opacity: 1; }
  #subtitle.response { color: var(--gold); text-shadow: 0 0 22px rgba(212,175,55,0.5); }

  /* === Tap hint === */
  #tap-hint {
    position: fixed; bottom: 6%; left: 50%;
    transform: translateX(-50%);
    font-size: 11px; letter-spacing: 0.18em;
    text-transform: uppercase; color: rgba(212,175,55,0.38);
    pointer-events: none; z-index: 20;
    transition: opacity 0.5s;
  }

  /* === Auth modal === */
  #auth-modal {
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.88);
    display: none; align-items: center; justify-content: center;
    z-index: 100;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
  }
  #auth-box {
    background: var(--surface);
    border: 1px solid rgba(212,175,55,0.28);
    border-radius: 20px; padding: 40px 32px;
    width: min(360px, 90vw);
    display: flex; flex-direction: column; gap: 20px;
    box-shadow: 0 0 70px rgba(212,175,55,0.12);
    pointer-events: all;
  }
  #auth-box h2 {
    font-size: 22px; font-weight: 700;
    letter-spacing: 0.12em; color: var(--gold);
    text-shadow: 0 0 20px rgba(212,175,55,0.4);
  }
  #auth-box p  { font-size: 14px; color: rgba(245,230,200,0.55); }
  #auth-input {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(212,175,55,0.28);
    color: var(--warm); padding: 12px 16px;
    border-radius: 10px; font-size: 15px; outline: none; width: 100%;
    transition: border-color 0.2s;
  }
  #auth-input:focus { border-color: var(--gold); }
  #auth-btn {
    background: var(--gold); color: #0a0705;
    border: none; border-radius: 10px; padding: 13px 20px;
    font-size: 15px; font-weight: 700; letter-spacing: 0.08em;
    cursor: pointer; transition: opacity 0.2s;
  }
  #auth-btn:hover { opacity: 0.85; }
  #auth-error { color: #dc2626; font-size: 13px; display: none; }

  /* === Mobile === */
  @media (max-width: 480px) {
    #face-container { width: 286px; height: 286px; }
    #subtitle { font-size: 15px; }
  }
  @media (max-height: 600px) {
    #face-container { width: 240px; height: 240px; }
    #waveform { height: 40px; margin-top: 14px; }
  }
</style>
</head>
<body>

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
<div id="tap-hint">tap to speak</div>

<!-- Auth modal -->
<div id="auth-modal">
  <div id="auth-box">
    <h2>LUCI</h2>
    <p>Enter access key to connect.</p>
    <input type="password" id="auth-input" placeholder="Access key"
           autocomplete="current-password">
    <button id="auth-btn">Connect</button>
    <span id="auth-error">Wrong key &mdash; try again.</span>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
'use strict';

// injected by server
const SECRET = __SECRET__;

// =========================================================================
// State
// =========================================================================
let currentState  = 'idle';
let ws            = null;
let wsReady       = false;
let storedSecret  = SECRET;

let micStream     = null;
let analyser      = null;
let mediaRecorder = null;
let audioChunks   = [];
let isRecording   = false;
let subtitleTimer = null;
let currentAudio  = null;
let dpr           = window.devicePixelRatio || 1;

// DOM refs
const faceImg       = document.getElementById('face-img');
const faceContainer = document.getElementById('face-container');
const waveCanvas    = document.getElementById('waveform');
const waveCtx       = waveCanvas.getContext('2d');
const subtitleEl    = document.getElementById('subtitle');
const wakeIndicator = document.getElementById('wake-indicator');
const tapHint       = document.getElementById('tap-hint');
const authModal     = document.getElementById('auth-modal');
const authInput     = document.getElementById('auth-input');
const authBtn       = document.getElementById('auth-btn');
const authError     = document.getElementById('auth-error');

const FACES = {
  idle:      '/faces/luci_neutral.png',
  thinking:  '/faces/luci_neutral.png',
  listening: '/faces/luci_listening.png',
  speaking:  '/faces/luci_speaking.png',
};

// =========================================================================
// State machine
// =========================================================================
function setState(next) {
  if (currentState === next) return;
  currentState = next;

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

  ws.onopen = () => {
    if (storedSecret) ws.send(JSON.stringify({type: 'auth', secret: storedSecret}));
  };

  ws.onmessage = (ev) => {
    let d; try { d = JSON.parse(ev.data); } catch { return; }

    if (d.type === 'status') {
      wsReady = true;
      hideAuthModal();
      showSubtitle('LUCI online', true);
      setTimeout(() => subtitleEl.classList.remove('visible', 'response'), 2800);

    } else if (d.type === 'state') {
      setState(d.state);

    } else if (d.type === 'transcript') {
      showSubtitle(d.text, false);

    } else if (d.type === 'response') {
      showSubtitle(d.text, true);
      if (currentState !== 'speaking') setState('speaking');

    } else if (d.type === 'auth_failed') {
      authError.style.display = 'block';
      ws.close();
    }
  };

  ws.onerror = () => {};
  ws.onclose = () => { wsReady = false; setTimeout(connectWS, 3000); };
}

// =========================================================================
// Auth modal
// =========================================================================
function hideAuthModal() { authModal.style.display = 'none'; }

if (SECRET) {
  authModal.style.display = 'flex';
  authBtn.addEventListener('click', () => {
    const pw = authInput.value.trim();
    if (!pw) return;
    storedSecret = pw;
    authError.style.display = 'none';
    connectWS();
  });
  authInput.addEventListener('keydown', e => { if (e.key === 'Enter') authBtn.click(); });
  authInput.focus();
} else {
  connectWS();
}

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
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  isRecording = false;
  setState('thinking');
}

async function sendVoice() {
  if (!audioChunks.length) { setState('idle'); return; }
  const mime = audioChunks[0]?.type || 'audio/webm';
  const blob = new Blob(audioChunks, {type: mime});
  const form = new FormData();
  form.append('audio', blob, 'voice.webm');
  const headers = storedSecret ? {'X-LUCI-SECRET': storedSecret} : {};

  try {
    const resp = await fetch('/voice', {method: 'POST', headers, body: form});
    const data = await resp.json();

    if (data.error) { showSubtitle('\u274c ' + data.error, false); setState('idle'); return; }
    if (data.transcription) showSubtitle(data.transcription, false);

    if (data.audio_url) {
      currentAudio = new Audio(data.audio_url);
      currentAudio.oncanplaythrough = () => {
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
</script>
</body>
</html>
"""


def _build_html() -> str:
    secret_js = '"{}"'.format(WEB_SECRET) if WEB_SECRET else '""'
    return _HTML.replace("__SECRET__", secret_js)


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
    if not re.match(r"^(reply|web_reply)_\d+\.wav$", filename):
        return JSONResponse({"error": "not found"}, status_code=404)
    path = RUNS_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(path), media_type="audio/wav")


@app.post("/voice")
async def voice_upload(request: Request, audio: UploadFile = File(...)):
    # Auth check
    if WEB_SECRET:
        secret = (
            request.headers.get("X-LUCI-SECRET")
            or request.query_params.get("secret", "")
        )
        if secret != WEB_SECRET:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

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

    # ---- Ollama ----
    await broadcast_state("thinking")
    model_name, category = route_model(text)
    persona = load_persona()
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    messages.append({"role": "user", "content": text})

    response_text: str = await loop.run_in_executor(
        None, lambda: ollama_chat(messages, 0.4, model_name)
    )
    tag = format_model_tag(model_name, category)

    # ---- Piper TTS ----
    await broadcast_state("speaking")
    wav_ok: bool = await loop.run_in_executor(
        None, lambda: tts_to_file(response_text, reply_path)
    )
    audio_url = f"/audio/web_reply_{ts}.wav" if wav_ok else None

    upload_path.unlink(missing_ok=True)

    if LUCI_AUTO_MEMORY:
        asyncio.create_task(asyncio.to_thread(auto_extract_memory, text, response_text))

    # Client transitions to idle when audio finishes
    return JSONResponse({
        "transcription": text,
        "response": response_text,
        "model": tag,
        "audio_url": audio_url,
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    # Auth handshake
    if WEB_SECRET:
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=15)
            msg = _parse_json(raw)
            if msg.get("type") != "auth" or msg.get("secret") != WEB_SECRET:
                await websocket.send_text('{"type":"auth_failed"}')
                await websocket.close(code=1008)
                return
        except Exception:
            await websocket.close(code=1008)
            return

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
            await broadcast_state("speaking")
            await websocket.send_text(json.dumps({
                "type": "response",
                "text": response,
                "model": tag,
            }))

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
