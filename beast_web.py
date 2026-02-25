#!/usr/bin/env python3
"""
BEAST Web Interface — Sub-phase B
FastAPI + WebSockets + voice upload.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEB_HOST = os.getenv("BEAST_WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("BEAST_WEB_PORT", "7860"))
WEB_SECRET = os.getenv("BEAST_WEB_SECRET", "")

# ---------------------------------------------------------------------------
# Imports from mini_beast
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from mini_beast import (  # noqa: E402
    MODEL,
    RUNS_DIR,
    MEMORY_PATH,
    PIPER_BIN,
    BEAST_AUTO_MEMORY,
    auto_extract_memory,
    ollama_chat,
    route_model,
    format_model_tag,
    load_persona,
    tts_to_file,
    stt_transcribe,
)

# ---------------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse

app = FastAPI(title="BEAST Web Interface")

# ---------------------------------------------------------------------------
# Chat UI HTML
# ---------------------------------------------------------------------------
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BEAST — Personal AI Agent</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0d0d0d;
    --surface:   #111116;
    --surface2:  #1a1a2a;
    --user-bg:   #1e1e2e;
    --accent:    #7c3aed;
    --accent-h:  #9d5cf6;
    --text:      #e2e2f0;
    --muted:     #6b6b8a;
    --border:    #2a2a3a;
    --danger:    #dc2626;
  }

  html, body { height: 100%; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 15px; }

  /* ---- layout ---- */
  #app { display: flex; flex-direction: column; height: 100vh; }

  #header {
    padding: 14px 20px; background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 12px; flex-shrink: 0;
  }
  #header .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--accent); flex-shrink: 0;
  }
  #header h1 { font-size: 1.1rem; font-weight: 700; letter-spacing: .04em; }
  #header p  { font-size: .75rem; color: var(--muted); }

  #messages {
    flex: 1; overflow-y: auto; padding: 20px 16px 10px;
    display: flex; flex-direction: column; gap: 12px;
    scroll-behavior: smooth;
  }

  /* ---- bubbles ---- */
  .row { display: flex; }
  .row.user  { justify-content: flex-end; }
  .row.beast { justify-content: flex-start; }

  .bubble {
    max-width: min(72%, 540px); padding: 10px 14px;
    border-radius: 14px; line-height: 1.55; white-space: pre-wrap;
    word-break: break-word;
  }
  .row.user  .bubble { background: var(--user-bg);  border-bottom-right-radius: 4px; }
  .row.beast .bubble { background: var(--surface2); border-bottom-left-radius: 4px; }

  .tag { font-size: .7rem; color: var(--muted); margin-top: 4px;
    padding: 0 2px; display: block; }

  .typing { color: var(--muted); font-style: italic; font-size: .9em; }

  /* ---- input bar ---- */
  #inputbar {
    padding: 12px 16px; background: var(--surface);
    border-top: 1px solid var(--border); flex-shrink: 0;
    display: flex; gap: 8px; align-items: flex-end;
  }
  #msg {
    flex: 1; resize: none; border: 1px solid var(--border);
    background: var(--surface2); color: var(--text);
    border-radius: 10px; padding: 10px 14px; font-size: .95rem;
    max-height: 120px; min-height: 42px; outline: none;
    font-family: inherit; line-height: 1.4;
  }
  #msg:focus { border-color: var(--accent); }

  button {
    height: 42px; border: none; border-radius: 10px; cursor: pointer;
    font-size: .9rem; font-weight: 600; transition: background .15s, transform .1s;
    flex-shrink: 0;
  }
  button:active { transform: scale(.96); }

  #sendBtn {
    background: var(--accent); color: #fff; padding: 0 18px;
  }
  #sendBtn:hover { background: var(--accent-h); }

  #recBtn {
    background: var(--surface2); color: var(--text);
    padding: 0 14px; font-size: 1.1rem;
    border: 1px solid var(--border);
  }
  #recBtn.recording { background: var(--danger); color: #fff; border-color: var(--danger); }

  /* ---- modal ---- */
  #authModal {
    position: fixed; inset: 0; background: rgba(0,0,0,.85);
    display: flex; align-items: center; justify-content: center;
    z-index: 100;
  }
  #authBox {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 32px 28px; width: min(360px, 90vw);
    display: flex; flex-direction: column; gap: 16px;
  }
  #authBox h2 { font-size: 1.1rem; }
  #authBox p  { color: var(--muted); font-size: .85rem; }
  #authBox input {
    background: var(--surface2); border: 1px solid var(--border);
    color: var(--text); padding: 10px 14px; border-radius: 8px;
    font-size: .95rem; outline: none; width: 100%;
  }
  #authBox input:focus { border-color: var(--accent); }
  #authError { color: var(--danger); font-size: .85rem; display: none; }

  @media (max-width: 480px) {
    .bubble { max-width: 88%; }
    #sendBtn { padding: 0 12px; }
    #recBtn  { padding: 0 10px; }
  }
</style>
</head>
<body>

<!-- Auth modal (shown only when WEB_SECRET is set) -->
<div id="authModal" style="display:none">
  <div id="authBox">
    <h2>🔒 BEAST Login</h2>
    <p>Enter access password to continue.</p>
    <input type="password" id="authInput" placeholder="Password" autocomplete="current-password">
    <button id="authBtn" style="background:var(--accent);color:#fff">Connect</button>
    <span id="authError">Wrong password — try again.</span>
  </div>
</div>

<div id="app">
  <div id="header">
    <div class="dot"></div>
    <div>
      <h1>BEAST</h1>
      <p>Personal AI Agent</p>
    </div>
  </div>

  <div id="messages"></div>

  <div id="inputbar">
    <textarea id="msg" rows="1" placeholder="Message BEAST..."></textarea>
    <button id="recBtn" title="Record voice">🎤</button>
    <button id="sendBtn">Send</button>
  </div>
</div>

<script>
const SECRET = __SECRET_PLACEHOLDER__;
let ws = null;
let wsReady = false;
let mediaRecorder = null;
let audioChunks = [];
let recording = false;

// ---- DOM refs ----
const msgs    = document.getElementById("messages");
const input   = document.getElementById("msg");
const sendBtn = document.getElementById("sendBtn");
const recBtn  = document.getElementById("recBtn");
const modal   = document.getElementById("authModal");
const authInp = document.getElementById("authInput");
const authBtn = document.getElementById("authBtn");
const authErr = document.getElementById("authError");

// ---- helpers ----
function scrollBottom() {
  msgs.scrollTop = msgs.scrollHeight;
}

function addBubble(side, text, tag) {
  const row = document.createElement("div");
  row.className = "row " + side;
  const bub = document.createElement("div");
  bub.className = "bubble";
  bub.textContent = text;
  row.appendChild(bub);
  if (tag) {
    const t = document.createElement("span");
    t.className = "tag";
    t.textContent = tag;
    row.appendChild(t);
  }
  msgs.appendChild(row);
  scrollBottom();
  return bub;
}

function addTyping() {
  const row = document.createElement("div");
  row.className = "row beast";
  row.id = "typing";
  const bub = document.createElement("div");
  bub.className = "bubble typing";
  bub.textContent = "BEAST is thinking...";
  row.appendChild(bub);
  msgs.appendChild(row);
  scrollBottom();
}

function removeTyping() {
  const t = document.getElementById("typing");
  if (t) t.remove();
}

// ---- WebSocket ----
function connectWS(secret) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(proto + "://" + location.host + "/ws");

  ws.onopen = () => {
    if (secret) {
      ws.send(JSON.stringify({type: "auth", secret: secret}));
    }
  };

  ws.onmessage = (ev) => {
    try { var d = JSON.parse(ev.data); } catch { return; }
    if (d.type === "status") {
      wsReady = true;
      modal.style.display = "none";
      if (d.text) addBubble("beast", d.text, "");
    } else if (d.type === "response") {
      removeTyping();
      addBubble("beast", d.text, d.model || "");
    } else if (d.type === "auth_failed") {
      authErr.style.display = "block";
      ws.close();
    }
  };

  ws.onerror = () => {};
  ws.onclose = () => { wsReady = false; };
}

// ---- send chat ----
function sendMessage() {
  const text = input.value.trim();
  if (!text || !wsReady) return;
  addBubble("user", text, "");
  addTyping();
  ws.send(JSON.stringify({type: "chat", text: text}));
  input.value = "";
  input.style.height = "auto";
}

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 120) + "px";
});

// ---- voice recording ----
recBtn.addEventListener("click", async () => {
  if (!recording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({audio: true});
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(audioChunks, {type: "audio/webm"});
        await sendVoice(blob);
      };
      mediaRecorder.start();
      recording = true;
      recBtn.classList.add("recording");
      recBtn.title = "Stop recording";
    } catch (err) {
      addBubble("beast", "❌ Mic access denied: " + err.message, "");
    }
  } else {
    mediaRecorder.stop();
    recording = false;
    recBtn.classList.remove("recording");
    recBtn.title = "Record voice";
  }
});

async function sendVoice(blob) {
  addBubble("user", "🎤 [voice message]", "");
  addTyping();
  const form = new FormData();
  form.append("audio", blob, "voice.webm");
  const headers = {};
  if (SECRET) headers["X-BEAST-SECRET"] = SECRET;
  try {
    const resp = await fetch("/voice", {method: "POST", headers, body: form});
    const data = await resp.json();
    removeTyping();
    if (data.error) {
      addBubble("beast", "❌ " + data.error, "");
      return;
    }
    if (data.transcription) {
      // Replace the "[voice message]" bubble with actual transcript
      const lastUser = [...msgs.querySelectorAll(".row.user .bubble")].at(-1);
      if (lastUser) lastUser.textContent = "🎤 " + data.transcription;
    }
    addBubble("beast", data.response, data.model || "");
    if (data.audio_url) {
      const audio = new Audio(data.audio_url);
      audio.play().catch(() => {});
    }
  } catch (err) {
    removeTyping();
    addBubble("beast", "❌ Voice request failed: " + err.message, "");
  }
}

// ---- auth modal ----
function tryConnect(secret) {
  connectWS(secret);
}

if (SECRET) {
  modal.style.display = "flex";
  authBtn.addEventListener("click", () => {
    const pw = authInp.value.trim();
    if (!pw) return;
    authErr.style.display = "none";
    tryConnect(pw);
  });
  authInp.addEventListener("keydown", e => {
    if (e.key === "Enter") authBtn.click();
  });
} else {
  tryConnect("");
}
</script>
</body>
</html>
"""


def _build_html() -> str:
    secret_js = f'"{WEB_SECRET}"' if WEB_SECRET else '""'
    return _HTML.replace("__SECRET_PLACEHOLDER__", secret_js)


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
            request.headers.get("X-BEAST-SECRET")
            or request.query_params.get("secret", "")
        )
        if secret != WEB_SECRET:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    ts = int(time.time())
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = RUNS_DIR / f"web_voice_{ts}.wav"
    reply_path = RUNS_DIR / f"web_reply_{ts}.wav"

    # Save upload
    data = await audio.read()
    upload_path.write_bytes(data)

    # Transcribe
    loop = asyncio.get_running_loop()
    text: str = await loop.run_in_executor(None, lambda: stt_transcribe(upload_path))
    if not text:
        upload_path.unlink(missing_ok=True)
        return JSONResponse({"error": "could not transcribe"})

    # Route + respond
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

    # TTS reply
    wav_ok: bool = await loop.run_in_executor(
        None, lambda: tts_to_file(response_text, reply_path)
    )
    audio_url = f"/audio/web_reply_{ts}.wav" if wav_ok else None

    upload_path.unlink(missing_ok=True)

    if BEAST_AUTO_MEMORY:
        asyncio.create_task(asyncio.to_thread(auto_extract_memory, text, response_text))

    return JSONResponse({
        "transcription": text,
        "response": response_text,
        "model": tag,
        "audio_url": audio_url,
    })


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    # Auth handshake
    if WEB_SECRET:
        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=15)
            msg = _parse_json(raw)
            if msg.get("type") != "auth" or msg.get("secret") != WEB_SECRET:
                await ws.send_text('{"type":"auth_failed"}')
                await ws.close(code=1008)
                return
        except Exception:
            await ws.close(code=1008)
            return

    await ws.send_text('{"type":"status","text":"BEAST online"}')

    try:
        while True:
            raw = await ws.receive_text()
            msg = _parse_json(raw)
            if not msg or msg.get("type") != "chat":
                continue
            text = str(msg.get("text", "")).strip()
            if not text:
                continue

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
            import json
            await ws.send_text(json.dumps({
                "type": "response",
                "text": response,
                "model": tag,
            }))

            if BEAST_AUTO_MEMORY:
                asyncio.create_task(
                    asyncio.to_thread(auto_extract_memory, text, response)
                )

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


def _parse_json(raw: str) -> dict:
    import json
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
