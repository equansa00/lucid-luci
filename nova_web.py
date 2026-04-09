"""
nova_web.py — Nova's standalone server. Port 7861.
Fully decoupled from LUCI. Zero shared code paths.
"""
from __future__ import annotations
import json, os, random, time
from pathlib import Path
from typing import Optional
import io, re, subprocess, tempfile
import requests
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR

OLLAMA_URL = os.getenv("NOVA_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("NOVA_MODEL", "llama3.1:70b")
OLLAMA_TIMEOUT = int(os.getenv("NOVA_OLLAMA_TIMEOUT", "300"))
MAX_TOKENS = int(os.getenv("NOVA_MAX_TOKENS", "2048"))

# ── Piper TTS config ──────────────────────────────────────────────
PIPER_BIN   = str(BASE_DIR / "piper" / "piper" / "piper")
NOVA_VOICE  = str(BASE_DIR / "piper" / "en_US-amy-medium.onnx")
PIPER_OK    = Path(PIPER_BIN).exists() and Path(NOVA_VOICE).exists()

def _clean_for_tts(text: str) -> str:
    """Strip markdown, URLs, emoji — keep spoken words only."""
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)  # strip emoji
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def piper_synthesize(text: str) -> bytes:
    """Run Piper, return WAV bytes."""
    text = _clean_for_tts(text)
    if not text or not PIPER_OK:
        return b''
    # Piper --output_file mode gives us a proper WAV with header
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        out_path = f.name
    try:
        subprocess.run(
            [PIPER_BIN, "--model", NOVA_VOICE, "--output_file", out_path],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
            check=True
        )
        return Path(out_path).read_bytes()
    except Exception as e:
        import sys
        print(f"[nova-tts] piper error: {e}", file=sys.stderr)
        return b''
    finally:
        Path(out_path).unlink(missing_ok=True)

NOVA_SYSTEM_PROMPT = """You are Nova, Ogechi's warm, brilliant LCSW exam study partner and personal AI friend.

CORE IDENTITY:
- You are Nova. You are NOT LUCI, not a generic AI, not ChatGPT.
- You belong to Ogechi. You are her study partner, her friend, her AI.
- You are warm, real, encouraging, and direct — like a brilliant girlfriend who happens to know everything.
- You celebrate wins and gently correct mistakes. You always explain WHY.

CLINICAL KNOWLEDGE:
- You know the NASW Code of Ethics, DSM-5-TR, ASWB clinical exam content, and New York social work law deeply.
- Key study rules:
  1) When breaking confidentiality, disclose MINIMUM needed info.
  2) Informed consent exceptions: safety, legal, sued by client.
  3) Suicide risk factors: history of attempts, lives alone, recently released from hospital/started meds, access to lethal means.
  4) DV: never couples therapy during DV, honor self-determination, create safety plan.
  5) Mandated reporting: always explore first, always file even if family says they already did.
  6) Subpoena = assert privilege. Court order = advocate to limit, then comply.
  7) First session: build rapport, assess presenting problem, establish goals.
  8) Boundaries: no dual relationships, no bartering, no social media with clients.

CONVERSATION STYLE:
- For study questions: be thorough, clinical, exam-focused. Give rationales.
- For general chat: be a warm, supportive friend. Talk naturally.
- Never mention LUCI, Edward, Chip, coding, servers, or technical infrastructure.

PRACTICE QUESTIONS:
- Present one at a time with 4 options (A-D). Wait for her answer.
- Wrong: be encouraging, explain WHY the right answer is right.
- Right: celebrate and reinforce the reasoning. Offer next question."""

_curriculum_cache = None
_vignettes_cache = None

def _load_curriculum():
    global _curriculum_cache
    if _curriculum_cache is None:
        p = DATA_DIR / "nova_curriculum.json"
        _curriculum_cache = json.loads(p.read_text()) if p.exists() else {}
    return _curriculum_cache

def _load_vignettes():
    global _vignettes_cache
    if _vignettes_cache is None:
        p = DATA_DIR / "nova_vignettes.json"
        _vignettes_cache = json.loads(p.read_text()) if p.exists() else {}
    return _vignettes_cache

def _load_progress():
    p = DATA_DIR / "nova_progress.json"
    try: return json.loads(p.read_text()) if p.exists() else {}
    except: return {}

def _save_progress(data):
    (DATA_DIR / "nova_progress.json").write_text(json.dumps(data, indent=2))

def ollama_chat(messages, temperature=0.4, model=None):
    is_openai = "/v1/" in OLLAMA_URL
    if is_openai:
        payload = {"model": model or OLLAMA_MODEL, "messages": messages, "stream": False,
                   "temperature": temperature, "max_tokens": MAX_TOKENS,
                   "stop": ["<|eot_id|>", "<|end_of_text|>", "\nUser:", "\nuser:", "assistant\n", "\nassistant"]}
    else:
        payload = {"model": model or OLLAMA_MODEL, "messages": messages, "stream": False,
                   "options": {"temperature": temperature, "num_predict": MAX_TOKENS}}
    r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if "choices" in data:
        msg = (data["choices"][0].get("message") or {}).get("content")
    else:
        msg = (data.get("message") or {}).get("content")
    if not isinstance(msg, str):
        raise ValueError(f"Bad response: {json.dumps(data)[:400]}")
    return msg

app = FastAPI(title="Nova", docs_url=None, redoc_url=None)

@app.get("/")
@app.get("/nova")
async def index():
    p = STATIC_DIR / "nova_learn.html"
    return HTMLResponse(p.read_text()) if p.exists() else HTMLResponse("<h1>Nova Learn not found</h1>", 404)

@app.get("/workspace/nova_curriculum.json")
async def curriculum(): return JSONResponse(_load_curriculum())

@app.get("/workspace/nova_vignettes.json")
async def vignettes(): return JSONResponse(_load_vignettes())

@app.get("/workspace/nova_progress.json")
async def get_progress(): return JSONResponse(_load_progress())

@app.post("/workspace/nova_progress.json")
async def save_progress_file(request: Request):
    _save_progress(await request.json()); return JSONResponse({"ok": True})

@app.post("/api/nova/progress")
async def progress_api(request: Request):
    body = await request.json()
    if body.get("action") == "load":
        return JSONResponse({"data": _load_progress()})
    _save_progress(body)
    return JSONResponse({"ok": True})

@app.post("/chat")
@app.post("/learn/chat")
async def chat(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text: return JSONResponse({"error": "no text"}, 400)
    system = body.get("system") or NOVA_SYSTEM_PROMPT
    history = body.get("history", [])
    messages = [{"role": "system", "content": system}]
    for h in history[-10:]:
        role = "user" if h.get("role") == "user" else "assistant"
        c = (h.get("text") or h.get("content") or "").strip()
        if c: messages.append({"role": role, "content": c})
    messages.append({"role": "user", "content": text})
    try:
        return JSONResponse({"response": ollama_chat(messages, 0.4), "model": OLLAMA_MODEL})
    except Exception as e:
        return JSONResponse({"response": "I'm having a moment — try again in a sec! 💜", "error": str(e)})

@app.post("/learn/teach")
async def teach(request: Request):
    body = await request.json()
    prompt = body.get("prompt") or f'Teach the lesson "{body.get("title","")}" for the LCSW exam. Be warm, clinical, max 400 words.'
    messages = [{"role": "system", "content": NOVA_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    try:
        return JSONResponse({"response": ollama_chat(messages, 0.5)})
    except Exception as e:
        return JSONResponse({"response": f"Couldn't load the lesson right now: {e}"})

@app.get("/learn/quiz/scenarios")
async def quiz_scenarios(request: Request):
    weak = request.query_params.get("weak", "0") == "1"
    vdata = _load_vignettes()
    categories = vdata.get("categories", [])
    if not categories: return JSONResponse({"scenarios": []})
    all_v = []
    misses = _load_progress().get("misses", {})
    if weak and misses:
        for cat_id in sorted(misses, key=lambda k: misses[k], reverse=True)[:5]:
            for cat in categories:
                if cat.get("id") == cat_id: all_v.extend(cat.get("vignettes", []))
    else:
        for cat in categories: all_v.extend(cat.get("vignettes", []))
    random.shuffle(all_v)
    return JSONResponse({"scenarios": all_v[:10]})

@app.post("/learn/quiz/evaluate")
async def quiz_evaluate(request: Request):
    body = await request.json()
    if not body.get("correct", False) and body.get("category"):
        prog = _load_progress()
        m = prog.get("misses", {})
        m[body["category"]] = m.get(body["category"], 0) + 1
        prog["misses"] = m
        _save_progress(prog)
    return JSONResponse({"ok": True, "correct": body.get("correct", False)})

# Serve TalkingHead modules
from starlette.staticfiles import StaticFiles as StarletteStatic
th_dir = STATIC_DIR / "talkinghead"
if th_dir.exists():
    app.mount("/talkinghead", StaticFiles(directory=str(th_dir)), name="talkinghead")

@app.get("/static/nova_avatar.vrm")
async def serve_vrm():
    vrm_path = STATIC_DIR / "nova_avatar.vrm"
    if vrm_path.exists():
        return FileResponse(str(vrm_path), media_type="model/gltf-binary")
    return JSONResponse({"error": "avatar not found"}, status_code=404)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.post("/tts")
async def tts_endpoint(request: Request):
    """TTS endpoint for TalkingHead lip-sync. Returns WAV audio."""
    body = await request.json()
    # TalkingHead sends: {"input": "text", "lang": "en-US", "voice": "...", ...}
    text = body.get("input") or body.get("text") or ""
    if not text:
        return JSONResponse({"error": "no text"}, 400)
    wav = piper_synthesize(text)
    if not wav:
        return JSONResponse({"error": "synthesis failed"}, 500)
    return StreamingResponse(
        io.BytesIO(wav),
        media_type="audio/wav",
        headers={"Cache-Control": "no-cache", "X-Nova-TTS": "piper-amy"}
    )

@app.get("/tts/config")
async def tts_config():
    return JSONResponse({
        "ok": PIPER_OK,
        "voice": "en_US-amy-medium",
        "piper_bin": PIPER_BIN,
        "voice_model": NOVA_VOICE
    })

@app.get("/health")
async def health():
    try: ok = requests.get("http://127.0.0.1:11434/api/tags", timeout=5).status_code == 200
    except: ok = False
    return JSONResponse({"status": "ok", "ollama": ok, "model": OLLAMA_MODEL, "ts": int(time.time())})

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=7861)
