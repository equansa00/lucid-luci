import { useState, useEffect, useRef } from "react";

// ─── Persistent storage via window.storage ───────────────────────────────────
const DB = {
  async getProjects() {
    try { const r = await window.storage.get("luci:projects"); return r ? JSON.parse(r.value) : []; }
    catch { return []; }
  },
  async saveProjects(projects) {
    try { await window.storage.set("luci:projects", JSON.stringify(projects)); } catch {}
  },
  async getLogs() {
    try { const r = await window.storage.get("luci:logs"); return r ? JSON.parse(r.value) : []; }
    catch { return []; }
  },
  async saveLogs(logs) {
    try { await window.storage.set("luci:logs", JSON.stringify(logs.slice(-100))); } catch {}
  }
};

// ─── Claude API ───────────────────────────────────────────────────────────────
async function callClaude(systemPrompt, userMessage) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      system: systemPrompt,
      messages: [{ role: "user", content: userMessage }]
    })
  });
  const data = await res.json();
  return data.content?.[0]?.text || "No response";
}

const UNBLOCK_SYSTEM = `You are LUCI, an execution engine for a senior developer and AI builder named Chip.
Your ONLY job is to unblock technical friction fast. No fluff, no encouragement, no lengthy explanations.

When given a friction description:
1. Diagnose the most likely cause in ONE sentence
2. Give the single most direct fix (code snippet if relevant, under 10 lines)
3. Give one alternative if the first doesn't work
4. State what to check to confirm it's resolved

Format with clear labels: DIAGNOSIS / FIX / FALLBACK / VERIFY
Be brutal and direct. Chip is technical. Treat him accordingly.`;

const CHECKPOINT_SYSTEM = `You are LUCI. Chip is logging a project checkpoint before stopping work.
Generate a precise re-entry brief — a structured snapshot that lets him pick up cold in under 60 seconds.

Format:
STATUS: one line current state
LAST ACTION: what was just completed
NEXT ACTION: the single most important next step (be specific)
OPEN QUESTIONS: up to 3 things unresolved
WATCH OUT: one risk or gotcha to remember

Be specific and technical. No filler.`;

// ─── Styles ───────────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;700&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --void: #06070a;
    --deep: #0d0f14;
    --surface: #13161d;
    --raised: #1a1e28;
    --border: #232838;
    --border2: #2e3447;
    --luci: #00e5ff;
    --luci-dim: rgba(0,229,255,0.12);
    --luci-glow: rgba(0,229,255,0.25);
    --warn: #ff6b35;
    --warn-dim: rgba(255,107,53,0.12);
    --ok: #39ff8f;
    --ok-dim: rgba(57,255,143,0.1);
    --muted: #4a5168;
    --text: #c8cedd;
    --text2: #8892a4;
    --red: #ff4455;
  }

  body {
    background: var(--void);
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* scanline overlay */
  body::after {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 2px,
      rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 4px
    );
    pointer-events: none; z-index: 9999;
  }

  .root { display: flex; flex-direction: column; min-height: 100vh; }

  /* ── Header ── */
  .hdr {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 28px; border-bottom: 1px solid var(--border);
    background: var(--deep); position: sticky; top: 0; z-index: 100;
  }
  .hdr-left { display: flex; align-items: center; gap: 16px; }
  .logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 28px; font-weight: 900; letter-spacing: 4px;
    color: var(--luci); text-shadow: 0 0 20px var(--luci-glow);
  }
  .logo span { color: var(--text2); font-weight: 300; }
  .status-pill {
    font-size: 10px; letter-spacing: 2px;
    padding: 4px 10px; border-radius: 2px;
    border: 1px solid var(--ok); color: var(--ok);
    background: var(--ok-dim); animation: blink 3s ease infinite;
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.5} }

  .hdr-right { display: flex; gap: 8px; }
  .tab {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    letter-spacing: 1px; padding: 7px 16px; border-radius: 2px;
    border: 1px solid var(--border2); background: transparent;
    color: var(--text2); cursor: pointer; transition: all 0.15s;
  }
  .tab:hover { border-color: var(--luci); color: var(--luci); }
  .tab.active {
    border-color: var(--luci); color: var(--luci);
    background: var(--luci-dim);
  }

  /* ── Layout ── */
  .main { flex: 1; display: grid; grid-template-columns: 280px 1fr; gap: 0; }

  /* ── Sidebar ── */
  .sidebar {
    background: var(--deep); border-right: 1px solid var(--border);
    padding: 20px 16px; display: flex; flex-direction: column; gap: 6px;
  }
  .sidebar-label {
    font-size: 9px; letter-spacing: 3px; color: var(--muted);
    text-transform: uppercase; padding: 8px 8px 4px;
  }
  .proj-item {
    padding: 10px 12px; border-radius: 3px; cursor: pointer;
    border: 1px solid transparent; transition: all 0.15s;
    display: flex; flex-direction: column; gap: 3px;
  }
  .proj-item:hover { background: var(--surface); border-color: var(--border2); }
  .proj-item.active { background: var(--luci-dim); border-color: var(--luci); }
  .proj-name { font-size: 12px; font-weight: 500; color: var(--text); }
  .proj-meta { font-size: 10px; color: var(--text2); display: flex; gap: 8px; }
  .proj-status {
    font-size: 9px; letter-spacing: 1px; padding: 1px 6px;
    border-radius: 1px; text-transform: uppercase;
  }
  .st-active { background: var(--ok-dim); color: var(--ok); border: 1px solid rgba(57,255,143,0.3); }
  .st-blocked { background: var(--warn-dim); color: var(--warn); border: 1px solid rgba(255,107,53,0.3); }
  .st-cold { background: rgba(74,81,104,0.2); color: var(--muted); border: 1px solid var(--border2); }

  .add-proj {
    margin-top: 8px; width: 100%; padding: 9px;
    background: transparent; border: 1px dashed var(--border2);
    color: var(--muted); font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; cursor: pointer; border-radius: 2px; transition: all 0.15s;
  }
  .add-proj:hover { border-color: var(--luci); color: var(--luci); }

  /* ── Content ── */
  .content { padding: 28px; overflow-y: auto; display: flex; flex-direction: column; gap: 24px; }

  /* ── Panel ── */
  .panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 3px; overflow: hidden;
  }
  .panel-hdr {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 18px; border-bottom: 1px solid var(--border);
    background: var(--deep);
  }
  .panel-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 13px; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; color: var(--luci);
  }
  .panel-body { padding: 20px 18px; }

  /* ── Friction form ── */
  .friction-grid { display: grid; gap: 14px; }
  .field-label { font-size: 10px; letter-spacing: 2px; color: var(--muted); margin-bottom: 6px; }
  .field-input, .field-textarea {
    width: 100%; background: var(--raised); border: 1px solid var(--border2);
    color: var(--text); font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    padding: 10px 12px; border-radius: 2px; transition: border-color 0.15s; outline: none;
    resize: none;
  }
  .field-input:focus, .field-textarea:focus { border-color: var(--luci); }
  .field-textarea { min-height: 80px; }

  .btn-fire {
    width: 100%; padding: 12px;
    background: var(--luci-dim); border: 1px solid var(--luci);
    color: var(--luci); font-family: 'Barlow Condensed', sans-serif;
    font-size: 16px; font-weight: 700; letter-spacing: 4px;
    cursor: pointer; border-radius: 2px; transition: all 0.15s;
    text-transform: uppercase;
  }
  .btn-fire:hover:not(:disabled) { background: var(--luci); color: var(--void); }
  .btn-fire:disabled { opacity: 0.4; cursor: not-allowed; }

  /* ── Response ── */
  .response-block {
    background: var(--raised); border: 1px solid var(--border2);
    border-left: 3px solid var(--luci); border-radius: 2px;
    padding: 16px; margin-top: 16px;
    font-size: 12px; line-height: 1.8; white-space: pre-wrap;
    color: var(--text);
  }
  .response-block.loading { color: var(--muted); animation: pulse 1.5s ease infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  .response-section { color: var(--luci); font-weight: 500; }

  /* ── Checkpoint ── */
  .checkpoint-form { display: grid; gap: 12px; }
  .btn-secondary {
    padding: 10px 18px; background: transparent;
    border: 1px solid var(--border2); color: var(--text2);
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    cursor: pointer; border-radius: 2px; transition: all 0.15s;
  }
  .btn-secondary:hover { border-color: var(--warn); color: var(--warn); }

  /* ── Log ── */
  .log-list { display: flex; flex-direction: column; gap: 8px; }
  .log-item {
    padding: 12px 14px; background: var(--raised);
    border: 1px solid var(--border); border-radius: 2px;
    display: grid; grid-template-columns: auto 1fr auto; gap: 12px; align-items: start;
  }
  .log-type {
    font-size: 9px; letter-spacing: 1px; padding: 2px 7px;
    border-radius: 1px; white-space: nowrap; margin-top: 2px;
  }
  .lt-friction { background: var(--warn-dim); color: var(--warn); border: 1px solid rgba(255,107,53,0.3); }
  .lt-checkpoint { background: var(--ok-dim); color: var(--ok); border: 1px solid rgba(57,255,143,0.3); }
  .log-text { font-size: 11px; color: var(--text); line-height: 1.5; }
  .log-time { font-size: 10px; color: var(--muted); white-space: nowrap; }

  /* ── Project detail ── */
  .proj-detail-hdr {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px;
  }
  .proj-detail-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 32px; font-weight: 900; letter-spacing: 2px;
  }
  .meta-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
  .meta-chip {
    font-size: 10px; padding: 4px 10px; border-radius: 2px;
    border: 1px solid var(--border2); color: var(--text2);
  }
  .reentry-brief {
    background: var(--void); border: 1px solid var(--luci);
    border-radius: 3px; padding: 18px;
    font-size: 12px; line-height: 1.9; white-space: pre-wrap;
    color: var(--text); border-left: 3px solid var(--luci);
  }
  .empty-state {
    text-align: center; padding: 60px 20px; color: var(--muted); font-size: 12px; line-height: 2;
  }
  .empty-icon { font-size: 36px; margin-bottom: 12px; opacity: 0.4; }

  /* ── New project modal ── */
  .modal-overlay {
    position: fixed; inset: 0; background: rgba(6,7,10,0.85);
    display: flex; align-items: center; justify-content: center;
    z-index: 200; backdrop-filter: blur(4px);
  }
  .modal {
    background: var(--deep); border: 1px solid var(--luci);
    border-radius: 3px; padding: 28px; width: 440px; display: grid; gap: 14px;
  }
  .modal-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 20px; font-weight: 900; letter-spacing: 3px; color: var(--luci);
  }
  .modal-btns { display: flex; gap: 10px; justify-content: flex-end; }
  .btn-cancel {
    padding: 9px 18px; background: transparent;
    border: 1px solid var(--border2); color: var(--text2);
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    cursor: pointer; border-radius: 2px;
  }
  .btn-create {
    padding: 9px 18px; background: var(--luci-dim);
    border: 1px solid var(--luci); color: var(--luci);
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    cursor: pointer; border-radius: 2px; transition: all 0.15s;
  }
  .btn-create:hover { background: var(--luci); color: var(--void); }

  @media (max-width: 768px) {
    .main { grid-template-columns: 1fr; }
    .sidebar { display: none; }
  }
`;

// ─── Helpers ──────────────────────────────────────────────────────────────────
const formatTime = (iso) => {
  const d = new Date(iso);
  return d.toLocaleString("en-US", { month:"short", day:"numeric", hour:"2-digit", minute:"2-digit" });
};

const colorResponse = (text) => {
  return text.split('\n').map((line, i) => {
    const isSection = /^(DIAGNOSIS|FIX|FALLBACK|VERIFY|STATUS|LAST ACTION|NEXT ACTION|OPEN QUESTIONS|WATCH OUT):/.test(line);
    return (
      <span key={i} className={isSection ? "response-section" : ""}>
        {line}{'\n'}
      </span>
    );
  });
};

// ─── Components ───────────────────────────────────────────────────────────────
function NewProjectModal({ onClose, onCreate }) {
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [stack, setStack] = useState("");

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal">
        <div className="modal-title">NEW PROJECT</div>
        <div>
          <div className="field-label">PROJECT NAME</div>
          <input className="field-input" value={name} onChange={e => setName(e.target.value)}
            placeholder="e.g. LUCI Trading Engine" autoFocus />
        </div>
        <div>
          <div className="field-label">WHAT IS IT</div>
          <textarea className="field-textarea" value={desc} onChange={e => setDesc(e.target.value)}
            placeholder="One sentence. What are you building and why?" rows={2} />
        </div>
        <div>
          <div className="field-label">TECH STACK</div>
          <input className="field-input" value={stack} onChange={e => setStack(e.target.value)}
            placeholder="e.g. Python, FastAPI, Docker, Postgres" />
        </div>
        <div className="modal-btns">
          <button className="btn-cancel" onClick={onClose}>CANCEL</button>
          <button className="btn-create" onClick={() => {
            if (!name.trim()) return;
            onCreate({ name: name.trim(), desc: desc.trim(), stack: stack.trim() });
            onClose();
          }}>CREATE</button>
        </div>
      </div>
    </div>
  );
}

function FrictionPanel({ project, onLog }) {
  const [what, setWhat] = useState("");
  const [expected, setExpected] = useState("");
  const [actual, setActual] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const fire = async () => {
    if (!what.trim()) return;
    setLoading(true); setResponse(null);
    const msg = `Project: ${project?.name || "Unknown"}
Stack: ${project?.stack || "Unknown"}
What I was trying to do: ${what}
${expected ? `Expected: ${expected}` : ""}
${actual ? `What happened instead: ${actual}` : ""}`;
    try {
      const r = await callClaude(UNBLOCK_SYSTEM, msg);
      setResponse(r);
      onLog({ type: "friction", project: project?.name, input: what, response: r });
    } catch (e) {
      setResponse(`Error: ${e.message}`);
    }
    setLoading(false);
  };

  return (
    <div className="panel">
      <div className="panel-hdr">
        <div className="panel-title">⚡ FRICTION INTERCEPTOR</div>
        <div style={{fontSize:10,color:"var(--muted)"}}>describe the block → get unblocked</div>
      </div>
      <div className="panel-body">
        <div className="friction-grid">
          <div>
            <div className="field-label">WHAT WERE YOU TRYING TO DO</div>
            <textarea className="field-textarea" value={what}
              onChange={e => setWhat(e.target.value)}
              placeholder="e.g. Connect Alpaca API, run the scanner, deploy the Docker container..." />
          </div>
          <div>
            <div className="field-label">WHAT DID YOU EXPECT</div>
            <input className="field-input" value={expected}
              onChange={e => setExpected(e.target.value)}
              placeholder="e.g. Server starts on port 8000" />
          </div>
          <div>
            <div className="field-label">WHAT ACTUALLY HAPPENED (error / symptom)</div>
            <textarea className="field-textarea" value={actual}
              onChange={e => setActual(e.target.value)}
              placeholder="Paste error message, describe symptom, or say 'nothing happened'" rows={3} />
          </div>
          <button className="btn-fire" onClick={fire} disabled={loading || !what.trim()}>
            {loading ? "ANALYZING..." : "UNBLOCK →"}
          </button>
        </div>
        {(response || loading) && (
          <div className={`response-block ${loading ? "loading" : ""}`}>
            {loading ? "LUCI is analyzing your friction point..." : colorResponse(response)}
          </div>
        )}
      </div>
    </div>
  );
}

function CheckpointPanel({ project, onLog, onUpdateProject }) {
  const [progress, setProgress] = useState("");
  const [nextIntent, setNextIntent] = useState("");
  const [brief, setBrief] = useState(null);
  const [loading, setLoading] = useState(false);

  const saveCheckpoint = async () => {
    if (!progress.trim()) return;
    setLoading(true); setBrief(null);
    const msg = `Project: ${project?.name || "Unknown"}
Stack: ${project?.stack || "Unknown"}
Description: ${project?.desc || ""}
What I just completed / where I am now: ${progress}
What I intend to do next session: ${nextIntent || "Not specified"}
Previous checkpoint: ${project?.lastBrief || "None — first checkpoint"}`;
    try {
      const r = await callClaude(CHECKPOINT_SYSTEM, msg);
      setBrief(r);
      onLog({ type: "checkpoint", project: project?.name, input: progress, response: r });
      onUpdateProject({ lastBrief: r, lastCheckpoint: new Date().toISOString(), status: "cold" });
    } catch (e) {
      setBrief(`Error: ${e.message}`);
    }
    setLoading(false);
  };

  return (
    <div className="panel">
      <div className="panel-hdr">
        <div className="panel-title">🔖 CHECKPOINT — SAVE STATE</div>
        <div style={{fontSize:10,color:"var(--muted)"}}>log before you stop — re-enter instantly later</div>
      </div>
      <div className="panel-body">
        <div className="checkpoint-form">
          <div>
            <div className="field-label">WHAT DID YOU JUST DO / WHERE ARE YOU NOW</div>
            <textarea className="field-textarea" value={progress}
              onChange={e => setProgress(e.target.value)}
              placeholder="e.g. Got the NOAA API working, scanner logs to DB but Telegram not sending yet..." />
          </div>
          <div>
            <div className="field-label">WHAT DO YOU PLAN TO DO NEXT SESSION</div>
            <input className="field-input" value={nextIntent}
              onChange={e => setNextIntent(e.target.value)}
              placeholder="e.g. Fix Telegram bot, then run first paper trade scan" />
          </div>
          <button className="btn-fire" onClick={saveCheckpoint} disabled={loading || !progress.trim()}>
            {loading ? "SAVING STATE..." : "SAVE CHECKPOINT →"}
          </button>
        </div>
        {(brief || loading) && (
          <div className={`response-block ${loading ? "loading" : ""}`}>
            {loading ? "LUCI is generating your re-entry brief..." : colorResponse(brief)}
          </div>
        )}
      </div>
    </div>
  );
}

function ProjectView({ project, onLog, onUpdateProject }) {
  if (!project) return (
    <div className="content">
      <div className="empty-state">
        <div className="empty-icon">◈</div>
        Select a project from the sidebar<br />or create a new one to get started.
      </div>
    </div>
  );

  return (
    <div className="content">
      <div>
        <div className="proj-detail-hdr">
          <div className="proj-detail-name">{project.name}</div>
          <div className={`proj-status ${project.status === "active" ? "st-active" : project.status === "blocked" ? "st-blocked" : "st-cold"}`}>
            {project.status || "active"}
          </div>
        </div>
        <div className="meta-row">
          {project.stack && <div className="meta-chip">stack: {project.stack}</div>}
          {project.lastCheckpoint && <div className="meta-chip">last saved: {formatTime(project.lastCheckpoint)}</div>}
          {project.desc && <div className="meta-chip" style={{color:"var(--text2)"}}>{project.desc}</div>}
        </div>

        {project.lastBrief && (
          <div className="panel" style={{marginBottom: 24}}>
            <div className="panel-hdr">
              <div className="panel-title">📡 RE-ENTRY BRIEF</div>
              <div style={{fontSize:10,color:"var(--muted)"}}>pick up exactly where you left off</div>
            </div>
            <div className="panel-body">
              <div className="reentry-brief">{colorResponse(project.lastBrief)}</div>
            </div>
          </div>
        )}
      </div>

      <FrictionPanel project={project} onLog={onLog} />
      <CheckpointPanel project={project} onLog={onLog} onUpdateProject={onUpdateProject} />
    </div>
  );
}

function LogView({ logs }) {
  return (
    <div className="content">
      <div className="panel">
        <div className="panel-hdr">
          <div className="panel-title">📋 ACTIVITY LOG</div>
          <div style={{fontSize:10,color:"var(--muted)"}}>{logs.length} entries</div>
        </div>
        <div className="panel-body">
          {logs.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">◈</div>
              No activity yet. Use the Friction or Checkpoint panels<br/>to start building your log.
            </div>
          ) : (
            <div className="log-list">
              {[...logs].reverse().map((l, i) => (
                <div className="log-item" key={i}>
                  <div className={`log-type ${l.type === "friction" ? "lt-friction" : "lt-checkpoint"}`}>
                    {l.type === "friction" ? "BLOCK" : "SAVE"}
                  </div>
                  <div className="log-text">
                    <div style={{color:"var(--luci)",fontSize:11,marginBottom:4}}>{l.project}</div>
                    <div style={{color:"var(--text2)"}}>{l.input}</div>
                    {l.response && (
                      <div style={{color:"var(--muted)",fontSize:10,marginTop:4,borderTop:"1px solid var(--border)",paddingTop:4}}>
                        {l.response.slice(0,120)}...
                      </div>
                    )}
                  </div>
                  <div className="log-time">{formatTime(l.timestamp)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [projects, setProjects] = useState([]);
  const [logs, setLogs] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [tab, setTab] = useState("projects");
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    const load = async () => {
      const p = await DB.getProjects();
      const l = await DB.getLogs();
      setProjects(p);
      setLogs(l);
      if (p.length > 0) setSelectedId(p[0].id);
    };
    load();
  }, []);

  const saveProjects = async (updated) => {
    setProjects(updated);
    await DB.saveProjects(updated);
  };

  const saveLogs = async (updated) => {
    setLogs(updated);
    await DB.saveLogs(updated);
  };

  const createProject = async ({ name, desc, stack }) => {
    const p = {
      id: Date.now().toString(),
      name, desc, stack,
      status: "active",
      createdAt: new Date().toISOString(),
      lastBrief: null,
      lastCheckpoint: null
    };
    const updated = [...projects, p];
    await saveProjects(updated);
    setSelectedId(p.id);
    setTab("projects");
  };

  const addLog = async (entry) => {
    const updated = [...logs, { ...entry, timestamp: new Date().toISOString() }];
    await saveLogs(updated);
  };

  const updateProject = async (id, changes) => {
    const updated = projects.map(p => p.id === id ? { ...p, ...changes } : p);
    await saveProjects(updated);
  };

  const selected = projects.find(p => p.id === selectedId) || null;

  return (
    <>
      <style>{css}</style>
      <div className="root">
        <header className="hdr">
          <div className="hdr-left">
            <div className="logo">LUCI<span>.exe</span></div>
            <div className="status-pill">● ONLINE</div>
          </div>
          <div className="hdr-right">
            <button className={`tab ${tab === "projects" ? "active" : ""}`}
              onClick={() => setTab("projects")}>PROJECTS</button>
            <button className={`tab ${tab === "log" ? "active" : ""}`}
              onClick={() => setTab("log")}>LOG</button>
          </div>
        </header>

        <div className="main">
          <aside className="sidebar">
            <div className="sidebar-label">ACTIVE PROJECTS</div>
            {projects.map(p => (
              <div key={p.id}
                className={`proj-item ${selectedId === p.id ? "active" : ""}`}
                onClick={() => { setSelectedId(p.id); setTab("projects"); }}>
                <div className="proj-name">{p.name}</div>
                <div className="proj-meta">
                  <span className={`proj-status ${p.status === "active" ? "st-active" : p.status === "blocked" ? "st-blocked" : "st-cold"}`}>
                    {p.status || "active"}
                  </span>
                  {p.lastCheckpoint && (
                    <span style={{fontSize:9,color:"var(--muted)"}}>
                      {formatTime(p.lastCheckpoint)}
                    </span>
                  )}
                </div>
              </div>
            ))}
            <button className="add-proj" onClick={() => setShowModal(true)}>
              + NEW PROJECT
            </button>
          </aside>

          {tab === "projects"
            ? <ProjectView
                project={selected}
                onLog={addLog}
                onUpdateProject={(changes) => selected && updateProject(selected.id, changes)}
              />
            : <LogView logs={logs} />
          }
        </div>
      </div>

      {showModal && (
        <NewProjectModal
          onClose={() => setShowModal(false)}
          onCreate={createProject}
        />
      )}
    </>
  );
}
