#!/usr/bin/env python3
"""
LUCI Learn — Web-based learning UI endpoint.
Serves the full curriculum dashboard with video embedding,
lesson teaching, quiz engine, and progress tracking.
"""

# This file contains the HTML/JS for the /learn web UI.
# It's imported by luci_web.py and adds the /learn route.

LEARN_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LUCI Learn</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0a0a0a;
  color: #e0e0e0;
  font-family: 'SF Mono', 'Fira Code', monospace;
  min-height: 100vh;
}

/* ── Layout ── */
.learn-container {
  display: grid;
  grid-template-columns: 280px 1fr;
  grid-template-rows: 60px 1fr;
  height: 100vh;
  overflow: hidden;
}

/* ── Header ── */
.learn-header {
  grid-column: 1 / -1;
  background: #111;
  border-bottom: 1px solid #222;
  display: flex;
  align-items: center;
  padding: 0 20px;
  gap: 16px;
}
.learn-header h1 {
  font-size: 16px;
  color: #D4AF37;
  letter-spacing: 2px;
}
.progress-bar-wrap {
  flex: 1;
  height: 4px;
  background: #222;
  border-radius: 2px;
  overflow: hidden;
}
.progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #D4AF37, #f0c840);
  transition: width 0.5s ease;
}
.progress-label {
  font-size: 11px;
  color: #666;
  white-space: nowrap;
}
.back-btn {
  background: none;
  border: 1px solid #333;
  color: #888;
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  font-family: inherit;
  text-decoration: none;
  display: flex;
  align-items: center;
  gap: 6px;
}
.back-btn:hover { border-color: #D4AF37; color: #D4AF37; }

/* ── Sidebar ── */
.learn-sidebar {
  background: #0d0d0d;
  border-right: 1px solid #1a1a1a;
  overflow-y: auto;
  padding: 16px 0;
}
.sidebar-section {
  padding: 8px 16px;
  font-size: 10px;
  color: #444;
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-top: 8px;
}
.phase-item {
  padding: 10px 16px;
  cursor: pointer;
  border-left: 3px solid transparent;
  transition: all 0.2s;
}
.phase-item:hover { background: #111; border-left-color: #333; }
.phase-item.active { background: #141414; border-left-color: #D4AF37; }
.phase-item.completed { opacity: 0.5; }
.phase-title {
  font-size: 12px;
  color: #ccc;
  line-height: 1.4;
}
.phase-meta {
  font-size: 10px;
  color: #555;
  margin-top: 3px;
}
.phase-item.active .phase-title { color: #D4AF37; }

/* ── Main content ── */
.learn-main {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Tab bar ── */
.tab-bar {
  display: flex;
  gap: 0;
  background: #0d0d0d;
  border-bottom: 1px solid #1a1a1a;
  padding: 0 20px;
}
.tab {
  padding: 12px 20px;
  font-size: 12px;
  color: #555;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s;
  font-family: inherit;
  background: none;
  border-top: none;
  border-left: none;
  border-right: none;
}
.tab:hover { color: #aaa; }
.tab.active { color: #D4AF37; border-bottom-color: #D4AF37; }

/* ── Content panels ── */
.content-panels {
  flex: 1;
  overflow: hidden;
  position: relative;
}
.panel {
  display: none;
  height: 100%;
  overflow-y: auto;
  padding: 24px;
}
.panel.active { display: block; }

/* ── Lesson panel ── */
.lesson-header {
  margin-bottom: 20px;
}
.lesson-phase-tag {
  font-size: 10px;
  color: #D4AF37;
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-bottom: 8px;
}
.lesson-title {
  font-size: 22px;
  color: #fff;
  line-height: 1.3;
  margin-bottom: 8px;
}
.lesson-meta {
  font-size: 12px;
  color: #555;
}

/* ── Video section ── */
.video-section {
  margin-bottom: 24px;
}
.video-section-title {
  font-size: 11px;
  color: #555;
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-bottom: 12px;
}
.video-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
}
.video-card {
  background: #111;
  border: 1px solid #1a1a1a;
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  transition: border-color 0.2s;
}
.video-card:hover { border-color: #D4AF37; }
.video-thumb {
  width: 100%;
  aspect-ratio: 16/9;
  background: #000;
  position: relative;
  overflow: hidden;
}
.video-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.video-thumb .play-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0,0,0,0.4);
  transition: background 0.2s;
}
.video-card:hover .play-overlay { background: rgba(0,0,0,0.2); }
.play-btn {
  width: 48px;
  height: 48px;
  background: #D4AF37;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
}
.video-info {
  padding: 10px 12px;
}
.video-title {
  font-size: 12px;
  color: #ccc;
  line-height: 1.4;
}
.video-source {
  font-size: 10px;
  color: #D4AF37;
  margin-top: 4px;
}

/* ── Video embed modal ── */
.video-modal {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.9);
  z-index: 1000;
  align-items: center;
  justify-content: center;
}
.video-modal.open { display: flex; }
.video-modal-inner {
  width: 90vw;
  max-width: 900px;
}
.video-modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}
.video-modal-title {
  font-size: 14px;
  color: #D4AF37;
}
.video-modal-close {
  background: none;
  border: 1px solid #333;
  color: #888;
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-family: inherit;
  font-size: 12px;
}
.video-embed {
  width: 100%;
  aspect-ratio: 16/9;
  border: none;
  border-radius: 8px;
}

/* ── Teaching area ── */
.teach-area {
  background: #0d0d0d;
  border: 1px solid #1a1a1a;
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 20px;
}
.teach-area-title {
  font-size: 11px;
  color: #D4AF37;
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-bottom: 12px;
}
.teach-content {
  font-size: 14px;
  color: #ccc;
  line-height: 1.8;
  white-space: pre-wrap;
  min-height: 80px;
}
.teach-placeholder {
  color: #333;
  font-style: italic;
}

/* ── Chat area ── */
.chat-messages {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 16px;
  max-height: 400px;
  overflow-y: auto;
}
.chat-msg {
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
}
.chat-msg.user {
  background: #1a1a2e;
  border: 1px solid #2a2a4e;
  color: #aab4e8;
  align-self: flex-end;
  max-width: 80%;
}
.chat-msg.luci {
  background: #0d1a0d;
  border: 1px solid #1a2e1a;
  color: #aae8aa;
}
.chat-input-row {
  display: flex;
  gap: 10px;
}
.chat-input {
  flex: 1;
  background: #111;
  border: 1px solid #222;
  color: #e0e0e0;
  padding: 10px 14px;
  border-radius: 8px;
  font-family: inherit;
  font-size: 13px;
  resize: none;
  min-height: 44px;
  max-height: 120px;
}
.chat-input:focus { outline: none; border-color: #D4AF37; }

/* ── Buttons ── */
.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-family: inherit;
  font-size: 13px;
  font-weight: bold;
  letter-spacing: 1px;
  transition: all 0.2s;
}
.btn-gold { background: #D4AF37; color: #000; }
.btn-gold:hover { background: #f0c840; }
.btn-outline {
  background: none;
  border: 1px solid #333;
  color: #888;
}
.btn-outline:hover { border-color: #D4AF37; color: #D4AF37; }
.btn-danger {
  background: none;
  border: 1px solid #ff3333;
  color: #ff3333;
}
.btn-danger:hover { background: #ff333322; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }

/* ── Action row ── */
.action-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 20px;
}

/* ── Quiz panel ── */
.quiz-scenario {
  background: #0d0d1a;
  border: 1px solid #1a1a3a;
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 16px;
  font-size: 14px;
  color: #aab4e8;
  line-height: 1.8;
  white-space: pre-wrap;
}
.quiz-scenario-label {
  font-size: 10px;
  color: #D4AF37;
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-bottom: 10px;
}
.quiz-result {
  padding: 16px;
  border-radius: 8px;
  margin-top: 12px;
  font-size: 14px;
  line-height: 1.7;
  white-space: pre-wrap;
}
.quiz-result.pass { background: #0d1a0d; border: 1px solid #1a4a1a; color: #aae8aa; }
.quiz-result.partial { background: #1a1a0d; border: 1px solid #3a3a1a; color: #e8e8aa; }
.quiz-result.needs-review { background: #1a0d0d; border: 1px solid #4a1a1a; color: #e8aaaa; }

/* ── Progress panel ── */
.progress-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}
.progress-card {
  background: #111;
  border: 1px solid #1a1a1a;
  border-radius: 8px;
  padding: 16px;
}
.progress-card-val {
  font-size: 28px;
  font-weight: bold;
  color: #D4AF37;
}
.progress-card-lbl {
  font-size: 11px;
  color: #555;
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.phase-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.phase-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  background: #0d0d0d;
  border-radius: 6px;
  border: 1px solid #1a1a1a;
}
.phase-row.current { border-color: #D4AF37; }
.phase-row-num {
  font-size: 11px;
  color: #555;
  width: 20px;
}
.phase-row.current .phase-row-num { color: #D4AF37; }
.phase-row-title { flex: 1; font-size: 13px; color: #ccc; }
.phase-row-status { font-size: 11px; color: #555; }

/* ── Loading spinner ── */
.spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid #333;
  border-top-color: #D4AF37;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  vertical-align: middle;
  margin-right: 8px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }
</style>
</head>
<body>
<div class="learn-container">

  <!-- Header -->
  <div class="learn-header">
    <a href="/" class="back-btn">← LUCI</a>
    <h1>📚 LEARN</h1>
    <div class="progress-bar-wrap">
      <div class="progress-bar-fill" id="progressFill" style="width:0%"></div>
    </div>
    <span class="progress-label" id="progressLabel">Loading...</span>
  </div>

  <!-- Sidebar -->
  <div class="learn-sidebar" id="sidebar">
    <div class="sidebar-section">Phases</div>
    <div id="phaseList"></div>
  </div>

  <!-- Main -->
  <div class="learn-main">
    <div class="tab-bar">
      <button class="tab active" onclick="switchTab('lesson')">📖 Lesson</button>
      <button class="tab" onclick="switchTab('videos')">🎬 Videos</button>
      <button class="tab" onclick="switchTab('quiz')">🎯 Quiz</button>
      <button class="tab" onclick="switchTab('progress')">📊 Progress</button>
    </div>

    <div class="content-panels">

      <!-- Lesson Tab -->
      <div class="panel active" id="panel-lesson">
        <div class="lesson-header">
          <div class="lesson-phase-tag" id="lessonPhaseTag">Loading...</div>
          <div class="lesson-title" id="lessonTitle">Loading curriculum...</div>
          <div class="lesson-meta" id="lessonMeta"></div>
        </div>

        <div class="action-row">
          <button class="btn btn-gold" id="teachBtn" onclick="startLesson()">
            ▶ Start Lesson
          </button>
          <button class="btn btn-outline" onclick="nextLesson()">
            Next Lesson →
          </button>
          <button class="btn btn-outline" onclick="switchTab('videos')">
            🎬 Watch Videos
          </button>
          <button class="btn btn-outline" onclick="switchTab('quiz')">
            🎯 Take Quiz
          </button>
        </div>

        <div class="teach-area">
          <div class="teach-area-title">LUCI Teaching</div>
          <div class="teach-content" id="teachContent">
            <span class="teach-placeholder">
Click "Start Lesson" to have LUCI teach this topic.
She'll explain the concept, give examples from your
actual codebase, and give you a hands-on exercise.
            </span>
          </div>
        </div>

        <div class="teach-area">
          <div class="teach-area-title">Ask LUCI</div>
          <div class="chat-messages" id="chatMessages"></div>
          <div class="chat-input-row">
            <textarea
              class="chat-input"
              id="chatInput"
              placeholder="Ask a follow-up question about this lesson..."
              rows="2"
              onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat();}"
            ></textarea>
            <button class="btn btn-gold" onclick="sendChat()">Send</button>
          </div>
        </div>
      </div>

      <!-- Videos Tab -->
      <div class="panel" id="panel-videos">
        <div class="lesson-header">
          <div class="lesson-phase-tag" id="videoPhaseTag"></div>
          <div class="lesson-title" id="videoLessonTitle">Videos for this lesson</div>
        </div>

        <div class="video-section" id="coltSection" style="display:none">
          <div class="video-section-title">🎓 Colt Steele — Recommended</div>
          <div class="video-grid" id="coltVideos"></div>
        </div>

        <div class="video-section" id="freeSection">
          <div class="video-section-title">▶ Free YouTube Videos</div>
          <div class="video-grid" id="freeVideos"></div>
        </div>

        <div class="video-section" id="refSection">
          <div class="video-section-title">📚 Reference Resources</div>
          <div id="refLinks"></div>
        </div>
      </div>

      <!-- Quiz Tab -->
      <div class="panel" id="panel-quiz">
        <div class="lesson-header">
          <div class="lesson-phase-tag" id="quizPhaseTag"></div>
          <div class="lesson-title" id="quizLessonTitle">Test Your Knowledge</div>
          <div class="lesson-meta">Real-world scenarios — no multiple choice</div>
        </div>

        <div class="action-row">
          <button class="btn btn-gold" onclick="loadQuiz()">🎲 Random Scenario</button>
          <button class="btn btn-outline" id="scenario1Btn" onclick="loadQuizScenario(0)">Scenario 1</button>
          <button class="btn btn-outline" id="scenario2Btn" onclick="loadQuizScenario(1)">Scenario 2</button>
          <button class="btn btn-outline" id="scenario3Btn" onclick="loadQuizScenario(2)">Scenario 3</button>
          <button class="btn btn-danger" onclick="startExam()">📝 Phase Exam</button>
        </div>

        <div id="quizScenarioBox" style="display:none">
          <div class="quiz-scenario">
            <div class="quiz-scenario-label">Scenario</div>
            <div id="quizScenarioText"></div>
          </div>
          <div class="chat-messages" id="quizMessages"></div>
          <div class="chat-input-row">
            <textarea
              class="chat-input"
              id="quizInput"
              placeholder="Type your answer here... Be specific. LUCI will evaluate it."
              rows="3"
              onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();submitAnswer();}"
            ></textarea>
            <button class="btn btn-gold" onclick="submitAnswer()">Submit</button>
          </div>
        </div>

        <div id="quizStats" style="margin-top:20px">
          <div class="teach-area-title" style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">YOUR QUIZ HISTORY</div>
          <div id="quizStatsContent" style="color:#555;font-size:13px;">Loading stats...</div>
        </div>
      </div>

      <!-- Progress Tab -->
      <div class="panel" id="panel-progress">
        <div class="lesson-title" style="margin-bottom:20px">Your Progress</div>
        <div class="progress-grid" id="statsGrid"></div>
        <div class="teach-area-title" style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:2px;margin:20px 0 12px;">ALL PHASES</div>
        <div class="phase-list" id="allPhases"></div>
      </div>

    </div>
  </div>
</div>

<!-- Video Modal -->
<div class="video-modal" id="videoModal" onclick="if(event.target===this)closeVideo()">
  <div class="video-modal-inner">
    <div class="video-modal-header">
      <span class="video-modal-title" id="modalTitle"></span>
      <button class="video-modal-close" onclick="closeVideo()">✕ Close</button>
    </div>
    <iframe class="video-embed" id="videoEmbed" allowfullscreen></iframe>
  </div>
</div>

<script>
'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let _curriculum = null;
let _currentLesson = null;
let _chatHistory = [];
let _quizHistory = [];
let _currentScenario = null;
let _currentScenarios = [];

// ── Video database — curated free YouTube videos per lesson ───────────────
// Format: { "p{phase}_m{module}_l{lesson}": [...videos] }
const VIDEO_DB = {

  // Phase 1 Module 1 — HTTP
  "p1_m1_l1": [
    { id: "keo0rSuTDuE", title: "How The Internet Works in 5 Minutes", source: "Aaron", colt: false },
    { id: "AlkDbnbpNyk", title: "HTTP Explained — Request & Response Cycle", source: "Fireship", colt: false },
    { id: "RsQ1tFLwldY", title: "How HTTP Requests Work", source: "Academind", colt: false },
  ],
  "p1_m1_l2": [
    { id: "guYMSP7JVTA", title: "HTTP Methods GET POST PUT DELETE Explained", source: "Traversy Media", colt: false },
    { id: "UObINRj2EGY", title: "REST API concepts and examples", source: "WebConcepts", colt: false },
  ],
  "p1_m1_l3": [
    { id: "qmpUfWN7hh8", title: "HTTP Status Codes Explained", source: "Traversy Media", colt: false },
    { id: "wJa5CTIFj7U", title: "HTTP Status Codes — Full Guide", source: "Fireship", colt: false },
  ],
  "p1_m1_l4": [
    { id: "UMc2gFSbqaU", title: "HTTP Headers Explained", source: "ByteByteGo", colt: false },
    { id: "FAnuh0_BU4c", title: "JSON Explained for Beginners", source: "Programming with Mosh", colt: false },
  ],
  "p1_m1_l5": [
    { id: "6sUbt-wD3J0", title: "REST API — What is REST?", source: "TechWorld with Nana", colt: false },
    { id: "lsMQRaeKNDk", title: "What Is A RESTful API? Explained With Examples", source: "Traversy Media", colt: false },
  ],
  "p1_m1_l6": [
    { id: "Xy4ov_GRPBM", title: "curl Tutorial — Learn How to Use curl", source: "Luke Smith", colt: false },
    { id: "7XUibDYw4mc", title: "httpie vs curl — REST Client Comparison", source: "Traversy Media", colt: false },
  ],

  // Phase 1 Module 2 — Terminal
  "p1_m2_l1": [
    { id: "yz7nYlnXLfU", title: "Linux Command Line Full Course — Beginner to Pro", source: "Traversy Media", colt: false },
    { id: "ZtqBQ68cfJc", title: "The 50 Most Popular Linux Commands", source: "freeCodeCamp", colt: false },
  ],
  "p1_m2_l2": [
    { id: "uaj8Akm9MdQ", title: "Linux stdin stdout stderr Explained", source: "Engineer Man", colt: false },
    { id: "s3ii48qYBxA", title: "Linux Pipes — Redirecting Input and Output", source: "Joe Collins", colt: false },
  ],
  "p1_m2_l3": [
    { id: "4XC_O3MQJNM", title: "Linux Environment Variables Tutorial", source: "ProgrammingKnowledge", colt: false },
    { id: "5iWhQWVXosU", title: ".env Files Explained — Node.js Environment Variables", source: "Traversy Media", colt: false },
  ],
  "p1_m2_l4": [
    { id: "jTnLkjSQIpM", title: "Linux Processes — ps aux, kill, top Explained", source: "DorianDotSlash", colt: false },
    { id: "N1vgvhiyq0E", title: "systemctl and systemd Explained", source: "TechWorld with Nana", colt: false },
  ],
  "p1_m2_l5": [
    { id: "qWKK_PNHnnA", title: "SSH Explained — Secure Shell Tutorial", source: "NetworkChuck", colt: false },
    { id: "hQWRgGGtZK8", title: "How to Setup SSH Keys", source: "Traversy Media", colt: false },
  ],
  "p1_m2_l6": [
    { id: "tK9Oc0f4EuM", title: "Bash Scripting Full Course — 3 Hours", source: "Joe Collins", colt: false },
    { id: "v-F3YLd6oMw", title: "Shell Scripting Crash Course", source: "Traversy Media", colt: false },
  ],

  // Phase 1 Module 3 — Git
  "p1_m3_l1": [
    { id: "2sjqTHE0zok", title: "Git Tutorial for Beginners — What is Git?", source: "Programming with Mosh", colt: false },
    { id: "hwP7WQkmECE", title: "Git Explained in 100 Seconds", source: "Fireship", colt: false },
  ],
  "p1_m3_l2": [
    { id: "HVsySz-h9r4", title: "Git Tutorial for Beginners — Full Course", source: "Traversy Media", colt: false },
    { id: "8JJ101D3knE", title: "Git and GitHub for Beginners — Crash Course", source: "freeCodeCamp", colt: false },
  ],
  "p1_m3_l3": [
    { id: "Uszj_k0DGsg", title: "Reading Git History Like a Pro", source: "The Coding Train", colt: false },
    { id: "sevc8HHqmC0", title: "Git Log — Advanced Usage", source: "Traversy Media", colt: false },
  ],
  "p1_m3_l4": [
    { id: "lX9hsdsAeTk", title: "Git Reset — Undo Commits Explained", source: "The Coding Train", colt: false },
    { id: "RGOj5yH7evk", title: "Git Stash Tutorial", source: "Traversy Media", colt: false },
  ],
  "p1_m3_l5": [
    { id: "QV0kVNvkMxc", title: "Git Branches Tutorial", source: "Traversy Media", colt: false },
    { id: "e2IbNHi5uEk", title: "GitHub Pull Requests In 100 Seconds", source: "Fireship", colt: false },
  ],
  "p1_m3_l6": [
    { id: "SONAFMnFQI8", title: ".gitignore Explained", source: "Traversy Media", colt: false },
    { id: "6q776BGCxRg", title: "GitHub Security — Never Push API Keys", source: "NetworkChuck", colt: false },
  ],

  // Phase 2 — Python
  "p2_m4_l1": [
    { id: "_uQrJ0TkZlc", title: "Python Tutorial for Beginners — Full Course", source: "Programming with Mosh", colt: false },
    { id: "kqtD5dpn9C8", title: "Python Data Types Explained", source: "Corey Schafer", colt: false },
  ],
  "p2_m5_l1": [
    { id: "ZDa-Z5JzLYM", title: "Python Classes and Objects — OOP Explained", source: "Corey Schafer", colt: false },
    { id: "jCzT9XFZ5bw", title: "Python Decorators — How They Work", source: "Corey Schafer", colt: false },
  ],
  "p2_m5_l4": [
    { id: "iG2e8LtGHVc", title: "Python Async Await — Asyncio Explained", source: "Traversy Media", colt: false },
    { id: "t5Bo1Je9EmE", title: "Python Asyncio Tutorial", source: "Tech With Tim", colt: false },
  ],

  // Phase 3 — HTML/CSS/JS
  "p3_m7_l1": [
    { id: "ysEN5RaKOlA", title: "HTML Tutorial — Full Course for Beginners", source: "Colt Steele / freeCodeCamp", colt: true },
    { id: "UB1O30fR-EE", title: "HTML Full Course — Build a Website Tutorial", source: "freeCodeCamp", colt: false },
  ],
  "p3_m8_l1": [
    { id: "yfoY53QXEnI", title: "CSS Crash Course For Absolute Beginners", source: "Traversy Media", colt: false },
    { id: "1PnVor36_40", title: "CSS Tutorial — Full Course for Beginners", source: "freeCodeCamp", colt: false },
  ],
  "p3_m8_l2": [
    { id: "u044iM9xsWU", title: "Flexbox in 100 Seconds", source: "Fireship", colt: false },
    { id: "3YW65K6LcIA", title: "Flexbox Crash Course 2022", source: "Traversy Media", colt: false },
  ],
  "p3_m9_l1": [
    { id: "W6NZfCO5SIk", title: "JavaScript Tutorial for Beginners — Full Course", source: "Programming with Mosh", colt: false },
    { id: "hdI2bqOjy3c", title: "JavaScript Crash Course For Beginners", source: "Traversy Media", colt: false },
  ],
  "p3_m9_l3": [
    { id: "vn3tm0quoqE", title: "Async Await JavaScript Tutorial", source: "Fireship", colt: false },
    { id: "PoRJizFvM7s", title: "JavaScript Promises vs Async Await", source: "Traversy Media", colt: false },
  ],

  // Phase 5 — SQL
  "p5_m11_l1": [
    { id: "HXV3zeQKqGY", title: "SQL Tutorial — Full Database Course for Beginners", source: "freeCodeCamp", colt: false },
    { id: "7S_tz1z_5bA", title: "MySQL Tutorial for Beginners", source: "Programming with Mosh", colt: false },
  ],

  // Phase 7 — React
  "p7_m15_l1": [
    { id: "bMknfKXIFA8", title: "React in 100 Seconds", source: "Fireship", colt: false },
    { id: "w7ejDZ8SWv8", title: "React Tutorial For Beginners", source: "Programming with Mosh", colt: false },
  ],

  // Phase 8 — AI
  "p8_m18_l1": [
    { id: "LPZh9BOjkQs", title: "LLMs Explained — Large Language Models", source: "Andrej Karpathy", colt: false },
    { id: "zjkBMFhNj_g", title: "Intro to Large Language Models", source: "Andrej Karpathy", colt: false },
  ],
  "p8_m18_l2": [
    { id: "T9aRN5JkmL8", title: "Prompt Engineering Guide — Full Tutorial", source: "freeCodeCamp", colt: false },
    { id: "jC4v5AS4ocY", title: "Prompt Engineering for Developers", source: "deeplearning.ai", colt: false },
  ],
};

// Reference resources per lesson
const REFS = {
  "p1_m1_l1": [
    { title: "MDN: HTTP Overview", url: "https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview" },
    { title: "How HTTPS Works (comic)", url: "https://howhttps.works" },
  ],
  "p1_m1_l2": [
    { title: "MDN: HTTP Methods", url: "https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods" },
    { title: "RESTful API Design Guide", url: "https://restfulapi.net" },
  ],
  "p1_m2_l1": [
    { title: "Linux Command Cheatsheet", url: "https://cheatography.com/davechild/cheat-sheets/linux-command-line/" },
    { title: "The Art of Command Line", url: "https://github.com/jlevy/the-art-of-command-line" },
  ],
  "p1_m3_l1": [
    { title: "Git Official Docs", url: "https://git-scm.com/doc" },
    { title: "Oh Shit Git (common mistakes)", url: "https://ohshitgit.com" },
  ],
  "p8_m18_l1": [
    { title: "Anthropic Prompt Engineering Guide", url: "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview" },
    { title: "Ollama API Docs", url: "https://github.com/ollama/ollama/blob/main/docs/api.md" },
  ],
};

// Colt Steele Udemy link per phase
const COLT_LINKS = {
  3: { title: "The Web Developer Bootcamp 2026 — Colt Steele on Udemy", url: "https://www.udemy.com/course/the-web-developer-bootcamp/" },
  7: { title: "The Web Developer Bootcamp 2026 — React Section", url: "https://www.udemy.com/course/the-web-developer-bootcamp/" },
};

// ── Helpers ────────────────────────────────────────────────────────────────
function lessonKey(lesson) {
  return `p${lesson.phase}_m${lesson.module}_l${lesson.lesson}`;
}

function phaseKey(phase) {
  return `p${phase}_m`;
}

async function apiGet(url) {
  const r = await fetch(url);
  return r.json();
}

async function apiPost(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return r.json();
}

function escHtml(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function setLoading(btn, loading) {
  if (loading) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>' + btn.innerHTML;
  } else {
    btn.disabled = false;
    btn.innerHTML = btn.innerHTML.replace(/<span[^>]*><\/span>/, '');
  }
}

// ── Tab switching ──────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    const names = ['lesson','videos','quiz','progress'];
    t.classList.toggle('active', names[i] === name);
  });
  document.querySelectorAll('.panel').forEach(p => {
    p.classList.toggle('active', p.id === 'panel-' + name);
  });
  if (name === 'videos') loadVideos();
  if (name === 'quiz')   loadQuizStats();
  if (name === 'progress') loadProgress();
}

// ── Load curriculum ────────────────────────────────────────────────────────
async function loadCurriculum() {
  const data = await apiGet('/learn/curriculum');
  _curriculum = data;
  _currentLesson = data.current_lesson;

  // Progress bar
  const pct = data.progress_pct || 0;
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressLabel').textContent =
    data.completed + '/' + data.total_lessons + ' lessons (' + pct + '%)';

  // Lesson header
  if (_currentLesson) {
    const l = _currentLesson;
    document.getElementById('lessonPhaseTag').textContent =
      'Phase ' + l.phase + ': ' + l.phase_title + ' · Module ' + l.module;
    document.getElementById('lessonTitle').textContent = l.lesson_title;
    document.getElementById('lessonMeta').textContent =
      'Lesson ' + l.lesson + ' of ' + l.total_lessons + ' in this module';

    document.getElementById('videoPhaseTag').textContent =
      'Phase ' + l.phase + ': ' + l.phase_title;
    document.getElementById('videoLessonTitle').textContent = l.lesson_title;
    document.getElementById('quizPhaseTag').textContent =
      'Phase ' + l.phase + ': ' + l.phase_title;
    document.getElementById('quizLessonTitle').textContent =
      'Quiz: ' + l.lesson_title;
  }

  // Sidebar
  buildSidebar(data);
}

function buildSidebar(data) {
  const list = document.getElementById('phaseList');
  list.innerHTML = '';
  const phases = data.phases || [];
  phases.forEach(p => {
    const isCurrent = p.phase === (data.current_lesson?.phase || 0);
    const div = document.createElement('div');
    div.className = 'phase-item' + (isCurrent ? ' active' : '');
    div.innerHTML = `
      <div class="phase-title">${p.phase}. ${p.title}</div>
      <div class="phase-meta">${p.duration} · ${p.module_count} modules</div>
    `;
    list.appendChild(div);
  });
}

// ── Start lesson ───────────────────────────────────────────────────────────
async function startLesson() {
  const btn = document.getElementById('teachBtn');
  btn.disabled = true;
  btn.textContent = '⏳ LUCI is teaching...';

  const content = document.getElementById('teachContent');
  content.innerHTML = '<span class="spinner"></span> LUCI is preparing your lesson...';

  try {
    const data = await apiPost('/learn/teach', {});
    content.textContent = data.response || 'No response';
    _chatHistory = [{ role: 'assistant', content: data.response }];
  } catch(e) {
    content.textContent = 'Error: ' + e.message;
  }

  btn.disabled = false;
  btn.textContent = '↺ Re-teach';
}

// ── Chat ───────────────────────────────────────────────────────────────────
async function sendChat() {
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';

  const msgs = document.getElementById('chatMessages');
  msgs.innerHTML += `<div class="chat-msg user">${escHtml(text)}</div>`;
  msgs.innerHTML += `<div class="chat-msg luci"><span class="spinner"></span> Thinking...</div>`;
  msgs.scrollTop = msgs.scrollHeight;

  _chatHistory.push({ role: 'user', content: text });

  try {
    const data = await apiPost('/learn/chat', {
      text,
      history: _chatHistory.slice(-6),
    });
    const luciMsgs = msgs.querySelectorAll('.chat-msg.luci');
    luciMsgs[luciMsgs.length - 1].textContent = data.response || '';
    _chatHistory.push({ role: 'assistant', content: data.response });
  } catch(e) {
    const luciMsgs = msgs.querySelectorAll('.chat-msg.luci');
    luciMsgs[luciMsgs.length - 1].textContent = 'Error: ' + e.message;
  }
  msgs.scrollTop = msgs.scrollHeight;
}

// ── Videos ─────────────────────────────────────────────────────────────────
function loadVideos() {
  if (!_currentLesson) return;
  const key = lessonKey(_currentLesson);
  const videos = VIDEO_DB[key] || [];

  const coltVids  = videos.filter(v => v.colt);
  const freeVids  = videos.filter(v => !v.colt);

  const coltSec = document.getElementById('coltSection');
  const coltGrid = document.getElementById('coltVideos');
  const freeGrid = document.getElementById('freeVideos');

  // Colt Steele videos
  if (coltVids.length) {
    coltSec.style.display = 'block';
    coltGrid.innerHTML = coltVids.map(v => videoCard(v)).join('');
  } else {
    coltSec.style.display = 'none';
  }

  // Add Udemy link for relevant phases
  const phase = _currentLesson.phase;
  if (COLT_LINKS[phase]) {
    const link = COLT_LINKS[phase];
    coltSec.style.display = 'block';
    coltGrid.innerHTML = `
      <div style="background:#111;border:1px solid #D4AF3766;border-radius:8px;padding:16px;grid-column:1/-1">
        <div style="color:#D4AF37;font-size:12px;font-weight:bold;margin-bottom:8px">🎓 COLT STEELE — UDEMY</div>
        <div style="color:#ccc;font-size:13px;margin-bottom:12px">${link.title}</div>
        <a href="${link.url}" target="_blank" style="background:#D4AF37;color:#000;padding:8px 16px;border-radius:6px;text-decoration:none;font-size:12px;font-weight:bold">Open on Udemy →</a>
        <div style="color:#555;font-size:11px;margin-top:8px">Paid course (~$15 on sale) — best structured curriculum for this phase</div>
      </div>
      ${coltVids.map(v => videoCard(v)).join('')}
    `;
  }

  // Free videos
  if (freeVids.length) {
    freeGrid.innerHTML = freeVids.map(v => videoCard(v)).join('');
  } else {
    freeGrid.innerHTML = `
      <div style="color:#555;font-size:13px;padding:20px">
        No curated videos for this specific lesson yet.<br>
        Ask LUCI on the Lesson tab and she'll teach you directly.
      </div>
    `;
  }

  // Reference links
  const refs = REFS[key] || [];
  const refEl = document.getElementById('refLinks');
  if (refs.length) {
    refEl.innerHTML = refs.map(r => `
      <a href="${r.url}" target="_blank" style="display:block;padding:10px 14px;background:#0d0d0d;border:1px solid #1a1a1a;border-radius:6px;margin-bottom:8px;color:#88ccff;font-size:13px;text-decoration:none;">
        📄 ${r.title} →
      </a>
    `).join('');
  } else {
    refEl.innerHTML = '';
  }
}

function videoCard(v) {
  return `
    <div class="video-card" onclick="openVideo('${v.id}','${escHtml(v.title)}')">
      <div class="video-thumb">
        <img src="https://img.youtube.com/vi/${v.id}/mqdefault.jpg" alt="">
        <div class="play-overlay"><div class="play-btn">▶</div></div>
      </div>
      <div class="video-info">
        <div class="video-title">${escHtml(v.title)}</div>
        <div class="video-source">${escHtml(v.source)}</div>
      </div>
    </div>
  `;
}

function openVideo(id, title) {
  document.getElementById('modalTitle').textContent = title;
  document.getElementById('videoEmbed').src =
    'https://www.youtube.com/embed/' + id + '?autoplay=1';
  document.getElementById('videoModal').classList.add('open');
}

function closeVideo() {
  document.getElementById('videoModal').classList.remove('open');
  document.getElementById('videoEmbed').src = '';
}

// ── Quiz ───────────────────────────────────────────────────────────────────
async function loadQuizStats() {
  const data = await apiGet('/learn/quiz/stats');
  const el = document.getElementById('quizStatsContent');
  if (!data.total) {
    el.innerHTML = 'No quizzes taken yet. Click "Random Scenario" to start.';
    return;
  }
  const pct = Math.round(data.pass / data.total * 100);
  el.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">
      <div style="background:#111;border-radius:6px;padding:12px;text-align:center">
        <div style="font-size:20px;font-weight:bold;color:#D4AF37">${data.total}</div>
        <div style="font-size:10px;color:#555;margin-top:4px">TOTAL</div>
      </div>
      <div style="background:#111;border-radius:6px;padding:12px;text-align:center">
        <div style="font-size:20px;font-weight:bold;color:#00ff88">${data.pass}</div>
        <div style="font-size:10px;color:#555;margin-top:4px">PASS</div>
      </div>
      <div style="background:#111;border-radius:6px;padding:12px;text-align:center">
        <div style="font-size:20px;font-weight:bold;color:#ffd700">${data.partial}</div>
        <div style="font-size:10px;color:#555;margin-top:4px">PARTIAL</div>
      </div>
      <div style="background:#111;border-radius:6px;padding:12px;text-align:center">
        <div style="font-size:20px;font-weight:bold;color:#ff3333">${data.needs_review}</div>
        <div style="font-size:10px;color:#555;margin-top:4px">REVIEW</div>
      </div>
    </div>
    <div style="margin-top:10px;font-size:12px;color:#555">Pass rate: ${pct}%</div>
  `;
}

async function loadQuiz() {
  await loadQuizScenario(Math.floor(Math.random() * 3));
}

async function loadQuizScenario(idx) {
  const data = await apiGet('/learn/quiz/scenarios');
  _currentScenarios = data.scenarios || [];
  if (!_currentScenarios.length) {
    document.getElementById('quizScenarioBox').style.display = 'block';
    document.getElementById('quizScenarioText').textContent =
      'No scenarios yet for this lesson. Ask LUCI to quiz you on the Lesson tab.';
    return;
  }
  const scenario = _currentScenarios[idx % _currentScenarios.length];
  _currentScenario = scenario;
  document.getElementById('quizScenarioBox').style.display = 'block';
  document.getElementById('quizScenarioText').textContent = scenario;
  document.getElementById('quizMessages').innerHTML = '';
  document.getElementById('quizInput').value = '';
}

async function submitAnswer() {
  const input = document.getElementById('quizInput');
  const answer = input.value.trim();
  if (!answer || !_currentScenario) return;
  input.value = '';

  const msgs = document.getElementById('quizMessages');
  msgs.innerHTML += `<div class="chat-msg user">${escHtml(answer)}</div>`;
  msgs.innerHTML += `<div class="chat-msg luci"><span class="spinner"></span> Evaluating...</div>`;
  msgs.scrollTop = msgs.scrollHeight;

  try {
    const data = await apiPost('/learn/quiz/evaluate', {
      scenario: _currentScenario,
      answer,
    });
    const luciMsgs = msgs.querySelectorAll('.chat-msg.luci');
    const last = luciMsgs[luciMsgs.length - 1];
    last.textContent = data.response || '';
    const resultClass = (data.result || '').toLowerCase().replace(' ', '-');
    if (resultClass) last.className = 'quiz-result ' + resultClass;
  } catch(e) {
    const luciMsgs = msgs.querySelectorAll('.chat-msg.luci');
    luciMsgs[luciMsgs.length - 1].textContent = 'Error: ' + e.message;
  }
  msgs.scrollTop = msgs.scrollHeight;
  loadQuizStats();
}

async function startExam() {
  switchTab('quiz');
  document.getElementById('quizScenarioBox').style.display = 'block';
  document.getElementById('quizScenarioText').textContent =
    '📝 Phase exam loading...';
  document.getElementById('quizMessages').innerHTML = '';

  const msgs = document.getElementById('quizMessages');
  msgs.innerHTML += `<div class="chat-msg luci"><span class="spinner"></span> Preparing your phase exam...</div>`;

  const data = await apiPost('/learn/exam', {});
  const luciMsgs = msgs.querySelectorAll('.chat-msg.luci');
  luciMsgs[luciMsgs.length - 1].textContent = data.response || '';
  document.getElementById('quizScenarioText').textContent =
    'Phase Exam — answer each question, then submit for grading';
}

async function nextLesson() {
  if (!confirm('Mark this lesson complete and advance to the next?')) return;
  const data = await apiPost('/learn/next', {});
  await loadCurriculum();
  document.getElementById('teachContent').innerHTML =
    '<span class="teach-placeholder">Lesson advanced! Click "Start Lesson" to begin.</span>';
  _chatHistory = [];
  switchTab('lesson');
}

// ── Progress ───────────────────────────────────────────────────────────────
async function loadProgress() {
  const data = await apiGet('/learn/curriculum');
  const stats = document.getElementById('statsGrid');
  const pct = data.progress_pct || 0;
  stats.innerHTML = `
    <div class="progress-card">
      <div class="progress-card-val">${data.completed || 0}</div>
      <div class="progress-card-lbl">Lessons Done</div>
    </div>
    <div class="progress-card">
      <div class="progress-card-val">${data.total_lessons || 0}</div>
      <div class="progress-card-lbl">Total Lessons</div>
    </div>
    <div class="progress-card">
      <div class="progress-card-val">${pct}%</div>
      <div class="progress-card-lbl">Complete</div>
    </div>
    <div class="progress-card">
      <div class="progress-card-val">${data.phases?.length || 10}</div>
      <div class="progress-card-lbl">Phases</div>
    </div>
  `;

  const allPhases = document.getElementById('allPhases');
  allPhases.innerHTML = (data.phases || []).map(p => {
    const isCurrent = p.phase === (data.current_lesson?.phase || 0);
    return `
      <div class="phase-row ${isCurrent ? 'current' : ''}">
        <div class="phase-row-num">${p.phase}</div>
        <div class="phase-row-title">${p.title}</div>
        <div class="phase-row-status">${p.duration}${isCurrent ? ' · IN PROGRESS' : ''}</div>
      </div>
    `;
  }).join('');
}

// ── Init ───────────────────────────────────────────────────────────────────
loadCurriculum();
</script>
</body>
</html>
"""
