#!/usr/bin/env python3
"""
LUCI Quiz Engine — Scenario-based testing for each lesson.
Tests understanding through real-world situations, not trivia.
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path

WORKSPACE  = Path(os.path.expanduser("~/beast/workspace"))
CURRICULUM = WORKSPACE / "luci_curriculum.json"
QUIZ_LOG   = WORKSPACE / "runs" / "quiz_results.json"

# ── Scenario bank — real-world situations for each lesson ─────────────────
# Each entry: phase, module, lesson → list of scenario prompts
SCENARIOS = {

    # Phase 1 — Foundation
    "p1_m1_l1": [  # HTTP request/response cycle
        "You open LUCI's web UI at http://localhost:7860 and type a message. Walk me through exactly what happens — from the moment you hit Enter to when you see the response. What travels across the network, in what order, and what does each piece do?",
        "You're debugging LUCI and notice the /chat endpoint isn't responding. Your teammate says 'the request isn't reaching the server.' What does that mean technically? What would you check first and why?",
        "Explain the HTTP request/response cycle to someone who has never coded before, using the analogy of ordering food at a restaurant. Then tell me where LUCI fits into that analogy.",
    ],
    "p1_m1_l2": [  # HTTP methods
        "The LUCI web UI uses fetch('/chat', {method: 'POST'}). Why POST and not GET? What would break if you changed it to GET? Give me a concrete reason based on what /chat actually does.",
        "You're building the HHA pre-bill validation API. You need endpoints to: get a list of denied claims, submit a corrected claim, update an authorization record, and delete a test entry. Which HTTP method do you use for each and why?",
        "Someone on your team says 'just use GET for everything, it's simpler.' What's wrong with that? Give me two real problems that would cause in the HHA system.",
    ],
    "p1_m1_l3": [  # Status codes
        "You call the Polymarket API and get back a 401. What does that mean? What are three possible causes and how would you fix each one?",
        "LUCI's /audit endpoint returns 500 when you refresh the dashboard. Walk me through your debugging process. What does 500 tell you, what doesn't it tell you, and what's your first move?",
        "You're building the HHA API. A billing coordinator submits a claim correction but the patient ID doesn't exist in the system. What status code should your API return and what should the response body contain?",
    ],
    "p1_m1_l4": [  # Headers and JSON
        "Open LUCI's luci_web.py and find the /chat endpoint. What headers does it expect? What would happen if the Content-Type header was missing from the request?",
        "You receive this response from the HHAeXchange API: the body is valid JSON but the data looks wrong. What headers would you examine first and why? What information do headers carry that the body doesn't?",
        "Write out what a complete HTTP POST request to LUCI's /chat endpoint looks like — the request line, headers, and body. Don't look it up. Just reason through what must be there.",
    ],
    "p1_m1_l5": [  # REST
        "LUCI has these endpoints: /chat, /memory/list, /memory/delete, /audit, /friction. Are these RESTful? What's missing to make them fully REST compliant? Does it matter for a local personal assistant?",
        "You're designing the HHA validation API. A junior dev suggests one endpoint: /hha?action=validate&patient=123&type=prebill. What's wrong with this design? How would you redesign it using REST principles?",
        "REST uses nouns for endpoints and HTTP methods as verbs. Apply that principle to design 5 endpoints for the HHA denial tracking system.",
    ],
    "p1_m1_l6": [  # Hands-on: curl
        "Using curl, hit LUCI's /health endpoint at localhost:7860. What command do you run? What do you expect to get back? Now hit /audit — what changes in the command?",
        "You need to test the /chat endpoint with curl, sending a POST request with a JSON body containing 'text': 'hello'. Write the exact curl command. What flag adds the JSON body? What flag sets Content-Type?",
        "LUCI's /memory/delete requires a POST with a JSON body containing a key. Write the curl command to delete a memory entry with key 'test_key'. Then explain what each part of the command does.",
    ],

    # Phase 1 Module 2 — Terminal
    "p1_m2_l1": [  # Navigation
        "You're SSH'd into lucid-luci and need to find all Python files in the LUCI workspace modified in the last 24 hours. What command do you run? Break down each flag.",
        "LUCI's logs are filling up disk space. You need to find all .log files over 100MB in /home/equansa00. Write the command. Now write the command to delete them after confirming they're safe to remove.",
        "You notice a file in the workspace has wrong permissions — it's not executable but it should be. How do you check permissions? How do you fix them? What does chmod 755 actually mean?",
    ],
    "p1_m2_l2": [  # stdin/stdout/stderr
        "You run: python3 luci_audit.py 2>/dev/null | grep HEALTH. Explain every part of this command. What goes to /dev/null? What gets piped to grep? Why would you want this?",
        "LUCI's systemd service is failing silently. How do you see only the error output, separate from normal output? Write the command. Now write a command to save errors to a file while still seeing them on screen.",
        "You want to run luci_audit.py every hour and append results to a log file. Write the command using redirection. What's the difference between > and >>?",
    ],
    "p1_m2_l3": [  # Environment variables
        "LUCI reads OLLAMA_MODEL from the environment. If it's not set, it defaults to llama3.1:70b. Show me three different ways to set this variable — temporarily for one command, for the current session, and permanently.",
        "You're deploying LUCI to lucid-luci and need to move the .env file securely. What are two things you should never do with .env files? How does LUCI's .gitignore handle this?",
        "Explain why LUCI stores API keys in .env instead of hardcoding them in luci.py. Give me a concrete scenario where hardcoding would cause a real problem.",
    ],
    "p1_m2_l4": [  # Processes
        "You run ps aux | grep python and see three LUCI processes. Which one is the main bot? Which is the web server? How do you tell them apart from the output?",
        "LUCI's Telegram bot is hanging — it's not responding but the process is still running. Walk me through how you'd diagnose and restart it using systemctl. What commands, in what order, and what does each tell you?",
        "Explain what happens when you run systemctl --user restart luci.service. What signal does systemd send? What happens to the old process? When does the new one start?",
    ],
    "p1_m2_l5": [  # SSH
        "You've just installed Ubuntu on lucid-luci. Walk me through setting up passwordless SSH access from your Framework laptop. What files do you create, where do they go, and what permissions do they need?",
        "You're SSH'd into lucid-luci and want to run LUCI's web server but keep it running after you disconnect. What are two ways to do this? What's the difference between nohup, screen, and a systemd service?",
        "Your SSH connection to lucid-luci keeps dropping after 60 seconds of inactivity. How do you fix this? Where does the config change go — client side or server side?",
    ],
    "p1_m2_l6": [  # Bash script hands-on
        "Write a bash script that checks if luci.service is active, prints its status, and sends you a Telegram message if it's down. What's the first line of any bash script? How do you check a command's exit code?",
        "Your script needs to run LUCI's audit and only send a Telegram alert if the health status is not HEALTHY. How do you capture command output into a variable? How do you do string comparison in bash?",
        "You want your health-check script to run every 5 minutes. How do you set that up with cron? What does '*/5 * * * *' mean? How is cron different from a systemd timer?",
    ],

    # Phase 1 Module 3 — Git
    "p1_m3_l1": [  # Mental model
        "You made three commits to luci.py today. Explain what git is actually storing — is it storing the full file each time, or just the changes? Why does this matter for a 172KB file like luci.py?",
        "Your colleague says 'git is just a backup tool.' What's wrong with that mental model? Give me two things git does that a simple backup doesn't.",
        "Explain the difference between your working directory, the staging area, and a commit. Use LUCI's workspace as the example.",
    ],
    "p1_m3_l2": [  # Core commands
        "You just added the curriculum system to LUCI (luci_curriculum.json, luci_teacher.py, luci_quiz.py). Walk me through the exact git commands to stage and commit these files with a meaningful commit message.",
        "You committed luci.py but forgot to include luci_teacher.py in the same commit. What are your options? Which is safest if you haven't pushed yet?",
        "What does git pull actually do? Break it into its two component operations. When would you use git fetch instead and why?",
    ],
    "p1_m3_l3": [  # Reading history
        "Run git log --oneline on the LUCI repo. What does each line tell you? How do you see what changed in a specific commit? How do you see who changed a specific line in luci.py?",
        "You introduced a bug in luci.py sometime in the last week. How do you find which commit caused it without reading every commit manually?",
        "Your teammate pushed a commit that broke the /audit endpoint. Before reverting, you need to understand exactly what changed. What git command shows you a diff of that commit?",
    ],
    "p1_m3_l4": [  # Fixing mistakes
        "You accidentally committed your .env file with API keys. It's not pushed yet. What's the safest way to remove it from the commit history? What do you do differently if it was already pushed?",
        "You've been working on a new LUCI feature for 2 hours and realize you started from the wrong branch. You haven't committed yet. How do you move your work to the correct branch without losing it?",
        "You want to undo the last commit but keep your changes in the working directory. What command do you use? What's the difference between git reset --soft, --mixed, and --hard?",
    ],
    "p1_m3_l5": [  # Branch workflows
        "You're about to add the HHA validation module to LUCI. Walk me through the branch workflow — from creating the branch to getting it merged. What commands, in what order?",
        "You're working on a feature branch and main has moved ahead with 3 new commits. How do you get those changes into your branch? What's the difference between merge and rebase here?",
        "Describe a scenario where a merge conflict would happen in the LUCI codebase. How do you resolve it? What does a conflict marker look like in the file?",
    ],
    "p1_m3_l6": [  # Hands-on gitignore
        "LUCI's workspace has .env, __pycache__, *.pyc, beast_memory.json, and runs/ that should never be committed. Write the .gitignore file. How do you make git forget a file that was already tracked?",
        "You added gmail_token.json to .gitignore but git still tracks it. Why? What's the command to stop tracking it without deleting the file?",
        "Design a branching strategy for LUCI development — one person working on features, bug fixes, and production deployments. What branches do you need and what are the rules for each?",
    ],
}

# ── Quiz engine ─────────────────────────────────────────────────────────────
def get_scenarios_for_lesson(phase: int, module: int, lesson: int) -> list:
    key = f"p{phase}_m{module}_l{lesson}"
    return SCENARIOS.get(key, [])


def get_random_scenario(phase: int, module: int, lesson: int) -> str | None:
    scenarios = get_scenarios_for_lesson(phase, module, lesson)
    if not scenarios:
        return None
    return random.choice(scenarios)


def get_all_scenarios(phase: int, module: int, lesson: int) -> list:
    return get_scenarios_for_lesson(phase, module, lesson)


def build_quiz_prompt(scenario: str, lesson_title: str, phase_title: str) -> str:
    return f"""You are LUCI acting as Chip's coding coach and examiner.

CONTEXT:
- Phase: {phase_title}
- Lesson: {lesson_title}

SCENARIO TO PRESENT:
{scenario}

YOUR ROLE:
1. Present the scenario clearly
2. Wait for Chip's answer
3. Evaluate his response:
   - What he got right (be specific)
   - What he missed or got wrong (be direct, not harsh)
   - The complete correct answer
   - One follow-up question that goes one level deeper
4. Score him: PASS / PARTIAL / NEEDS REVIEW
   - PASS: Got the core concept right
   - PARTIAL: Right direction but missing key details
   - NEEDS REVIEW: Fundamental misunderstanding

EVALUATION STYLE:
- Be direct: "That's right because..." or "That's wrong — here's why..."
- Don't pad with praise. If he's right, confirm it and push deeper.
- If he's wrong, explain clearly without making him feel bad.
- Connect everything back to LUCI's real code or the HHA project.
- End every evaluation with one question that tests the next level of understanding.

Present the scenario now and ask for his answer."""


def build_exam_prompt(phase: int, phase_title: str, lessons_completed: list) -> str:
    """Build a comprehensive phase exam prompt."""
    return f"""You are LUCI acting as Chip's exam proctor for Phase {phase}: {phase_title}.

This is a comprehensive exam covering all lessons in this phase.
Lessons completed: {', '.join(lessons_completed)}

EXAM FORMAT:
1. Present 3 scenario-based questions (one easy, one medium, one hard)
2. Each scenario should require applying multiple concepts from this phase together
3. After all answers, give a final score and specific feedback on each

EXAM RULES:
- No multiple choice — all questions require explanation
- Each scenario should be something that could happen while building LUCI or the HHA system
- Grade strictly but fairly
- At the end: PASS (>70%), NEEDS MORE PRACTICE (<70%)

Begin the exam. Present question 1 of 3."""


def log_quiz_result(phase: int, module: int, lesson: int, scenario: str,
                    result: str, notes: str = "") -> None:
    QUIZ_LOG.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if QUIZ_LOG.exists():
        try:
            results = json.loads(QUIZ_LOG.read_text())
        except Exception:
            results = []

    results.append({
        "date":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "phase":    phase,
        "module":   module,
        "lesson":   lesson,
        "scenario": scenario[:100],
        "result":   result,
        "notes":    notes,
    })
    QUIZ_LOG.write_text(json.dumps(results, indent=2))


def get_quiz_stats() -> dict:
    if not QUIZ_LOG.exists():
        return {"total": 0, "pass": 0, "partial": 0, "needs_review": 0}
    try:
        results = json.loads(QUIZ_LOG.read_text())
        return {
            "total":        len(results),
            "pass":         sum(1 for r in results if r.get("result") == "PASS"),
            "partial":      sum(1 for r in results if r.get("result") == "PARTIAL"),
            "needs_review": sum(1 for r in results if r.get("result") == "NEEDS_REVIEW"),
        }
    except Exception:
        return {"total": 0, "pass": 0, "partial": 0, "needs_review": 0}


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if cmd == "stats":
        stats = get_quiz_stats()
        print(f"Quiz results: {stats}")
    elif cmd == "scenarios":
        phase  = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        module = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        lesson = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        scenarios = get_all_scenarios(phase, module, lesson)
        print(f"Scenarios for p{phase}_m{module}_l{lesson}: {len(scenarios)}")
        for i, s in enumerate(scenarios, 1):
            print(f"\n{i}. {s}")
