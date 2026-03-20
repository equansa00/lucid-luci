#!/usr/bin/env python3
"""
LUCI Autonomous Agent Loop
Goal → Plan → Execute → Evaluate → Retry → Done
"""

import json
import subprocess
import os
from pathlib import Path
from datetime import datetime

WORKSPACE = Path(__file__).parent
TASK_LOG  = WORKSPACE / "runs" / "agent_tasks.json"

# ── Tool definitions ──────────────────────────────────────────────────────────
TOOLS = {
    "shell": {
        "description": "Run a shell command",
        "example": {"tool": "shell", "command": "systemctl --user status luci-web"}
    },
    "file_read": {
        "description": "Read a file",
        "example": {"tool": "file_read", "path": "~/beast/workspace/luci_web.py"}
    },
    "file_write": {
        "description": "Write content to a file (backs up first)",
        "example": {"tool": "file_write", "path": "~/beast/workspace/test.py", "content": "# test"}
    },
    "file_patch": {
        "description": "Replace exact text in a file",
        "example": {"tool": "file_patch", "path": "file.py", "old": "bad code", "new": "good code"}
    },
    "service_restart": {
        "description": "Restart a systemd user service",
        "example": {"tool": "service_restart", "name": "luci-web"}
    },
    "service_status": {
        "description": "Check a systemd user service status",
        "example": {"tool": "service_status", "name": "luci-web"}
    },
    "http_get": {
        "description": "Make an HTTP GET request",
        "example": {"tool": "http_get", "url": "http://localhost:7860/audit"}
    },
    "done": {
        "description": "Mark the task complete with a result",
        "example": {"tool": "done", "result": "Fixed the error in luci_web.py and restarted the service"}
    },
    "fail": {
        "description": "Mark the task as failed with reason",
        "example": {"tool": "fail", "reason": "Could not find the error after 3 attempts"}
    }
}

# ── Safety layer ──────────────────────────────────────────────────────────────
BLOCKED_COMMANDS = [
    "rm -rf /", "mkfs", "dd if=", ":(){:|:&}", "chmod -R 777 /",
    "shutdown", "reboot", "halt", "poweroff"
]

CONFIRM_REQUIRED = [
    "rm ", "apt remove", "pip uninstall", "git reset --hard",
    "truncate", "DROP TABLE", "DELETE FROM"
]

# Tools that require confirmation before executing
CONFIRM_TOOLS = ["service_restart", "file_write", "file_patch"]

def is_safe(command: str) -> tuple[bool, str]:
    for blocked in BLOCKED_COMMANDS:
        if blocked in command:
            return False, f"Blocked: contains '{blocked}'"
    return True, "ok"

def needs_confirm(command: str) -> bool:
    return any(c in command for c in CONFIRM_REQUIRED)


# ── Tool executor ─────────────────────────────────────────────────────────────
def execute_tool(tool_call: dict, confirm_fn=None) -> dict:
    tool = tool_call.get("tool")

    # Block destructive tools when no confirm_fn provided
    if tool in globals().get("CONFIRM_TOOLS", []) and confirm_fn is None:
        # Auto-allow status checks, block mutations in unattended mode
        if tool == "service_restart":
            print(f"  ⚠️  Skipping service_restart (no confirm_fn) — use /goal from Telegram for destructive actions")
            return {"ok": False, "error": "service_restart requires confirmation"}

    if tool == "shell":
        cmd = tool_call.get("command", "")
        safe, reason = is_safe(cmd)
        if not safe:
            return {"ok": False, "error": reason}
        if needs_confirm(cmd) and confirm_fn:
            if not confirm_fn(f"Run: {cmd}"):
                return {"ok": False, "error": "User declined"}
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True,
                text=True, timeout=30,
                cwd=str(WORKSPACE)
            )
            return {
                "ok":     result.returncode == 0,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-500:],
                "code":   result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Command timed out after 30s"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif tool == "file_read":
        path = Path(tool_call.get("path", "")).expanduser()
        try:
            content = path.read_text(encoding="utf-8")
            return {"ok": True, "content": content[:5000]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif tool == "file_write":
        path = Path(tool_call.get("path", "")).expanduser()
        content = tool_call.get("content", "")
        try:
            # Backup first
            if path.exists():
                backup = path.with_suffix(path.suffix + ".bak")
                backup.write_text(path.read_text())
            path.write_text(content, encoding="utf-8")
            return {"ok": True, "written": len(content)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif tool == "file_patch":
        path    = Path(tool_call.get("path", "")).expanduser()
        old_str = tool_call.get("old", "")
        new_str = tool_call.get("new", "")
        try:
            src = path.read_text(encoding="utf-8")
            if old_str not in src:
                return {"ok": False, "error": "Old string not found in file"}
            # Backup
            path.with_suffix(path.suffix + ".bak").write_text(src)
            path.write_text(src.replace(old_str, new_str, 1))
            return {"ok": True, "patched": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif tool == "service_restart":
        name = tool_call.get("name", "")
        result = subprocess.run(
            ["systemctl", "--user", "restart", name],
            capture_output=True, text=True
        )
        return {"ok": result.returncode == 0, "output": result.stderr}

    elif tool == "service_status":
        name = tool_call.get("name", "")
        result = subprocess.run(
            ["systemctl", "--user", "is-active", name],
            capture_output=True, text=True
        )
        return {"ok": True, "status": result.stdout.strip()}

    elif tool == "http_get":
        import urllib.request
        url = tool_call.get("url", "")
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                return {"ok": True, "body": r.read().decode()[:2000], "status": r.status}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    elif tool in ("done", "fail"):
        return {"ok": True, "terminal": True, "tool": tool,
                "message": tool_call.get("result") or tool_call.get("reason", "")}

    return {"ok": False, "error": f"Unknown tool: {tool}"}


# ── Tool restriction by goal type ─────────────────────────────────────────────
def classify_goal(goal: str) -> list:
    """Return allowed tools based on goal content."""
    goal_lower = goal.lower()

    # Read-only goals — no writing, no restarts
    if any(w in goal_lower for w in [
        "check", "find", "show", "list", "count", "search",
        "what", "how many", "status", "report", "disk", "usage",
        "largest", "errors", "log", "health", "verify", "look"
    ]):
        return ["shell", "file_read", "service_status", "http_get", "done", "fail"]

    # Service management
    if any(w in goal_lower for w in [
        "restart", "start", "stop", "service", "fix service"
    ]):
        return ["service_status", "service_restart", "shell", "http_get", "done", "fail"]

    # File editing goals
    if any(w in goal_lower for w in [
        "fix", "patch", "edit", "update", "change", "modify",
        "replace", "correct", "rewrite"
    ]):
        return ["file_read", "file_patch", "file_write", "shell", "service_restart", "done", "fail"]

    # Default — all tools except destructive ones
    return ["shell", "file_read", "file_patch", "file_write",
            "service_status", "service_restart", "http_get", "done", "fail"]


def get_completion_check(goal: str, history: list) -> str:
    """Check if goal is already satisfied based on history."""
    if not history:
        return ""
    # Look for successful results that answer the goal
    goal_lower = goal.lower()
    for h in history:
        res = h.get("result", {})
        if not res.get("ok"):
            continue
        stdout = res.get("stdout", "")
        body   = res.get("body", "")
        content = stdout or body
        # If we have substantial output and goal is info-seeking
        if content and len(content) > 50 and any(w in goal_lower for w in [
            "check", "find", "show", "list", "status", "report",
            "disk", "usage", "largest", "count", "what"
        ]):
            return "\nNOTE: You already have useful output from previous steps. If it answers the goal, call done NOW with a summary."
    return ""


# ── Planner prompt ────────────────────────────────────────────────────────────
def build_planner_prompt(goal: str, history: list, tools: dict) -> str:
    tool_list = "\n".join(
        f'- {name}: {info["description"]} — e.g. {json.dumps(info["example"])}'
        for name, info in tools.items()
    )

    history_str = ""
    for h in history[-6:]:
        history_str += f"\nStep: {json.dumps(h.get('tool_call'))}\nResult: {json.dumps(h.get('result'))}\n"

    steps_done   = len(history)
    max_before_done = 4
    allowed      = classify_goal(goal)
    completion   = get_completion_check(goal, history)

    # Only show allowed tools
    allowed_tools = {k: v for k, v in tools.items() if k in allowed}
    tool_list = "\n".join(
        f'- {name}: {info["description"]} — e.g. {json.dumps(info["example"])}' 
        for name, info in allowed_tools.items()
    )

    return f"""You are LUCI's autonomous execution engine.

GOAL: {goal}

ALLOWED TOOLS FOR THIS GOAL ({len(allowed_tools)} available):
{tool_list}

EXECUTION HISTORY ({steps_done} steps so far):
{history_str or "None yet — this is the first step."}
{completion}
RULES:
- Respond with ONLY one JSON tool call object — no text, no markdown
- Only use tools from the ALLOWED list above
- If you have the answer, call "done" immediately with a clear summary
- After {max_before_done} steps, you MUST call "done"
- Never repeat a failed tool call — try a different approach
- File paths: absolute or ~/beast/workspace/

{"IMPORTANT: " + str(steps_done) + " steps taken. You MUST call done NOW with summary of findings." if steps_done >= max_before_done else ""}

Respond with exactly one JSON object:"""


# ── Main agent loop ───────────────────────────────────────────────────────────
def run_agent(goal: str, max_steps: int = 10, confirm_fn=None,
              chat_fn=None) -> dict:
    """
    Run the autonomous agent loop.
    chat_fn: function(messages) -> str  (ollama_chat or similar)
    confirm_fn: function(prompt) -> bool
    """
    history  = []
    task_id  = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n🤖 LUCI Agent starting: {goal}")
    print(f"   Max steps: {max_steps}")
    print("-" * 60)

    for step in range(1, max_steps + 1):
        print(f"\nStep {step}/{max_steps}")

        # Build prompt
        prompt = build_planner_prompt(goal, history, TOOLS)
        messages = [
            {"role": "system", "content": "You are a precise autonomous agent. Respond only with valid JSON tool calls."},
            {"role": "user",   "content": prompt}
        ]

        # Get next action from model
        try:
            if chat_fn:
                response = chat_fn(messages, temperature=0.1)
            else:
                # Fallback: print prompt for testing
                print(f"PROMPT:\n{prompt[:500]}")
                response = '{"tool": "done", "result": "Test mode"}'

            # Parse tool call
            raw = response.strip()
            # Strip markdown if model added it
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            tool_call = json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"  ❌ Invalid JSON from model: {e}")
            tool_call = {"tool": "fail", "reason": f"Model returned invalid JSON: {response[:100]}"}

        tool_name = tool_call.get("tool", "")
        print(f"  → Tool: {tool_name}")
        if tool_call.get("command"):
            print(f"    Command: {tool_call['command']}")

        # Reject hallucinated tools not in allowed list
        allowed = classify_goal(goal)
        if tool_name not in allowed and tool_name not in ("done", "fail"):
            print(f"  ⛔ Tool '{tool_name}' not allowed for this goal — skipping")
            result = {"ok": False, "error": f"Tool '{tool_name}' not allowed. Use: {allowed}"}
        else:
            # Execute tool
            result = execute_tool(tool_call, confirm_fn=confirm_fn)
        print(f"  {'✅' if result.get('ok') else '❌'} Result: {str(result)[:200]}")

        # Record step
        history.append({"step": step, "tool_call": tool_call, "result": result})

        # Check if terminal
        if result.get("terminal"):
            status = "success" if tool_call.get("tool") == "done" else "failed"
            message = result.get("message", "")
            print(f"\n{'✅' if status == 'success' else '❌'} Task {status}: {message}")
            log_task(task_id, goal, history, status, message)
            return {"status": status, "message": message, "steps": len(history), "history": history}

        # Force done after 4 steps — build summary from history
        if step >= 4:
            print(f"  ⚡ Force-completing after {step} steps")
            summary_lines = []
            for h in history:
                tc  = h.get("tool_call", {})
                res = h.get("result", {})
                t   = tc.get("tool", "")
                if t == "service_status":
                    summary_lines.append(f"{tc.get('name','?')}: {res.get('status','?')}")
                elif t == "shell":
                    out = res.get("stdout","").strip()[:200]
                    if out:
                        summary_lines.append(f"$ {tc.get('command','?')[:60]}\n{out}")
                elif t == "http_get":
                    summary_lines.append(f"GET {tc.get('url','?')}: {res.get('status','?')}")
                elif t in ("file_read","file_patch","file_write"):
                    summary_lines.append(f"{t}: {tc.get('path','?')} — {'ok' if res.get('ok') else 'failed'}")
            message = "\n".join(summary_lines) if summary_lines else "Task completed after exploration."
            log_task(task_id, goal, history, "success", message)
            return {"status": "success", "message": message, "steps": len(history), "history": history}

    # Max steps reached
    log_task(task_id, goal, history, "timeout", "Max steps reached")
    return {"status": "timeout", "steps": max_steps, "history": history}


def log_task(task_id: str, goal: str, history: list,
             status: str, message: str) -> None:
    TASK_LOG.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    if TASK_LOG.exists():
        try:
            tasks = json.loads(TASK_LOG.read_text())
        except Exception:
            tasks = []
    tasks.append({
        "id":        task_id,
        "timestamp": datetime.now().isoformat(),
        "goal":      goal,
        "status":    status,
        "message":   message,
        "steps":     len(history),
        "history":   history
    })
    TASK_LOG.write_text(json.dumps(tasks[-50:], indent=2))


if __name__ == "__main__":
    # Test the tool executor directly
    import sys
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])
        result = run_agent(goal, max_steps=5)
        print(f"\nFinal: {result}")
    else:
        # Test individual tools
        print("Testing shell tool:")
        print(execute_tool({"tool": "shell", "command": "echo hello from LUCI"}))
        print("\nTesting service_status:")
        print(execute_tool({"tool": "service_status", "name": "luci-web"}))
        print("\nTesting http_get:")
        print(execute_tool({"tool": "http_get", "url": "http://localhost:7860/audit"}))
