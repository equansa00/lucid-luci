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


# ── Planner prompt ────────────────────────────────────────────────────────────
def build_planner_prompt(goal: str, history: list, tools: dict) -> str:
    tool_list = "\n".join(
        f'- {name}: {info["description"]} — e.g. {json.dumps(info["example"])}'
        for name, info in tools.items()
    )

    history_str = ""
    for h in history[-6:]:
        history_str += f"\nStep: {json.dumps(h.get('tool_call'))}\nResult: {json.dumps(h.get('result'))}\n"

    return f"""You are LUCI's autonomous execution engine.

GOAL: {goal}

AVAILABLE TOOLS:
{tool_list}

EXECUTION HISTORY:
{history_str or "None yet — this is the first step."}

INSTRUCTIONS:
- Respond with ONLY a JSON tool call object
- No explanation, no markdown, just raw JSON
- Choose the best next tool to make progress toward the goal
- If the goal is complete, use "done"
- If you've tried 3+ times and failed, use "fail"
- File paths should be absolute or use ~/beast/workspace/
- Keep commands simple and targeted

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

        print(f"  → Tool: {tool_call.get('tool')}")
        if tool_call.get("command"):
            print(f"    Command: {tool_call['command']}")

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
            return {"status": status, "message": message, "steps": len(history)}

    # Max steps reached
    log_task(task_id, goal, history, "timeout", "Max steps reached")
    return {"status": "timeout", "steps": max_steps}


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
