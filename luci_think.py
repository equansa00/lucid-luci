#!/usr/bin/env python3
"""
LUCI Think — Plan → Execute → Critique → Fix → Respond.
This is LUCI's cognitive loop for complex tasks.
"""
from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional
from luci_sandbox import (
    run_command, run_python, install_packages,
    write_file, read_file, list_workspace,
    git_operation, send_telegram_safe, WORKSPACE,
    VENV_PYTHON, fetch_url
)
from luci_search import search_and_summarize, should_search

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
PLAN_MODEL  = os.getenv("LUCI_PLAN_MODEL",  "luci-core:latest")
CODE_MODEL  = os.getenv("LUCI_CODE_MODEL",  "llama3.1:70b")
MAX_ITERATIONS = int(os.getenv("LUCI_MAX_ITER", "40"))
MAX_CRITIQUE_ROUNDS = 2

TOOL_SYSTEM = f"""You are LUCI — Chip's autonomous AI agent.
Workspace: {WORKSPACE}
Python: {VENV_PYTHON}

CRITICAL: You communicate actions ONLY via XML tool tags.
NEVER write ```bash or ```python blocks. They do nothing.
ONLY tool tags cause real actions. Everything else is just text.

ONE TOOL PER RESPONSE. Wait for result before next tool.

TOOLS:

Search web:
<tool>search_web</tool>
<args>{{"query": "exact search query"}}</args>

Run Python in venv (ONLY way to execute code):
<tool>run_python</tool>
<args>{{"code": "print('hello')", "timeout": 30}}</args>

Write complete file (NEVER use placeholders):
<tool>write_file</tool>
<args>{{"path": "luci_trading/config.py", "content": "# full content\\nimport os\\n"}}</args>

Read file:
<tool>read_file</tool>
<args>{{"path": "luci_trading/config.py"}}</args>

List files:
<tool>list_files</tool>
<args>{{"subpath": "luci_trading"}}</args>

Install packages into venv:
<tool>install_packages</tool>
<args>{{"packages": ["yfinance", "pandas"]}}</args>

Signal completion (ONLY after verified working output):
<tool>done</tool>
<args>{{"summary": "What was built and how to use it"}}</args>

RULES:
1. ONE tool per response. Always.
2. Never call done without first running the code and seeing real output.
3. Never write markdown code blocks — use run_python instead.
4. Write complete file contents in write_file — never "# content here".
5. After every write_file, immediately run_python to test it.
6. Fix ALL errors before calling done.
7. Relative paths only (e.g. "luci_trading/main.py" not full path).
8. You have {MAX_ITERATIONS} steps — start writing files in step 2.
"""

CRITIQUE_PROMPT = """Review your own work. You MUST do the following before calling done:

1. run_python the actual code you wrote and show me the real output
2. If there are ANY errors — fix them completely, then run again
3. Only call done when run_python shows clean output with no exceptions
4. If you used any library (psutil, requests, etc) — verify it's installed in the venv first
5. Prefer stdlib (os, platform, shutil, pathlib) over third-party packages
6. Check: does the output actually answer the original request?

DO NOT call done based on assumption. Run the code. Show the output. Then done."""


def call_ollama(
    messages: list,
    model: str = CODE_MODEL,
    on_token: Optional[Callable] = None
) -> str:
    """Stream response from Ollama."""
    import urllib.request
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.15,
            "num_ctx": 8192,
            "num_predict": 4096,
        }
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}
    )
    full = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    full += token
                    if on_token:
                        on_token(token)
                    if data.get("done"):
                        break
                except Exception:
                    continue
    except Exception as e:
        return f"Ollama error: {e}"
    return full.strip()


def _recover_write_file_args(raw: str) -> dict:
    """
    Try to recover path + content from a non-JSON args string.
    Handles: YAML-ish `path: foo\ncontent: |...`, plain text with
    a filename header, or just bare content.
    """
    # Try: path on first line, rest is content
    # e.g.  path: luci_trading/foo.py\ncontent: |\n  <code>
    path_m = re.search(r'(?:path|file)[:\s]+([^\n]+)', raw)
    # content between ```...``` code fence
    code_m = re.search(r'```(?:\w+)?\n?([\s\S]+?)```', raw)
    if not code_m:
        # content after 'content:' key
        code_m = re.search(r'content[:\s]+\|?\n([\s\S]+)', raw)

    path = path_m.group(1).strip().strip('"\'') if path_m else ""
    content = code_m.group(1) if code_m else ""
    return {"path": path, "content": content}


def parse_tools(text: str) -> list[dict]:
    """Parse XML tool tags. Fallback: extract markdown code blocks."""
    pattern = r'<tool>(.*?)</tool>\s*<args>(.*?)</args>'
    tools = []
    for m in re.finditer(pattern, text, re.DOTALL):
        name = m.group(1).strip()
        raw_args = m.group(2).strip()
        try:
            args = json.loads(raw_args)
        except Exception:
            # For write_file, try to salvage path + content from malformed args
            if name == "write_file":
                args = _recover_write_file_args(raw_args)
            else:
                args = {"raw": raw_args}
        tools.append({"name": name, "args": args})

    if not tools:
        # Fallback: extract annotated code blocks as write_file calls
        # Pattern: ```python\n# path: some/file.py\n<code>```
        for m in re.finditer(
            r'```(?:python|py)\n#\s*(?:path|file)[:\s]+([^\n]+)\n([\s\S]+?)```',
            text
        ):
            tools.append({
                "name": "write_file",
                "args": {"path": m.group(1).strip(), "content": m.group(2)}
            })

    if not tools:
        # Last resort: bare python code blocks → run_python
        py_blocks = re.findall(r'```python\n?(.*?)```', text, re.DOTALL)
        for code in py_blocks:
            if code.strip():
                tools.append({
                    "name": "run_python",
                    "args": {"code": code.strip(), "timeout": 30}
                })

    return tools


def execute_tool(name: str, args: dict,
                 bot_token: str = "", chat_id: str = "") -> str:
    """Execute tool and return string result."""

    if name == "search_web":
        query = args.get("query", "")
        return search_and_summarize(query)

    elif name == "run_python":
        code = args.get("code", "")
        timeout = int(args.get("timeout", 30))
        ok, out = run_python(code, timeout)
        return f"{'✅' if ok else '❌'} Python:\n{out}"

    elif name == "run_command":
        argv = args.get("argv", [])
        if isinstance(argv, str):
            import shlex
            argv = shlex.split(argv)
        timeout = int(args.get("timeout", 60))
        ok, out = run_command(argv, timeout)
        return f"{'✅' if ok else '❌'} Command:\n{out}"

    elif name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        if not path or not content:
            return "❌ write_file needs path and content"
        if content.strip() in ("# content here", "...", "pass"):
            return "❌ PLACEHOLDER DETECTED — write real content"
        ok, msg = write_file(path, content)
        if ok and path.endswith('.py'):
            # Auto syntax check
            check_ok, check_out = run_command(
                [VENV_PYTHON, '-m', 'py_compile',
                 str(WORKSPACE / path)],
                timeout=10
            )
            if not check_ok:
                return f"❌ Written but SYNTAX ERROR:\n{check_out}\nFix the syntax before proceeding."
        return f"{'✅' if ok else '❌'} {msg}"

    elif name == "read_file":
        path = args.get("path", "")
        ok, content = read_file(path)
        return f"📄 {path}:\n{content}" if ok else f"❌ {content}"

    elif name == "list_files":
        subpath = args.get("subpath", "")
        ok, listing = list_workspace(subpath)
        return listing if ok else f"❌ {listing}"

    elif name == "install_packages":
        pkgs = args.get("packages", [])
        if isinstance(pkgs, str):
            pkgs = pkgs.split()
        ok, out = install_packages(pkgs)
        return f"{'✅' if ok else '❌'} pip install:\n{out}"

    elif name == "git":
        git_args = args.get("args", [])
        ok, out = git_operation(git_args)
        return f"{'✅' if ok else '❌'} git:\n{out}"

    elif name == "fetch_url":
        url = args.get("url", "")
        ok, content = fetch_url(url)
        return content if ok else f"❌ {content}"

    elif name == "done":
        summary = args.get("summary", "Task complete.")
        # Hallucination guard: check actual files were written for known task types
        summary_lower = summary.lower()
        trading_dir = WORKSPACE / "luci_trading"
        builds_dir = WORKSPACE / "builds"
        if "trading" in summary_lower and not (
            trading_dir.exists() and any(trading_dir.rglob("*.py"))
        ):
            return (
                "BLOCKED: done called but luci_trading/ has no Python files. "
                "You must write the files first using write_file tools. "
                "Use write_file NOW to create luci_trading/config.py first."
            )
        return f"DONE:{summary}"

    else:
        return f"❌ Unknown tool: {name}"


class LUCIThink:
    """
    The main cognitive loop.
    Plan → Execute → Critique → Fix → Done.
    """
    def __init__(
        self,
        on_update: Optional[Callable] = None,
        on_token: Optional[Callable] = None,
        bot_token: str = "",
        chat_id: str = ""
    ):
        self.on_update = on_update or print
        self.on_token = on_token
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.messages = []
        self.iteration = 0
        self.done = False
        self.result = ""

    def _log(self, msg: str):
        self.on_update(msg)

    def run(self, task: str) -> str:
        """Execute a task through full cognitive loop."""
        self._log(f"🧠 LUCI thinking: {task[:100]}")

        # Phase 1: Plan
        plan = self._plan(task)
        self._log(f"📋 Plan:\n{plan}")

        # Phase 2: Search if needed
        if should_search(task):
            self._log("🔍 Searching for current information...")
            search_context = search_and_summarize(task[:100])
            task = f"{task}\n\n{search_context}"

        # Phase 3: Execute
        self.messages = [
            {"role": "system", "content": TOOL_SYSTEM},
            {"role": "user", "content":
             f"TASK: {task}\n\nPLAN:\n{plan}\n\nNow execute the plan step by step."}
        ]

        critique_rounds = 0
        for i in range(MAX_ITERATIONS):
            self.iteration = i + 1
            self._log(f"\n[Step {self.iteration}]")

            response = call_ollama(
                self.messages,
                model=CODE_MODEL,
                on_token=self.on_token
            )
            if not response:
                break

            self.messages.append({
                "role": "assistant",
                "content": response
            })

            # Show thinking (strip tool blocks)
            thinking = re.sub(
                r'<tool>.*?</tool>\s*<args>.*?</args>', '',
                response, flags=re.DOTALL
            ).strip()
            if thinking:
                self._log(f"💭 {thinking[:200]}")

            # Execute tools
            tools = parse_tools(response)
            if not tools:
                self.messages.append({
                    "role": "user",
                    "content": "Use a tool to make progress, or call done if complete."
                })
                continue

            tool_results = []
            for tool in tools:
                tname = tool["name"]
                targs = tool["args"]
                self._log(f"🔧 {tname}")
                result = execute_tool(
                    tname, targs,
                    self.bot_token, self.chat_id
                )
                self._log(f"   → {result[:150]}")
                tool_results.append(f"[{tname}]: {result}")

                if result.startswith("DONE:"):
                    # Phase 4: Critique before accepting done
                    if critique_rounds < MAX_CRITIQUE_ROUNDS:
                        critique_rounds += 1
                        self._log(f"🔍 Critique round {critique_rounds}...")
                        self.messages.append({
                            "role": "user",
                            "content": CRITIQUE_PROMPT
                        })
                        # Don't add tool_results yet — let critique run
                        break
                    else:
                        self.done = True
                        self.result = result[5:]
                        self._log(f"✅ DONE: {self.result[:200]}")
                        return self.result
            else:
                # Only append results if we didn't break for critique
                self.messages.append({
                    "role": "user",
                    "content": "Results:\n" + "\n".join(tool_results) +
                               "\n\nContinue to next step."
                })

        return self.result or f"Completed {self.iteration} steps."

    def _plan(self, task: str) -> str:
        """Generate a plan before executing."""
        plan_messages = [
            {"role": "system", "content":
             "You are LUCI planning a task. "
             "List the exact steps needed as a numbered list. "
             "Be specific. 5-10 steps max. No code yet."},
            {"role": "user", "content": f"Plan this task:\n{task}"}
        ]
        return call_ollama(plan_messages, model=PLAN_MODEL)
