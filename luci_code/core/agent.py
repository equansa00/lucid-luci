"""
LUCI Code Agent — Claude Code feature-parity agentic loop.

Features:
- Real-time streaming with elapsed time per task
- Task decomposition: plan → approve → execute
- Auto-fix loop: run → see error → fix → retry (up to 5x)
- Dependency detection + auto-install
- Key decision approval flow
- Rollback via git stash
- Multi-file change tracking
"""
from __future__ import annotations
import json
import re
import time
import subprocess
import requests
from pathlib import Path
from typing import Callable

from .tools import Tools, ToolResult
from .context import CodebaseContext

OLLAMA_URL    = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "luci-coder:latest"
MAX_ITER      = 20
MAX_FIX_LOOP  = 5

SYSTEM_PROMPT = """You are LUCI Code — an expert software engineering agent, \
the local equivalent of Claude Code.
You help Chip (Edward Equansa) write, read, edit, debug, and understand code.
You run on luci-coder (qwen2.5-coder base) — built for software engineering.

═══ TOOL PROTOCOL ═══
Output tool calls as XML blocks. Chain them — read → analyze → act → verify.

<tool>read_file</tool>
<args>{"path": "src/main.py", "start": 1, "end": 50}</args>

<tool>create_file</tool>
<args>{"path": "new_file.py", "content": "COMPLETE FILE CONTENT — NO PLACEHOLDERS"}</args>

<tool>str_replace</tool>
<args>{"path": "file.py", "old_str": "exact text", "new_str": "replacement"}</args>

<tool>bash</tool>
<args>{"command": "python3 -m pytest tests/ -v", "timeout": 60}</args>

<tool>glob</tool><args>{"pattern": "**/*.py"}</args>

<tool>grep</tool>
<args>{"pattern": "def process", "path": ".", "include": "*.py"}</args>

<tool>install_deps</tool>
<args>{"packages": ["requests", "fastapi"]}</args>

<tool>tree</tool><args>{"path": ".", "max_depth": 3}</args>
<tool>ls</tool><args>{"path": "src/"}</args>
<tool>git_diff</tool><args>{}</args>
<tool>git_status</tool><args>{}</args>
<tool>git_commit</tool><args>{"message": "Fix: error handling"}</args>

═══ DECISION PROTOCOL ═══
For KEY DECISIONS that affect architecture or are hard to reverse, output:
<decision>
WHAT: Brief description of the decision
OPTIONS: Option A | Option B | Option C
RECOMMEND: Which option and why (1 sentence)
IMPACT: What this changes
</decision>
Wait for user input before proceeding.

═══ PLAN PROTOCOL ═══
ALWAYS output a plan before calling any tool — even for simple 1-step tasks. No exceptions.
<plan>
1. Step one description
2. Step two description
</plan>
Wait for user approval before executing. Do not call any tool before the plan is approved.

═══ AVAILABLE TOOLS (use ONLY these) ═══
read_file, create_file, str_replace, bash, glob, grep, tree, ls,
git_diff, git_status, git_commit, install_deps
DO NOT invent tools like git_add, git_push, git_pull, verify, analyze_file.
Use bash to run any git commands not in the list above.
STOP after task is complete — do not loop on git_status or git_diff.

═══ RULES ═══
0. NEVER describe what a file contains. ALWAYS call read_file first and use the ACTUAL output.
   If you have not called read_file in this response, you do not know what is in the file.
1. Always read a file before editing it.
2. Use str_replace for targeted edits — never rewrite unless necessary.
3. After creating/editing code, ALWAYS run it to verify it works.
4. If a run fails, fix it immediately — don't report the error and stop.
5. Auto-detect and install missing dependencies.
6. Be direct. No "let me know what you'd like" — just do it.
7. Complete the full task. Do not stop halfway.
8. Chain tools: read → analyze → act → verify → commit.
9. When creating files, ALWAYS put complete content in args. No placeholders.
10. Default to action. A reasonable attempt beats a clarifying question.
11. ALWAYS read_file before str_replace. Never guess old_str — read it first.
12. When editing your own source, read the EXACT lines first, then str_replace.
11. After completing a task, give a concise summary of what changed.
12. Track elapsed time mentally — flag if a step takes unexpectedly long.
"""

TOOL_PATTERN = re.compile(
    r"<tool>\s*(\w+)\s*</tool>\s*<args>(.*?)</args>",
    re.DOTALL
)
DECISION_PATTERN = re.compile(r"<decision>(.*?)</decision>", re.DOTALL)
PLAN_PATTERN     = re.compile(r"<plan>(.*?)</plan>",     re.DOTALL)

# Patterns that indicate a bash command needs approval
DESTRUCTIVE_PATTERNS = [
    r"\brm\s+-rf\b", r"\bdrop\s+table\b", r"\bformat\b",
    r"\btruncate\b", r"\bdelete\s+from\b", r"\bmkfs\b",
    r"\bdd\s+if=", r"\bchmod\s+777\b",
]


def _is_destructive(cmd: str) -> bool:
    return any(re.search(p, cmd, re.IGNORECASE) for p in DESTRUCTIVE_PATTERNS)


def _detect_missing_dep(error_output: str) -> list[str]:
    """Parse error output for missing Python packages."""
    pkgs = []
    for m in re.finditer(r"No module named '([^']+)'", error_output):
        pkg = m.group(1).split(".")[0]
        pkgs.append(pkg)
    for m in re.finditer(r"ModuleNotFoundError: No module named '([^']+)'", error_output):
        pkg = m.group(1).split(".")[0]
        pkgs.append(pkg)
    return list(set(pkgs))


class Agent:
    def __init__(
        self,
        workspace: Path,
        model: str = DEFAULT_MODEL,
        on_token:    Callable[[str], None]             | None = None,
        on_tool:     Callable[[str, dict, ToolResult], None] | None = None,
        on_status:   Callable[[str], None]             | None = None,
        on_decision: Callable[[str], str]              | None = None,
        on_plan:     Callable[[str], bool]             | None = None,
        confirm_bash: bool = True,
        auto_commit:  bool = False,
    ):
        self.workspace    = workspace
        self.model        = model
        self.tools        = Tools(workspace)
        self.context      = CodebaseContext(workspace)
        self.messages:    list[dict] = []
        self.on_token     = on_token    or (lambda t: None)
        self.on_tool      = on_tool     or (lambda n, a, r: None)
        self.on_status    = on_status   or (lambda s: None)
        self.on_decision  = on_decision or (lambda d: "proceed")
        self.on_plan      = on_plan     or (lambda p: True)
        self.confirm_bash = confirm_bash
        self.auto_commit  = auto_commit
        self._changed_files: set[str] = set()
        self._stash_created = False
        self._setup_system()

    def _setup_system(self):
        ctx = self.context.get_summary()
        self.messages = [{"role": "system",
                          "content": SYSTEM_PROMPT + f"\n\nCURRENT PROJECT:\n{ctx}"}]

    # ── PUBLIC ────────────────────────────────────────────────────────────
    def chat(self, user_message: str) -> str:
        """Full agentic loop: plan → approve → execute → verify → fix."""
        self._changed_files.clear()
        self.messages.append({"role": "user", "content": user_message})
        return self._agent_loop()

    def rollback(self) -> str:
        """Undo all changes since last chat() call via git stash."""
        r = self.tools.bash("git stash")
        return r.output if r.success else r.error

    def reset(self):
        self._setup_system()

    # ── AGENT LOOP ────────────────────────────────────────────────────────
    def _agent_loop(self) -> str:
        full_response = ""
        iterations    = 0
        fix_attempts  = 0
        last_error    = ""

        while iterations < MAX_ITER:
            iterations += 1
            t_iter = time.perf_counter()
            self.on_status(f"Thinking... (iteration {iterations})")

            response_text = self._call_ollama()
            full_response = response_text

            elapsed = time.perf_counter() - t_iter

            # ── Handle plan block ────────────────────────────────────────
            plan_m = PLAN_PATTERN.search(response_text)
            if plan_m and iterations > 1:
                # Model re-planned instead of executing — strip plan and continue
                response_text = PLAN_PATTERN.sub("", response_text).strip()
            if plan_m and iterations == 1:
                plan_text = plan_m.group(1).strip()
                proceed = self.on_plan(plan_text)
                if not proceed:
                    self.messages.append({"role": "assistant", "content": response_text})
                    return response_text
                # Plan approved — loop to execute
                self.messages.append({"role": "assistant", "content": response_text})
                first_step = plan_text.strip().splitlines()[0] if plan_text.strip() else ""
                self.messages.append({"role": "user", "content": f"Plan approved. Execute now using tools. Start with: {first_step}. Do NOT output another plan — use tools directly."})
                continue

            # ── Handle decision block ────────────────────────────────────
            decision_m = DECISION_PATTERN.search(response_text)
            if decision_m:
                decision_text = decision_m.group(1).strip()
                user_choice = self.on_decision(decision_text)
                self.messages.append({"role": "assistant", "content": response_text})
                self.messages.append({"role": "user",
                                      "content": f"Decision: {user_choice}"})
                continue

            # ── Find tool calls ──────────────────────────────────────────
            tool_calls = TOOL_PATTERN.findall(response_text)
            if not tool_calls:
                self.messages.append({"role": "assistant", "content": response_text})
                # If model returned text but no tools and task needs file creation, nudge once
                if iterations == 1 and any(w in self.messages[0]["content"].lower() 
                                           for w in ["create", "write", "make", "build"]):
                    self.messages.append({"role": "user", "content": 
                        "You must use tools to complete this task. Use create_file now."})
                    continue
                break

            # ── Execute tools ────────────────────────────────────────────
            tool_results    = []
            run_failed      = False
            failed_output   = ""

            for tool_name, args_str in tool_calls:
                t_tool = time.perf_counter()
                try:
                    clean = args_str.strip()
                    # Replace triple-quoted strings with JSON-safe equivalents
                    clean = re.sub(r'"""(.*?)"""',
                        lambda m: json.dumps(m.group(1)), clean, flags=re.DOTALL)
                    args = json.loads(clean)
                except Exception:
                    args = {}

                # Bash confirmation / destructive check
                if tool_name == "bash":
                    cmd = args.get("command", "")
                    if _is_destructive(cmd):
                        # Always confirm destructive commands
                        answer = self.on_decision(
                            f"DESTRUCTIVE COMMAND DETECTED\n"
                            f"Command: {cmd}\n"
                            f"OPTIONS: proceed | cancel\n"
                            f"RECOMMEND: cancel — verify intent first\n"
                            f"IMPACT: May permanently delete or modify data"
                        )
                        if "cancel" in answer.lower() or "no" in answer.lower():
                            tool_results.append(
                                "<tool_result tool='bash'>\nCANCELLED by user\n</tool_result>"
                            )
                            continue
                    elif self.confirm_bash:
                        # Standard bash confirmation
                        answer = self.on_decision(
                            f"BASH COMMAND\n"
                            f"Command: {cmd}\n"
                            f"OPTIONS: proceed | cancel | edit\n"
                            f"RECOMMEND: proceed\n"
                            f"IMPACT: Runs in {self.workspace}"
                        )
                        if "cancel" in answer.lower():
                            tool_results.append(
                                "<tool_result tool='bash'>\nCANCELLED by user\n</tool_result>"
                            )
                            continue

                # Self-edit gate
                if tool_name in ("create_file", "str_replace") and \
                        self.tools._is_self_edit(args.get("path", "")):
                    answer = self.on_decision(
                        f"SELF-EDIT REQUEST\n"
                        f"WHAT: LUCI wants to modify its own source\n"
                        f"File: {args.get('path', '')}\n"
                        f"OPTIONS: approve | reject\n"
                        f"RECOMMEND: review carefully before approving\n"
                        f"IMPACT: Changes LUCI Code itself — takes effect immediately"
                    )
                    if "reject" in answer.lower() or answer.strip().lower() in ("n", "no"):
                        tool_results.append(
                            f"<tool_result tool='{tool_name}'>Self-edit rejected by user</tool_result>"
                        )
                        continue

                result = self._execute_tool(tool_name, args)
                tool_elapsed = time.perf_counter() - t_tool
                self.on_tool(tool_name, args, result, tool_elapsed)

                # Track changed files for summary
                if tool_name in ("str_replace", "create_file"):
                    self._changed_files.add(args.get("path", ""))
                    self.context.invalidate_cache(args.get("path", ""))

                # Detect run failures for auto-fix loop
                if tool_name == "bash" and not result.success:
                    run_failed   = True
                    failed_output = str(result)

                    # Auto-install missing dependencies
                    missing = _detect_missing_dep(failed_output)
                    if missing and fix_attempts < MAX_FIX_LOOP:
                        self.on_status(f"Auto-installing missing deps: {missing}")
                        install_r = self._execute_tool(
                            "install_deps", {"packages": missing}
                        )
                        self.on_tool("install_deps", {"packages": missing}, install_r, 0)
                        if install_r.success:
                            # Retry the same command
                            result = self._execute_tool(tool_name, args)
                            self.on_tool(tool_name, args, result, 0)
                            run_failed = not result.success
                            failed_output = str(result) if run_failed else ""

                result_str = str(result)[:6000]
                tool_results.append(
                    f"<tool_result tool='{tool_name}' "
                    f"elapsed='{tool_elapsed:.1f}s' "
                    f"success='{result.success}'>\n{result_str}\n</tool_result>"
                )

            # ── Auto-fix loop ────────────────────────────────────────────
            combined = response_text + "\n\n" + "\n".join(tool_results)
            self.messages.append({"role": "assistant", "content": combined})

            if run_failed and fix_attempts < MAX_FIX_LOOP:
                fix_attempts += 1
                self.on_status(
                    f"Run failed — auto-fixing (attempt {fix_attempts}/{MAX_FIX_LOOP})..."
                )
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"The command failed. Fix it now. Error:\n{failed_output}\n\n"
                        f"Read the relevant files, fix the issue, run again to verify. "
                        f"Do not ask — just fix it."
                    )
                })
                last_error = failed_output
                continue
            elif not run_failed and tool_calls:
                # Tools succeeded — nudge model to continue with next step
                completed = [tc[0] for tc in tool_calls]
                housekeeping = {"git_status", "git_diff", "git_commit", "git_push", "git_pull"}
                if all(t in housekeeping for t in completed):
                    break
                self.messages.append({"role": "user", "content": f"Good. Completed: {completed}. Now execute the next step using a tool. You must call a tool — do not respond with text only."})
                continue

        # ── Post-task summary ────────────────────────────────────────────
        if self._changed_files:
            self.on_status(
                f"Done. Changed: {', '.join(sorted(self._changed_files))}"
            )

        return full_response

    # ── OLLAMA ────────────────────────────────────────────────────────────
    def _call_ollama(self) -> str:
        payload = {
            "model":   self.model,
            "messages": self.messages,
            "stream":  True,
            "options": {"temperature": 0.2, "num_ctx": 16384},
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload,
                                 stream=True, timeout=300)
            resp.raise_for_status()
        except Exception as e:
            return f"[Ollama error: {e}]"

        full = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data  = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    self.on_token(token)
                    full += token
                if data.get("done"):
                    break
            except json.JSONDecodeError:
                continue
        return full

    # ── TOOL DISPATCH ─────────────────────────────────────────────────────
    def _execute_tool(self, name: str, args: dict) -> ToolResult:
        t = self.tools
        dispatch = {
            "read_file":    lambda: t.read_file(
                                args.get("path", ""),
                                args.get("start", 1),
                                args.get("end", -1)),
            "create_file":  lambda: t.create_file(
                                args.get("path", ""),
                                args.get("content", "")),
            "str_replace":  lambda: t.str_replace(
                                args.get("path", ""),
                                args.get("old_str", ""),
                                args.get("new_str", "")),
            "bash":         lambda: t.bash(
                                args.get("command", ""),
                                args.get("timeout", 60)),
            "glob":         lambda: t.glob(
                                args.get("pattern", "*"),
                                args.get("base", "")),
            "grep":         lambda: t.grep(
                                args.get("pattern", ""),
                                args.get("path", "."),
                                args.get("recursive", True),
                                args.get("include", "")),
            "install_deps": lambda: t.install_deps(args.get("packages", [])),
            "git_diff":     lambda: t.git_diff(args.get("staged", False)),
            "git_status":   lambda: t.git_status(),
            "git_log":      lambda: t.git_log(args.get("n", 10)),
            "git_commit":   lambda: t.git_commit(args.get("message", "update")),
            "git_push":     lambda: t.git_push(),
            "tree":         lambda: t.tree(
                                args.get("path", "."),
                                args.get("max_depth", 3)),
            "ls":           lambda: t.ls(args.get("path", ".")),
            "change_dir":   lambda: t.change_dir(args.get("path", ".")),
        }
        fn = dispatch.get(name)
        if fn is None:
            return ToolResult(name, False, "", f"Unknown tool: {name}")
        result = fn()
        if name == "change_dir" and result.success:
            self.workspace      = t.workspace
            self.context.root   = t.workspace
            self.context.invalidate_cache()
        return result
