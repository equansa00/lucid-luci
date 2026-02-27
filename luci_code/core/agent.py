"""
LUCI Code Agent — the think→act→observe loop.
Uses Ollama with tool-use JSON protocol.
"""
from __future__ import annotations
import json
import re
import requests
from pathlib import Path
from typing import Callable, Iterator

from .tools import Tools, ToolResult
from .context import CodebaseContext

OLLAMA_URL   = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:14b"

SYSTEM_PROMPT = """You are LUCI Code — an expert software engineering agent.
You help Chip (Edward Equansa) write, read, edit, debug, and understand code.

You have access to these tools. Use them by outputting JSON blocks:

<tool>read_file</tool>
<args>{"path": "src/main.py", "start": 1, "end": 50}</args>

<tool>create_file</tool>
<args>{"path": "new_file.py", "content": "# content here"}</args>

<tool>str_replace</tool>
<args>{"path": "file.py", "old_str": "exact text to replace", "new_str": "new text"}</args>

<tool>bash</tool>
<args>{"command": "python3 -m pytest tests/", "timeout": 30}</args>

<tool>glob</tool>
<args>{"pattern": "**/*.py"}</args>

<tool>grep</tool>
<args>{"pattern": "def process_message", "path": ".", "include": "*.py"}</args>

<tool>git_diff</tool>
<args>{}</args>

<tool>git_status</tool>
<args>{}</args>

<tool>git_commit</tool>
<args>{"message": "Fix: error handling in auth module"}</args>

<tool>git_push</tool>
<args>{}</args>

<tool>tree</tool>
<args>{"path": ".", "max_depth": 3}</args>

<tool>ls</tool>
<args>{"path": "src/"}</args>

<tool>change_dir</tool>
<args>{"path": "/home/user/myproject"}</args>

RULES:
1. Always read a file before editing it.
2. Use str_replace for targeted edits — never rewrite entire files unless necessary.
3. After editing, verify with bash when possible.
4. Be direct and concise. Show diffs, not full files.
5. Read multiple related files before acting on complex tasks.
6. Never fabricate file contents or command output.
7. Commit with clear messages after completing a task.
8. You ARE LUCI — speak in first person, own your actions.
9. NEVER ask "let me know what you'd like to explore" or
   "which files would you like me to look at" — just look at them.
10. When asked "what files make up this project", read ALL of them.
    Use tree first, then read key files (README, main entry points,
    config files). Do not stop and ask permission between steps.
11. Complete the full task in one response. Do not stop halfway.
12. Never end a response with a question unless you genuinely cannot
    proceed without the answer.
13. Default to action. A reasonable attempt beats a clarifying question.
14. When asked for a test or benchmark, create and run one immediately.
15. Chain tool calls — read → analyze → act → verify, all in sequence.
"""

TOOL_PATTERN = re.compile(
    r"<tool>\s*(\w+)\s*</tool>\s*<args>(.*?)</args>",
    re.DOTALL
)


class Agent:
    def __init__(
        self,
        workspace: Path,
        model: str = DEFAULT_MODEL,
        on_token: Callable[[str], None] | None = None,
        on_tool: Callable[[str, dict, ToolResult], None] | None = None,
        confirm_bash: bool = True,
        auto_commit: bool = False,
    ):
        self.workspace = workspace
        self.model = model
        self.tools = Tools(workspace)
        self.context = CodebaseContext(workspace)
        self.messages: list[dict] = []
        self.on_token = on_token or (lambda t: None)
        self.on_tool = on_tool or (lambda n, a, r: None)
        self.confirm_bash = confirm_bash
        self.auto_commit = auto_commit
        self._setup_system()

    def _setup_system(self):
        ctx_summary = self.context.get_summary()
        system = SYSTEM_PROMPT + f"\n\nCURRENT PROJECT:\n{ctx_summary}"
        self.messages = [{"role": "system", "content": system}]

    def chat(self, user_message: str) -> str:
        """Send a message, run the tool loop, return final response."""
        self.messages.append({"role": "user", "content": user_message})
        full_response = ""
        iterations = 0
        max_iterations = 20

        while iterations < max_iterations:
            iterations += 1
            response_text = self._call_ollama()
            full_response = response_text

            # Find all tool calls in the response
            tool_calls = TOOL_PATTERN.findall(response_text)
            if not tool_calls:
                # No tools — final answer
                self.messages.append({"role": "assistant", "content": response_text})
                break

            # Execute each tool
            tool_results = []
            for tool_name, args_str in tool_calls:
                try:
                    args = json.loads(args_str.strip())
                except json.JSONDecodeError:
                    args = {}
                result = self._execute_tool(tool_name, args)
                self.on_tool(tool_name, args, result)
                tool_results.append(
                    f"<tool_result tool='{tool_name}'>\n{str(result)[:4000]}\n</tool_result>"
                )
                # Invalidate context cache after file changes
                if tool_name in ("str_replace", "create_file"):
                    self.context.invalidate_cache(args.get("path", ""))

            # Add assistant message + tool results
            combined = response_text + "\n\n" + "\n\n".join(tool_results)
            self.messages.append({"role": "assistant", "content": combined})

        return full_response

    def _call_ollama(self) -> str:
        """Call Ollama streaming, return full text."""
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "options": {"temperature": 0.2, "num_ctx": 16384},
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300)
            resp.raise_for_status()
        except Exception as e:
            return f"[Ollama error: {e}]"

        full = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    self.on_token(token)
                    full += token
                if data.get("done"):
                    break
            except json.JSONDecodeError:
                continue
        return full

    def _execute_tool(self, name: str, args: dict) -> ToolResult:
        """Dispatch tool call to Tools instance."""
        t = self.tools
        dispatch = {
            "read_file":    lambda: t.read_file(
                                args.get("path", ""),
                                args.get("start", 1),
                                args.get("end", -1)
                            ),
            "create_file":  lambda: t.create_file(
                                args.get("path", ""),
                                args.get("content", "")
                            ),
            "str_replace":  lambda: t.str_replace(
                                args.get("path", ""),
                                args.get("old_str", ""),
                                args.get("new_str", "")
                            ),
            "bash":         lambda: t.bash(
                                args.get("command", ""),
                                args.get("timeout", 60)
                            ),
            "glob":         lambda: t.glob(
                                args.get("pattern", "*"),
                                args.get("base", "")
                            ),
            "grep":         lambda: t.grep(
                                args.get("pattern", ""),
                                args.get("path", "."),
                                args.get("recursive", True),
                                args.get("include", "")
                            ),
            "git_diff":     lambda: t.git_diff(args.get("staged", False)),
            "git_status":   lambda: t.git_status(),
            "git_log":      lambda: t.git_log(args.get("n", 10)),
            "git_commit":   lambda: t.git_commit(args.get("message", "update")),
            "git_push":     lambda: t.git_push(),
            "tree":         lambda: t.tree(
                                args.get("path", "."),
                                args.get("max_depth", 3)
                            ),
            "ls":           lambda: t.ls(args.get("path", ".")),
            "change_dir":   lambda: t.change_dir(args.get("path", ".")),
        }
        fn = dispatch.get(name)
        if fn is None:
            return ToolResult(name, False, "", f"Unknown tool: {name}")
        result = fn()
        # Keep agent workspace in sync when agent uses change_dir
        if name == "change_dir" and result.success:
            self.workspace = t.workspace
            self.context.root = t.workspace
            self.context.invalidate_cache()
        return result

    def reset(self):
        """Start a new conversation."""
        self._setup_system()
