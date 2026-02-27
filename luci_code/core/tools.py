"""
LUCI Code Tools — all actions the agent can take.
Each tool returns a ToolResult with output and metadata.
"""
from __future__ import annotations
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ToolResult:
    tool: str
    success: bool
    output: str
    error: str = ""
    metadata: dict = field(default_factory=dict)

    def __str__(self):
        if self.success:
            return self.output
        return f"ERROR: {self.error}\n{self.output}"


class Tools:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.history: list[ToolResult] = []

    def _record(self, result: ToolResult) -> ToolResult:
        self.history.append(result)
        return result

    # ── READ ──────────────────────────────────────────────────────────────
    def read_file(self, path: str, start: int = 1, end: int = -1) -> ToolResult:
        """Read a file, optionally a line range."""
        p = self._resolve(path)
        if not p.exists():
            return self._record(ToolResult("read_file", False, "", f"File not found: {path}"))
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            s = max(0, start - 1)
            e = total if end == -1 else min(end, total)
            numbered = "\n".join(f"{i+s+1:4d} │ {l}" for i, l in enumerate(lines[s:e]))
            return self._record(ToolResult(
                "read_file", True, numbered,
                metadata={"path": str(p), "total_lines": total, "shown": f"{s+1}-{e}"}
            ))
        except Exception as ex:
            return self._record(ToolResult("read_file", False, "", str(ex)))

    # ── WRITE ─────────────────────────────────────────────────────────────
    def create_file(self, path: str, content: str) -> ToolResult:
        """Create a new file with content."""
        p = self._resolve(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            lines = content.count("\n") + 1
            return self._record(ToolResult(
                "create_file", True,
                f"Created {path} ({lines} lines)",
                metadata={"path": str(p), "lines": lines}
            ))
        except Exception as ex:
            return self._record(ToolResult("create_file", False, "", str(ex)))

    # ── STR_REPLACE ───────────────────────────────────────────────────────
    def str_replace(self, path: str, old_str: str, new_str: str) -> ToolResult:
        """
        Replace EXACTLY ONE occurrence of old_str with new_str.
        Fails if old_str appears 0 or 2+ times.
        """
        p = self._resolve(path)
        if not p.exists():
            return self._record(ToolResult("str_replace", False, "", f"File not found: {path}"))
        try:
            original = p.read_text(encoding="utf-8")
            count = original.count(old_str)
            if count == 0:
                return self._record(ToolResult(
                    "str_replace", False, "",
                    f"old_str not found in {path}. Check whitespace/indentation."
                ))
            if count > 1:
                return self._record(ToolResult(
                    "str_replace", False, "",
                    f"old_str appears {count} times — must be unique. Add more context."
                ))
            updated = original.replace(old_str, new_str, 1)
            p.write_text(updated, encoding="utf-8")

            # Build a simple unified diff for display
            old_lines = old_str.splitlines()
            new_lines = new_str.splitlines()
            diff_lines = (
                [f"  - {l}" for l in old_lines] +
                [f"  + {l}" for l in new_lines]
            )
            diff_preview = "\n".join(diff_lines[:40])
            return self._record(ToolResult(
                "str_replace", True,
                f"Replaced in {path}:\n{diff_preview}",
                metadata={"path": str(p), "old_lines": len(old_lines), "new_lines": len(new_lines)}
            ))
        except Exception as ex:
            return self._record(ToolResult("str_replace", False, "", str(ex)))

    # ── BASH ──────────────────────────────────────────────────────────────
    def bash(self, command: str, timeout: int = 60, cwd: str = "") -> ToolResult:
        """Run a shell command and return REAL output."""
        work_dir = Path(cwd) if cwd else self.workspace
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True,
                text=True, timeout=timeout, cwd=str(work_dir)
            )
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            if not output.strip():
                output = f"[Exit code {result.returncode} — no output]"
            return self._record(ToolResult(
                "bash", result.returncode == 0, output,
                error="" if result.returncode == 0 else f"Exit code {result.returncode}",
                metadata={"command": command, "returncode": result.returncode}
            ))
        except subprocess.TimeoutExpired:
            return self._record(ToolResult("bash", False, "", f"Command timed out after {timeout}s"))
        except Exception as ex:
            return self._record(ToolResult("bash", False, "", str(ex)))

    # ── GLOB / FIND ───────────────────────────────────────────────────────
    def glob(self, pattern: str, base: str = "") -> ToolResult:
        """Find files matching a glob pattern."""
        base_path = self._resolve(base) if base else self.workspace
        try:
            matches = sorted(base_path.glob(pattern))
            if not matches:
                return self._record(ToolResult("glob", True, f"No files matching {pattern}"))
            lines = "\n".join(str(m.relative_to(self.workspace)) for m in matches[:200])
            return self._record(ToolResult(
                "glob", True, lines,
                metadata={"count": len(matches), "pattern": pattern}
            ))
        except Exception as ex:
            return self._record(ToolResult("glob", False, "", str(ex)))

    # ── GREP ──────────────────────────────────────────────────────────────
    def grep(self, pattern: str, path: str = ".", recursive: bool = True,
             include: str = "") -> ToolResult:
        """Search for a pattern in files."""
        cmd = f"grep -n"
        if recursive:
            cmd += " -r"
        if include:
            cmd += f" --include='{include}'"
        cmd += f" {repr(pattern)} {path}"
        result = self.bash(cmd, cwd=str(self.workspace))
        result.tool = "grep"
        return result

    # ── GIT ───────────────────────────────────────────────────────────────
    def git_diff(self, staged: bool = False) -> ToolResult:
        cmd = "git diff --staged" if staged else "git diff"
        r = self.bash(cmd)
        r.tool = "git_diff"
        return r

    def git_status(self) -> ToolResult:
        r = self.bash("git status --short")
        r.tool = "git_status"
        return r

    def git_log(self, n: int = 10) -> ToolResult:
        r = self.bash(f"git log --oneline -{n}")
        r.tool = "git_log"
        return r

    def git_commit(self, message: str) -> ToolResult:
        r = self.bash(f'git add -A && git commit -m {repr(message)}')
        r.tool = "git_commit"
        return r

    def git_push(self) -> ToolResult:
        r = self.bash("git push")
        r.tool = "git_push"
        return r

    # ── DIRECTORY TREE ────────────────────────────────────────────────────
    def tree(self, path: str = ".", max_depth: int = 3,
             exclude: list[str] | None = None) -> ToolResult:
        """Show directory tree."""
        exclude = exclude or [
            "__pycache__", ".git", "node_modules", ".venv",
            "venv", "*.pyc", ".DS_Store", "dist", "build"
        ]
        cmd = f"find {path} -maxdepth {max_depth} -not -path '*/.git/*' -not -name '__pycache__' | sort | head -200"
        r = self.bash(cmd)
        r.tool = "tree"
        return r

    # ── LIST DIRECTORY ────────────────────────────────────────────────────
    def ls(self, path: str = ".") -> ToolResult:
        p = self._resolve(path)
        try:
            entries = sorted(p.iterdir())
            lines = []
            for e in entries:
                if e.name.startswith("."):
                    continue
                size = f"{e.stat().st_size:>8,}" if e.is_file() else "     dir"
                lines.append(f"{size}  {e.name}{'/' if e.is_dir() else ''}")
            return self._record(ToolResult("ls", True, "\n".join(lines) or "(empty)"))
        except Exception as ex:
            return self._record(ToolResult("ls", False, "", str(ex)))

    # ── CHANGE DIR ────────────────────────────────────────────────────────
    def change_dir(self, path: str) -> ToolResult:
        """Change the working directory."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return self._record(ToolResult(
                "change_dir", False, "", f"Directory not found: {path}"
            ))
        if not p.is_dir():
            return self._record(ToolResult(
                "change_dir", False, "", f"Not a directory: {path}"
            ))
        self.workspace = p
        return self._record(ToolResult(
            "change_dir", True, f"Working directory: {p}"
        ))

    # ── INTERNAL ──────────────────────────────────────────────────────────
    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self.workspace / p
