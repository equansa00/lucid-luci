"""
LUCI Code Context — understands the codebase structure.
Builds a file tree, reads relevant files, tracks what's been seen.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

IGNORE_DIRS  = {".git", "__pycache__", "node_modules", ".venv", "venv",
                "dist", "build", ".next", ".nuxt", "coverage"}
IGNORE_EXTS  = {".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
                ".jpg", ".jpeg", ".png", ".gif", ".ico", ".svg",
                ".mp3", ".mp4", ".wav", ".onnx", ".bin", ".pt", ".pth"}
CODE_EXTS    = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
                ".c", ".cpp", ".h", ".java", ".rb", ".sh", ".yml",
                ".yaml", ".json", ".toml", ".md", ".txt", ".env",
                ".html", ".css", ".sql"}
MAX_FILE_SIZE = 500_000  # 500KB


class CodebaseContext:
    def __init__(self, root: Path):
        self.root = root
        self._file_cache: dict[str, str] = {}
        self._tree_cache: str = ""

    def get_tree(self, max_depth: int = 4) -> str:
        """Return a compact file tree of the project."""
        if self._tree_cache:
            return self._tree_cache
        lines = [f"Project: {self.root.name}/"]
        self._walk_tree(self.root, lines, "", 0, max_depth)
        self._tree_cache = "\n".join(lines)
        return self._tree_cache

    def _walk_tree(self, path: Path, lines: list, prefix: str,
                   depth: int, max_depth: int):
        if depth >= max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
        except PermissionError:
            return
        entries = [e for e in entries
                   if e.name not in IGNORE_DIRS
                   and not e.name.startswith(".")
                   and e.suffix not in IGNORE_EXTS]
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                self._walk_tree(entry, lines, prefix + extension, depth + 1, max_depth)

    def read_file(self, path: str) -> str:
        """Read a file, using cache."""
        key = str(Path(path).resolve())
        if key in self._file_cache:
            return self._file_cache[key]
        p = Path(path) if Path(path).is_absolute() else self.root / path
        if not p.exists():
            return f"[File not found: {path}]"
        if p.stat().st_size > MAX_FILE_SIZE:
            return f"[File too large: {p.stat().st_size:,} bytes — use read_file with line range]"
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            self._file_cache[key] = content
            return content
        except Exception as e:
            return f"[Read error: {e}]"

    def find_files(self, pattern: str) -> list[Path]:
        """Find files matching a name pattern."""
        return [
            p for p in self.root.rglob(pattern)
            if not any(part in IGNORE_DIRS for part in p.parts)
            and p.suffix not in IGNORE_EXTS
        ]

    def get_summary(self) -> str:
        """Quick project summary for context."""
        files = []
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix in CODE_EXTS:
                if not any(part in IGNORE_DIRS for part in p.parts):
                    files.append(p)
        by_ext: dict[str, int] = {}
        for f in files:
            by_ext[f.suffix] = by_ext.get(f.suffix, 0) + 1
        ext_summary = ", ".join(
            f"{v} {k}" for k, v in
            sorted(by_ext.items(), key=lambda x: -x[1])[:8]
        )
        return (
            f"Root: {self.root}\n"
            f"Files: {len(files)} code files ({ext_summary})\n"
            f"Tree:\n{self.get_tree()}"
        )

    def invalidate_cache(self, path: str = ""):
        """Clear cache after edits."""
        if path:
            key = str(Path(path).resolve())
            self._file_cache.pop(key, None)
        else:
            self._file_cache.clear()
            self._tree_cache = ""
