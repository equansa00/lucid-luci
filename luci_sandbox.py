#!/usr/bin/env python3
"""
LUCI Sandbox — secure execution environment.
All file and command operations go through here.
NO shell=True anywhere. Workspace-only enforcement.
"""
from __future__ import annotations
import os
import re
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

WORKSPACE = Path("/home/equansa00/beast/workspace").resolve()
VENV_PYTHON = str(WORKSPACE / ".venv" / "bin" / "python")
VENV_PIP = str(WORKSPACE / ".venv" / "bin" / "pip")


class SecurityError(Exception):
    pass


# ── Path enforcement ───────────────────────────────────
def safe_path(p: str | Path) -> Path:
    """Resolve path and ensure it's inside workspace."""
    try:
        if str(p).startswith("~/"):
            p = str(p).replace("~/", str(Path.home()) + "/", 1)
        resolved = (WORKSPACE / p).resolve() if not Path(p).is_absolute() else Path(p).expanduser().resolve()
        if not str(resolved).startswith(str(WORKSPACE)):
            raise SecurityError(
                f"Path '{p}' resolves to '{resolved}' "
                f"which is outside workspace '{WORKSPACE}'"
            )
        return resolved
    except SecurityError:
        raise
    except Exception as e:
        raise SecurityError(f"Invalid path '{p}': {e}")


def safe_env() -> dict:
    """Minimal safe environment for subprocess calls."""
    return {
        "PATH": f"{WORKSPACE}/.venv/bin:/usr/bin:/bin",
        "PYTHONPATH": str(WORKSPACE),
        "HOME": str(Path.home()),
        "VIRTUAL_ENV": str(WORKSPACE / ".venv"),
        "LANG": "en_US.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


# ── Command allowlist ──────────────────────────────────
ALLOWED_EXECUTABLES = {
    "python3", "python3.14", "python",
    "pip", "pip3",
    "git", "ls", "find", "cat", "echo",
    "mkdir", "touch",
    "grep", "head", "tail", "wc", "sort",
    "pytest", "black", "mypy", "ruff",
    "luci-trader", "luci-code",
    # curl/rm/mv/cp removed — use fetch_url() for network, confirm gate for destructive ops
}

# Operations that require confirmation before executing
CONFIRM_REQUIRED = os.getenv("LUCI_CONFIRM", "1") == "1"
NEEDS_CONFIRM_OPS = {"rm", "mv", "cp"}

BLOCKED_PATTERNS = [
    r";\s*rm\s+-rf",
    r"\|\s*bash",
    r"\|\s*sh\b",
    r"`.*`",
    r"\$\(.*\)",
    r">\s*/(?!home)",
    r"chmod\s+[0-7]*7\s+/",
    r":\(\)\s*\{",
    r"sudo",
    r"su\s+-",
    r"passwd",
    r"dd\s+if=",
    r"mkfs",
    r"shutdown",
    r"reboot",
    r"curl.*\|\s*(bash|sh)",
    r"wget.*\|\s*(bash|sh)",
]


def validate_command(argv: list[str]) -> None:
    """Validate command argv. Raises SecurityError if unsafe."""
    if not argv:
        raise SecurityError("Empty command")

    executable = Path(argv[0]).name
    if executable not in ALLOWED_EXECUTABLES:
        raise SecurityError(
            f"Executable '{executable}' not in allowlist. "
            f"Allowed: {sorted(ALLOWED_EXECUTABLES)}"
        )

    full_cmd = " ".join(argv)
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, full_cmd, re.IGNORECASE):
            raise SecurityError(
                f"Blocked pattern '{pattern}' in command: {full_cmd[:100]}"
            )

    # Check no path args point outside workspace
    for arg in argv[1:]:
        if arg.startswith("/") and not arg.startswith(str(WORKSPACE)):
            # Allow read-only system paths
            read_only_ok = [
                "/usr/", "/bin/", "/etc/hostname",
                "/proc/meminfo", "/proc/cpuinfo",
            ]
            if not any(arg.startswith(p) for p in read_only_ok):
                raise SecurityError(
                    f"Argument '{arg}' points outside workspace"
                )


# ── Safe operations ────────────────────────────────────
MAX_OUTPUT = 8000


def run_command(
    argv: list[str],
    timeout: int = 60,
    input_data: str = None
) -> tuple[bool, str]:
    """
    Run command safely. No shell=True. Validates argv first.
    Returns (success, output).
    """
    try:
        validate_command(argv)
    except SecurityError as e:
        return False, f"BLOCKED: {e}"

    # Confirmation gate for destructive file operations
    if CONFIRM_REQUIRED:
        executable = Path(argv[0]).name
        if executable in NEEDS_CONFIRM_OPS:
            return False, f"NEEDS_CONFIRM: {' '.join(argv)}"

    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORKSPACE),
            env=safe_env(),
            input=input_data,
        )
        output = (result.stdout + result.stderr).strip()
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + "\n...[output truncated]"
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout}s"
    except Exception as e:
        return False, f"Run error: {e}"


def check_venv() -> bool:
    """Verify venv exists and has required packages."""
    venv_py = Path(VENV_PYTHON)
    if not venv_py.exists():
        return False
    ok, out = run_command(
        [VENV_PYTHON, "-c", "import tavily, pypdf, pandas; print('OK')"],
        timeout=10
    )
    return ok and "OK" in out


def run_python(
    code: str,
    timeout: int = 30
) -> tuple[bool, str]:
    """
    Run Python code safely in venv.
    Code is written to temp file — no shell injection possible.
    """
    builds_dir = WORKSPACE / "builds"
    builds_dir.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py",
        dir=str(builds_dir),
        delete=False
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        return run_command(
            [VENV_PYTHON, tmp_path],
            timeout=timeout
        )
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass


def install_packages(packages: list[str], timeout: int = 120) -> tuple[bool, str]:
    """Install packages into workspace venv only."""
    for pkg in packages:
        if not re.match(r'^[a-zA-Z0-9_\-\.\[\]>=<!]+$', pkg.strip()):
            return False, f"Invalid package name: {pkg}"
    return run_command(
        [VENV_PIP, "install", "--quiet"] + packages,
        timeout=timeout
    )


def write_file(path: str, content: str) -> tuple[bool, str]:
    """Write file — enforces workspace boundary."""
    try:
        p = safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        lines = content.count("\n") + 1
        return True, f"Wrote {p.name} ({lines} lines, {len(content)} chars)"
    except SecurityError as e:
        return False, f"BLOCKED: {e}"
    except Exception as e:
        return False, f"Write error: {e}"


def read_file(path: str) -> tuple[bool, str]:
    """Read file — enforces workspace boundary."""
    try:
        p = safe_path(path)
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > MAX_OUTPUT:
            content = content[:MAX_OUTPUT] + "\n...[truncated]"
        return True, content
    except SecurityError as e:
        return False, f"BLOCKED: {e}"
    except Exception as e:
        return False, f"Read error: {e}"


def read_pdf(path: str) -> tuple[bool, str]:
    """Extract text from PDF."""
    try:
        p = safe_path(path)
        try:
            import pypdf
        except ImportError:
            return False, "pypdf not installed — run: install_packages(['pypdf'])"
        reader = pypdf.PdfReader(str(p))
        text = ""
        for page in reader.pages[:20]:  # Max 20 pages
            text += page.extract_text() + "\n"
        if len(text) > MAX_OUTPUT:
            text = text[:MAX_OUTPUT] + "\n...[truncated]"
        return True, text.strip()
    except SecurityError as e:
        return False, f"BLOCKED: {e}"
    except Exception as e:
        return False, f"PDF error: {e}"


def read_docx(path: str) -> tuple[bool, str]:
    """Extract text from Word document."""
    try:
        p = safe_path(path)
        from docx import Document
        doc = Document(str(p))
        text = "\n".join(para.text for para in doc.paragraphs)
        if len(text) > MAX_OUTPUT:
            text = text[:MAX_OUTPUT] + "\n...[truncated]"
        return True, text.strip()
    except SecurityError as e:
        return False, f"BLOCKED: {e}"
    except Exception as e:
        return False, f"Docx error: {e}"


def read_csv(path: str) -> tuple[bool, str]:
    """Read CSV and return summary + first rows."""
    try:
        p = safe_path(path)
        import pandas as pd
        df = pd.read_csv(str(p))
        summary = (
            f"Shape: {df.shape[0]} rows x {df.shape[1]} cols\n"
            f"Columns: {list(df.columns)}\n"
            f"Types:\n{df.dtypes.to_string()}\n\n"
            f"First 5 rows:\n{df.head().to_string()}"
        )
        return True, summary
    except SecurityError as e:
        return False, f"BLOCKED: {e}"
    except Exception as e:
        return False, f"CSV error: {e}"


def fetch_url(url: str, max_chars: int = 5000) -> tuple[bool, str]:
    """Fetch web page content safely."""
    if not url.startswith(("http://", "https://")):
        return False, f"Invalid URL: {url}"
    blocked_hosts = ["localhost", "127.0.0.1", "0.0.0.0", "169.254"]
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    if any(b in host for b in blocked_hosts):
        return False, f"Blocked host: {host}"
    try:
        import httpx
        from bs4 import BeautifulSoup
        resp = httpx.get(
            url,
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "LUCI-Agent/1.0"}
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return True, text
    except Exception as e:
        return False, f"Fetch error: {e}"


def list_workspace(subpath: str = "") -> tuple[bool, str]:
    """List files in workspace or subdir."""
    try:
        base = safe_path(subpath) if subpath else WORKSPACE
        files = []
        for p in sorted(base.rglob("*")):
            if any(skip in str(p) for skip in
                   ["__pycache__", ".git", ".venv", "node_modules",
                    ".parquet", "builds/"]):
                continue
            rel = p.relative_to(WORKSPACE)
            if p.is_file():
                size = p.stat().st_size
                files.append(f"{rel} ({size:,} bytes)")
        return True, "\n".join(files[:100])
    except SecurityError as e:
        return False, f"BLOCKED: {e}"


def git_operation(args: list[str]) -> tuple[bool, str]:
    """Safe git operations in workspace."""
    allowed_git = {
        "status", "add", "commit", "push", "pull",
        "log", "diff", "branch", "checkout", "show"
    }
    if not args or args[0] not in allowed_git:
        return False, f"Git op '{args[0] if args else ''}' not allowed"
    return run_command(["git"] + args, timeout=30)


def send_telegram_safe(message: str, bot_token: str, chat_id: str) -> tuple[bool, str]:
    """
    Send Telegram message safely.
    Uses environment variables — no string interpolation in commands.
    """
    code = """
import os, urllib.request, urllib.parse, json
token = os.environ['TG_TOKEN']
chat = os.environ['TG_CHAT']
msg = os.environ['TG_MSG']
url = f"https://api.telegram.org/bot{token}/sendMessage"
data = urllib.parse.urlencode({
    "chat_id": chat, "text": msg, "parse_mode": "HTML"
}).encode()
req = urllib.request.Request(url, data=data)
resp = urllib.request.urlopen(req, timeout=10)
print("sent:", resp.status)
"""
    env = {
        **safe_env(),
        "TG_TOKEN": bot_token,
        "TG_CHAT": str(chat_id),
        "TG_MSG": message,
    }
    try:
        result = subprocess.run(
            [VENV_PYTHON, "-c", code],
            capture_output=True, text=True,
            timeout=15, env=env
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)
