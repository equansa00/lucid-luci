#!/usr/bin/env python3
"""
LUCI Self-Diagnosis System
Captures frontend errors, backend exceptions, and systemd logs.
LUCI analyzes them and suggests fixes. Separate from self-audit.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

WORKSPACE  = Path(__file__).parent
ERROR_LOG  = WORKSPACE / "runs" / "error_log.json"
DIAG_LOG   = WORKSPACE / "runs" / "diagnosis_log.json"


def log_error(error: dict) -> dict:
    """Store an incoming error report."""
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    if ERROR_LOG.exists():
        try:
            errors = json.loads(ERROR_LOG.read_text())
        except Exception:
            errors = []

    entry = {
        "id":        len(errors) + 1,
        "timestamp": datetime.now().isoformat(),
        "resolved":  False,
        **error
    }
    errors.append(entry)
    # Keep last 200 errors
    ERROR_LOG.write_text(json.dumps(errors[-200:], indent=2))
    return entry


def get_recent_errors(n: int = 10) -> list:
    if not ERROR_LOG.exists():
        return []
    try:
        errors = json.loads(ERROR_LOG.read_text())
        return errors[-n:]
    except Exception:
        return []


def get_systemd_errors(service: str = "luci-web", n: int = 20) -> str:
    """Get recent errors from a systemd service journal."""
    try:
        result = subprocess.run(
            ["journalctl", "--user", "-u", service,
             "-n", str(n), "--no-pager", "-p", "err..warning"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or "No recent errors"
    except Exception as e:
        return f"Could not read journal: {e}"


def get_all_service_errors() -> dict:
    """Get errors from all LUCI services."""
    services = ["luci", "luci-web", "luci-whisper", "luci-wake"]
    return {svc: get_systemd_errors(svc) for svc in services}


def build_diagnosis_prompt(errors: list, service_logs: dict) -> str:
    """Build a prompt for LUCI to diagnose errors."""
    recent = json.dumps(errors[-5:], indent=2) if errors else "None"
    logs_summary = "\n".join(
        f"[{svc}]: {log[:500]}"
        for svc, log in service_logs.items()
        if "No recent" not in log and "Could not" not in log
    ) or "No service errors"

    return f"""You are LUCI diagnosing errors in your own codebase.

RECENT FRONTEND ERRORS:
{recent}

RECENT SERVICE LOG ERRORS:
{logs_summary}

YOUR TASK:
1. Identify the root cause of each error
2. Classify it: SYNTAX | RUNTIME | NETWORK | STATE | CONFIG
3. Suggest a specific fix with the exact file and line if possible
4. Rate severity: CRITICAL | WARNING | INFO
5. If you can generate a patch command, do so

Focus on actionable fixes. Reference actual file paths in ~/beast/workspace/.
Be direct — no preamble, just diagnosis and fix."""


def save_diagnosis(error_ids: list, diagnosis: str) -> None:
    """Save a diagnosis result."""
    DIAG_LOG.parent.mkdir(parents=True, exist_ok=True)
    diags = []
    if DIAG_LOG.exists():
        try:
            diags = json.loads(DIAG_LOG.read_text())
        except Exception:
            diags = []

    diags.append({
        "timestamp":  datetime.now().isoformat(),
        "error_ids":  error_ids,
        "diagnosis":  diagnosis,
    })
    DIAG_LOG.write_text(json.dumps(diags[-100:], indent=2))


def get_unresolved_count() -> int:
    errors = get_recent_errors(200)
    return sum(1 for e in errors if not e.get("resolved"))


if __name__ == "__main__":
    print("Recent errors:", len(get_recent_errors()))
    print("Unresolved:", get_unresolved_count())
