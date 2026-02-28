#!/usr/bin/env python3
"""
LUCI Builder v2 — uses secure sandbox + full cognitive loop.
Entry point for autonomous feature building.
"""
from __future__ import annotations
import os
import sys
import json
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

WORKSPACE = Path("/home/equansa00/beast/workspace")

try:
    from dotenv import dotenv_values
    _env = dotenv_values(WORKSPACE / ".env")
except ImportError:
    _env = {}

BOT_TOKEN = os.getenv("BOT_TOKEN") or _env.get("BOT_TOKEN", "")
CHAT_ID = os.getenv("ALLOWED_USER_ID") or _env.get("ALLOWED_USER_ID", "")

# Active builds registry
_builds: dict[str, dict] = {}


def start_build(
    request: str,
    on_update: Optional[Callable] = None
) -> str:
    """Start async build. Returns build_id."""
    from luci_think import LUCIThink

    build_id = str(uuid.uuid4())[:8]
    updates = []
    _builds[build_id] = {
        "request": request,
        "updates": updates,
        "done": False,
        "result": ""
    }

    def _update(msg: str):
        updates.append(msg)
        if on_update:
            on_update(msg)
        # Telegram updates for major milestones
        if any(k in msg for k in ["✅", "❌", "DONE", "[Step 1]"]):
            from luci_sandbox import send_telegram_safe
            send_telegram_safe(
                f"🔨 [{build_id}] {msg[:200]}",
                BOT_TOKEN, CHAT_ID
            )

    def _run():
        try:
            thinker = LUCIThink(
                on_update=_update,
                bot_token=BOT_TOKEN,
                chat_id=CHAT_ID
            )
            result = thinker.run(request)
            _builds[build_id]["done"] = True
            _builds[build_id]["result"] = result
            from luci_sandbox import send_telegram_safe
            send_telegram_safe(
                f"✅ Build {build_id} complete!\n{result[:400]}",
                BOT_TOKEN, CHAT_ID
            )
        except Exception as e:
            err = f"Build {build_id} failed: {e}"
            _builds[build_id]["done"] = True
            _builds[build_id]["result"] = err
            _update(f"❌ {err}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return build_id


def get_build_status(build_id: str) -> dict:
    """Return current status of a build."""
    b = _builds.get(build_id)
    if not b:
        return {"error": "Build not found"}
    return {
        "build_id": build_id,
        "request": b["request"][:200],
        "done": b["done"],
        "result": b.get("result", ""),
        "updates": b["updates"][-20:],  # last 20 updates
        "total_updates": len(b["updates"]),
    }


def build_sync(request: str) -> str:
    """Synchronous build — for CLI use."""
    from luci_think import LUCIThink
    thinker = LUCIThink(
        on_update=print,
        bot_token=BOT_TOKEN,
        chat_id=CHAT_ID
    )
    return thinker.run(request)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3.14 luci_builder.py 'feature request'")
        sys.exit(1)
    request = " ".join(sys.argv[1:])
    result = build_sync(request)
    print(f"\n✅ Result: {result}")
