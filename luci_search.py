#!/usr/bin/env python3
"""
LUCI Search — Tavily web search.
Uses venv python where needed.
"""
from __future__ import annotations
import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

WORKSPACE = Path("/home/equansa00/beast/workspace")
VENV_PYTHON = str(WORKSPACE / ".venv" / "bin" / "python")


def _load_key() -> str:
    """Load Tavily key from env or .env file."""
    key = os.getenv("TAVILY_API_KEY", "")
    if key and key.startswith("tvly-") and len(key) > 20:
        return key
    for env_path in [
        WORKSPACE / ".env",
        Path(__file__).parent / ".env",
    ]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("TAVILY_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key and key.startswith("tvly-") and len(key) > 20:
                        os.environ["TAVILY_API_KEY"] = key
                        return key
    return ""


TAVILY_API_KEY = _load_key()


def search(query: str, max_results: int = 5) -> dict:
    """
    Search via Tavily. Returns raw response dict.
    Runs in current process if venv is active,
    otherwise spawns venv python subprocess.
    """
    if not TAVILY_API_KEY:
        return {"error": "No Tavily API key", "results": []}

    # Try in current process first
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        return client.search(query, max_results=max_results)
    except ImportError:
        pass

    # Fallback: run in venv python subprocess
    code = f"""
import json, sys
from tavily import TavilyClient
client = TavilyClient(api_key={repr(TAVILY_API_KEY)})
result = client.search({repr(query)}, max_results={max_results})
print(json.dumps(result))
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                     dir=str(WORKSPACE / "builds"),
                                     delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            [VENV_PYTHON, tmp],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout.strip())
        return {"error": result.stderr[:200], "results": []}
    except Exception as e:
        return {"error": str(e), "results": []}
    finally:
        Path(tmp).unlink(missing_ok=True)


def search_and_summarize(query: str) -> str:
    """Search and return formatted string for context injection."""
    response = search(query)
    if "error" in response and not response.get("results"):
        return f"[Search error for '{query}': {response.get('error')}]"

    parts = [f"[WEB SEARCH: {query}]"]
    answer = response.get("answer")
    if answer:
        parts.append(f"Answer: {answer}")
    for i, r in enumerate(response.get("results", [])[:3], 1):
        parts.append(
            f"\nSource {i}: {r.get('title', '')}\n"
            f"{r.get('content', '')[:400]}\n"
            f"URL: {r.get('url', '')}"
        )
    return "\n".join(parts)


def should_search(text: str) -> bool:
    """Return True if query needs live web data."""
    if not TAVILY_API_KEY:
        return False
    triggers = [
        "what is", "who is", "latest", "current", "today",
        "now", "recent", "news", "price", "how to",
        "best", "compare", "vs", "versus", "review",
        "2024", "2025", "2026", "release", "update",
        "stock", "market", "weather", "search",
        "find", "look up", "what are", "tell me about",
    ]
    return any(t in text.lower() for t in triggers)
