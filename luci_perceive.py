#!/usr/bin/env python3
"""
LUCI Perceive — understand any file type.
PDF, Word, CSV, Excel, images, web pages, code.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from luci_sandbox import (
    safe_path, read_file, read_pdf,
    read_docx, read_csv, fetch_url,
    SecurityError
)

WORKSPACE = Path("/home/equansa00/beast/workspace").resolve()


def perceive_file(path: str) -> tuple[str, str]:
    """
    Auto-detect file type and extract content.
    Returns (file_type, content).
    """
    try:
        p = safe_path(path)
    except SecurityError as e:
        return "error", str(e)

    suffix = p.suffix.lower()

    if suffix == ".pdf":
        ok, content = read_pdf(path)
        return "pdf", content if ok else f"PDF error: {content}"

    elif suffix in (".docx", ".doc"):
        ok, content = read_docx(path)
        return "word", content if ok else f"Word error: {content}"

    elif suffix == ".csv":
        ok, content = read_csv(path)
        return "csv", content if ok else f"CSV error: {content}"

    elif suffix in (".xlsx", ".xls"):
        try:
            import pandas as pd
            df = pd.read_excel(str(p))
            content = (
                f"Shape: {df.shape}\nColumns: {list(df.columns)}\n"
                f"First 5 rows:\n{df.head().to_string()}"
            )
            return "excel", content
        except Exception as e:
            return "excel", f"Excel error: {e}"

    elif suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
        return "image", describe_image(str(p))

    elif suffix in (".py", ".js", ".ts", ".go", ".rs", ".sh",
                    ".yaml", ".yml", ".json", ".toml", ".md",
                    ".txt", ".html", ".css", ".sql"):
        ok, content = read_file(path)
        file_type = "code" if suffix == ".py" else "text"
        return file_type, content if ok else f"Read error: {content}"

    else:
        ok, content = read_file(path)
        return "text", content if ok else f"Read error: {content}"


def describe_image(path: str) -> str:
    """
    Describe image using llava vision model via Ollama.
    Falls back to metadata if model unavailable.
    """
    import base64, json, urllib.request
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        payload = json.dumps({
            "model": "llava:latest",
            "messages": [{
                "role": "user",
                "content": "Describe this image in detail. "
                           "Include all visible text, objects, layout.",
                "images": [b64]
            }],
            "stream": False,
            "options": {"num_predict": 512}
        }).encode()

        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["message"]["content"]
    except Exception:
        # Fallback: basic metadata
        try:
            from PIL import Image
            img = Image.open(path)
            return (f"Image: {img.format} {img.mode} "
                    f"{img.width}x{img.height}px")
        except Exception as e:
            return f"Image at {path} (could not describe: {e})"


def perceive_url(url: str) -> tuple[str, str]:
    """Fetch and understand a web page."""
    ok, content = fetch_url(url)
    return "webpage", content if ok else f"Fetch error: {content}"


def summarize_perception(file_type: str, content: str,
                          filename: str = "") -> str:
    """Format perception for injection into LUCI's context."""
    header = f"[FILE: {filename} | TYPE: {file_type.upper()}]\n"
    return header + content[:6000]
