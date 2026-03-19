#!/usr/bin/env python3
from pathlib import Path

def get_learn_html() -> str:
    return (Path(__file__).parent / "static" / "learn.html").read_text(encoding="utf-8")
