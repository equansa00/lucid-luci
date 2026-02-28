#!/usr/bin/env python3
"""
LUCI Search — Tavily-powered web search for AI agents.
Gives LUCI ability to find current information autonomously.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import dotenv_values
    _env = dotenv_values(Path(__file__).parent / ".env")
except ImportError:
    _env = {}

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or _env.get("TAVILY_API_KEY", "")


class SearchResult:
    def __init__(self, title: str, url: str, content: str, score: float):
        self.title = title
        self.url = url
        self.content = content
        self.score = score

    def __str__(self):
        return f"[{self.title}]\n{self.content[:500]}\nSource: {self.url}"


def search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = True,
    topic: str = "general"
) -> tuple[Optional[str], list[SearchResult]]:
    """
    Search the web via Tavily.
    Returns (direct_answer, list_of_results).
    direct_answer is Tavily's AI-synthesized answer if available.
    """
    if not TAVILY_API_KEY:
        return None, []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=include_answer,
            topic=topic
        )
        answer = response.get("answer", "")
        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=r.get("score", 0)
            )
            for r in response.get("results", [])
        ]
        return answer or None, results
    except Exception:
        return None, []


def search_and_summarize(query: str) -> str:
    """
    Search and return formatted summary for injection into LUCI's context.
    """
    answer, results = search(query, max_results=5)
    if not results and not answer:
        return f"[Search returned no results for: {query}]"

    parts = [f"[WEB SEARCH: {query}]"]
    if answer:
        parts.append(f"Summary: {answer}")
    for i, r in enumerate(results[:3], 1):
        parts.append(f"\nSource {i}: {r.title}\n{r.content[:400]}\nURL: {r.url}")
    return "\n".join(parts)


def should_search(text: str) -> bool:
    """
    Decide if LUCI should search the web for this query.
    True if query needs current/external information.
    """
    # Don't search if no API key configured
    if not TAVILY_API_KEY:
        return False

    search_triggers = [
        "what is", "who is", "latest", "current", "today",
        "now", "recent", "news", "price", "how to",
        "best", "compare", "vs", "versus", "review",
        "2024", "2025", "2026", "release", "update",
        "documentation", "docs", "api", "library",
        "stock", "market", "weather", "search",
        "find", "look up", "what are", "tell me about",
    ]
    text_lower = text.lower()
    return any(t in text_lower for t in search_triggers)
