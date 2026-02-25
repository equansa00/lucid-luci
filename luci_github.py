#!/usr/bin/env python3
"""
LUCI GitHub Integration
Scan, review, and auto-fix GitHub repositories via LLM.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path.home() / "beast" / "workspace" / ".env")

from luci import ollama_chat, ROUTER_CODE_MODEL  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "")
REPOS_DIR = Path.home() / "beast" / "workspace" / "repos"
REVIEWS_DIR = Path.home() / "beast" / "workspace" / "reviews"

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".mypy_cache", ".pytest_cache",
}


# ---------------------------------------------------------------------------
# 1. GitHub client
# ---------------------------------------------------------------------------

def github_client():
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN not set in .env")
    from github import Github
    return Github(GITHUB_TOKEN)


# ---------------------------------------------------------------------------
# 2. List repos
# ---------------------------------------------------------------------------

def github_list_repos() -> list[dict]:
    g = github_client()
    user = g.get_user(GITHUB_USERNAME) if GITHUB_USERNAME else g.get_user()
    repos = []
    for r in user.get_repos():
        repos.append({
            "name": r.name,
            "full_name": r.full_name,
            "description": r.description or "",
            "language": r.language or "unknown",
            "updated_at": r.updated_at.isoformat() if r.updated_at else "",
            "stars": r.stargazers_count,
            "open_issues": r.open_issues_count,
            "private": r.private,
            "url": r.html_url,
            "default_branch": r.default_branch,
        })
    repos.sort(key=lambda x: x["updated_at"], reverse=True)
    return repos


# ---------------------------------------------------------------------------
# 3. Single repo summary
# ---------------------------------------------------------------------------

def github_repo_summary(repo_name: str) -> dict:
    g = github_client()
    username = GITHUB_USERNAME or g.get_user().login
    r = g.get_repo(f"{username}/{repo_name}")
    return {
        "name": r.name,
        "full_name": r.full_name,
        "description": r.description or "",
        "language": r.language or "unknown",
        "updated_at": r.updated_at.isoformat() if r.updated_at else "",
        "stars": r.stargazers_count,
        "open_issues": r.open_issues_count,
        "private": r.private,
        "url": r.html_url,
        "default_branch": r.default_branch,
        "top_languages": dict(r.get_languages()),
        "topics": list(r.get_topics()),
        "size_kb": r.size,
    }


# ---------------------------------------------------------------------------
# 4. Clone or pull
# ---------------------------------------------------------------------------

def clone_or_pull_repo(repo_name: str) -> Path:
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = REPOS_DIR / repo_name
    username = GITHUB_USERNAME

    if local_path.exists():
        subprocess.run(
            ["git", "pull"],
            cwd=str(local_path),
            capture_output=True,
            timeout=60,
        )
    else:
        url = f"https://{GITHUB_TOKEN}@github.com/{username}/{repo_name}.git"
        subprocess.run(
            ["git", "clone", url, str(local_path)],
            capture_output=True,
            timeout=120,
            check=True,
        )
    return local_path


# ---------------------------------------------------------------------------
# 5. Static analysis
# ---------------------------------------------------------------------------

def analyze_repo(repo_path: Path) -> dict:
    file_count = 0
    languages: dict[str, int] = {}
    total_lines = 0
    issues: list[dict] = []
    todo_count = 0
    large_files: list[str] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            fpath = Path(root) / fname
            rel = str(fpath.relative_to(repo_path))
            ext = fpath.suffix.lower()
            file_count += 1
            languages[ext] = languages.get(ext, 0) + 1

            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            lines = text.splitlines()
            total_lines += len(lines)

            if len(lines) > 500:
                large_files.append(rel)

            # TODO/FIXME count
            for line in lines:
                if "TODO" in line or "FIXME" in line:
                    todo_count += 1

            # Python syntax check
            if ext == ".py":
                try:
                    ast.parse(text)
                except SyntaxError as e:
                    issues.append({
                        "file": rel,
                        "issue_type": "syntax_error",
                        "detail": str(e),
                    })
                # Bare except
                for i, line in enumerate(lines, 1):
                    if re.search(r"except\s*:", line):
                        issues.append({
                            "file": rel,
                            "issue_type": "bare_except",
                            "detail": f"line {i}: {line.strip()}",
                        })

            # Hardcoded passwords (all text files)
            for i, line in enumerate(lines, 1):
                if re.search(r"password\s*=\s*['\"][^'\"]{4,}", line, re.IGNORECASE):
                    issues.append({
                        "file": rel,
                        "issue_type": "hardcoded_password",
                        "detail": f"line {i}: potential hardcoded credential",
                    })

    return {
        "file_count": file_count,
        "languages": languages,
        "total_lines": total_lines,
        "issues": issues,
        "todo_count": todo_count,
        "has_requirements": (repo_path / "requirements.txt").exists()
            and (repo_path / "requirements.txt").stat().st_size > 0,
        "has_package_json": (repo_path / "package.json").exists(),
        "has_dockerfile": (repo_path / "Dockerfile").exists(),
        "has_readme": any(
            (repo_path / f).exists()
            for f in ("README.md", "readme.md", "README.txt", "README")
        ),
        "large_files": large_files,
    }


# ---------------------------------------------------------------------------
# 6. LLM review
# ---------------------------------------------------------------------------

_KEY_FILES = [
    "README.md", "main.py", "app.py", "index.js", "index.ts",
    "package.json", "requirements.txt",
    "src/App.js", "src/App.tsx",
]


def llm_review_repo(repo_path: Path, analysis: dict) -> str:
    file_sections = []
    for rel in _KEY_FILES:
        fpath = repo_path / rel
        if fpath.exists():
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")[:3000]
                file_sections.append(f"=== {rel} ===\n{content}")
            except Exception:
                pass
        if len(file_sections) >= 5:
            break

    issues_text = (
        "\n".join(
            f"{i+1}. [{r['issue_type']}] {r['file']}: {r['detail']}"
            for i, r in enumerate(analysis["issues"])
        )
        or "None found"
    )

    lang_str = ", ".join(
        f"{ext}:{n}" for ext, n in
        sorted(analysis["languages"].items(), key=lambda x: -x[1])[:8]
    )

    prompt = (
        "You are a senior code reviewer. Review this codebase.\n\n"
        f"Repository stats:\n"
        f"- Files: {analysis['file_count']}\n"
        f"- Languages: {lang_str}\n"
        f"- Total lines: {analysis['total_lines']}\n"
        f"- TODOs/FIXMEs: {analysis['todo_count']}\n\n"
        f"Static analysis found these issues:\n{issues_text}\n\n"
        "Key files:\n"
        + "\n\n".join(file_sections)
        + "\n\nProvide a structured review:\n"
        "1. Overall quality score (1-10) — put score on its own line as 'Score: X/10'\n"
        "2. Top 3 critical issues to fix\n"
        "3. Top 3 recommended improvements\n"
        "4. Security concerns (if any)\n"
        "5. One-paragraph summary\n\n"
        "Be specific and actionable."
    )
    return ollama_chat([{"role": "user", "content": prompt}], model=ROUTER_CODE_MODEL)


# ---------------------------------------------------------------------------
# 7. Save review
# ---------------------------------------------------------------------------

def save_review(repo_name: str, analysis: dict, llm_review: str) -> Path:
    review_dir = REVIEWS_DIR / repo_name
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / "LUCI_REVIEW.md"

    lang_str = ", ".join(
        f"{ext}:{n}" for ext, n in
        sorted(analysis["languages"].items(), key=lambda x: -x[1])[:8]
    )

    issues_md = (
        "\n".join(
            f"- [{r['issue_type']}] `{r['file']}`: {r['detail']}"
            for r in analysis["issues"]
        )
        or "No issues found"
    )

    large_md = (
        "\n".join(f"- {f}" for f in analysis["large_files"])
        or "None"
    )

    content = (
        f"# LUCI Review: {repo_name}\n"
        f"Generated: {datetime.now().isoformat()}\n\n"
        f"## Repository Stats\n"
        f"- Files: {analysis['file_count']}\n"
        f"- Languages: {lang_str}\n"
        f"- Total lines: {analysis['total_lines']}\n"
        f"- TODOs/FIXMEs: {analysis['todo_count']}\n\n"
        f"## Static Analysis Issues\n{issues_md}\n\n"
        f"## Large Files (>500 lines)\n{large_md}\n\n"
        f"## LLM Review\n{llm_review}\n"
    )
    review_path.write_text(content, encoding="utf-8")
    return review_path


# ---------------------------------------------------------------------------
# 8. Parse quality score
# ---------------------------------------------------------------------------

def parse_quality_score(llm_review: str) -> int:
    m = re.search(r"Score:\s*(\d+)\s*/\s*10", llm_review, re.IGNORECASE)
    if m:
        return min(10, max(1, int(m.group(1))))
    return 5


# ---------------------------------------------------------------------------
# 9. Scan one repo
# ---------------------------------------------------------------------------

def scan_one_repo(repo_name: str) -> dict:
    try:
        repo_path = clone_or_pull_repo(repo_name)
        analysis = analyze_repo(repo_path)
        llm_review = llm_review_repo(repo_path, analysis)
        review_path = save_review(repo_name, analysis, llm_review)

        # Most common language
        lang = "unknown"
        if analysis["languages"]:
            lang = max(analysis["languages"], key=lambda k: analysis["languages"][k])

        return {
            "repo_name": repo_name,
            "review_path": str(review_path),
            "quality_score": parse_quality_score(llm_review),
            "issue_count": len(analysis["issues"]),
            "language": lang,
            "status": "ok",
        }
    except Exception as e:
        return {
            "repo_name": repo_name,
            "status": "error",
            "error": str(e),
            "quality_score": 0,
            "issue_count": 0,
            "language": "unknown",
            "review_path": "",
        }


# ---------------------------------------------------------------------------
# 10. Scan all repos
# ---------------------------------------------------------------------------

def scan_all_repos(progress_callback=None) -> list[dict]:
    repos = github_list_repos()
    total = len(repos)
    results = []
    for i, repo in enumerate(repos):
        name = repo["name"]
        if progress_callback:
            progress_callback(name, i + 1, total, "scanning")
        result = scan_one_repo(name)
        if progress_callback:
            progress_callback(name, i + 1, total, "done")
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# 11. Apply LUCI fix
# ---------------------------------------------------------------------------

def apply_luci_fix(repo_name: str, fix_description: str) -> dict:
    repo_path = clone_or_pull_repo(repo_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = f"luci/fixes-{ts}"

    subprocess.run(
        ["git", "checkout", "-b", branch],
        cwd=str(repo_path),
        capture_output=True,
        check=True,
    )

    # Build context from existing review if available
    review_path = REVIEWS_DIR / repo_name / "LUCI_REVIEW.md"
    if review_path.exists():
        context_text = review_path.read_text(encoding="utf-8", errors="ignore")[:2000]
    else:
        analysis = analyze_repo(repo_path)
        lang_str = ", ".join(
            f"{ext}:{n}" for ext, n in
            sorted(analysis["languages"].items(), key=lambda x: -x[1])[:5]
        )
        context_text = (
            f"Files: {analysis['file_count']}, Languages: {lang_str}, "
            f"Lines: {analysis['total_lines']}, Issues: {len(analysis['issues'])}"
        )

    # Read key files
    file_sections = []
    for rel in _KEY_FILES:
        fpath = repo_path / rel
        if fpath.exists():
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")[:3000]
                file_sections.append(f"=== {rel} ===\n{content}")
            except Exception:
                pass
        if len(file_sections) >= 5:
            break

    prompt = (
        "You are an expert software engineer.\n"
        f"Fix the following in this repository: {fix_description}\n\n"
        f"Repository context:\n{context_text}\n\n"
        "Key files:\n"
        + "\n\n".join(file_sections)
        + "\n\nProvide fixes as complete file contents using this exact format:\n"
        "```path/to/file.py\n"
        "[complete file content here]\n"
        "```\n\n"
        "Only include files you are actually changing.\n"
        "Explain each change briefly after the code blocks."
    )

    llm_response = ollama_chat(
        [{"role": "user", "content": prompt}],
        model=ROUTER_CODE_MODEL,
    )

    # Parse and write changed files
    written_paths = []
    for m in re.finditer(r"```([^\n`]+)\n(.*?)```", llm_response, re.DOTALL):
        filepath = m.group(1).strip()
        file_content = m.group(2)
        if not filepath or "/" not in filepath and "." not in filepath:
            continue
        target = repo_path / filepath
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file_content, encoding="utf-8")
            written_paths.append(filepath)
        except Exception:
            pass

    subprocess.run(["git", "add", "-A"], cwd=str(repo_path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", f"LUCI: {fix_description}"],
        cwd=str(repo_path),
        capture_output=True,
    )

    push_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{repo_name}.git"
    subprocess.run(
        ["git", "push", push_url, branch],
        cwd=str(repo_path),
        capture_output=True,
        check=True,
    )

    g = github_client()
    gh_repo = g.get_repo(f"{GITHUB_USERNAME}/{repo_name}")
    pr = gh_repo.create_pull(
        title=f"LUCI Fix: {fix_description}",
        body=llm_response,
        head=branch,
        base=gh_repo.default_branch,
    )

    return {
        "branch": branch,
        "pr_url": pr.html_url,
        "files_changed": written_paths,
        "commit_sha": "",
    }


# ---------------------------------------------------------------------------
# 12. Format repo list
# ---------------------------------------------------------------------------

def format_repo_list(repos: list[dict]) -> str:
    total = len(repos)
    lines = [f"📦 Your GitHub Repos ({total} total)\n"]
    for i, r in enumerate(repos[:20], 1):
        flag = " 🔴" if r["open_issues"] > 0 else ""
        date = r["updated_at"][:10] if r["updated_at"] else "unknown"
        lines.append(f"{i}. {r['name']} [{r['language']}] — {date}{flag}")
    if total > 20:
        lines.append(f"\n...and {total - 20} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 13. Format scan summary
# ---------------------------------------------------------------------------

def format_scan_summary(results: list[dict]) -> str:
    total = len(results)
    errors = [r for r in results if r["status"] == "error"]
    ok = [r for r in results if r["status"] == "ok"]
    ok_sorted = sorted(ok, key=lambda x: x["quality_score"])

    lines = [
        f"🔍 LUCI Scan Complete — {total} repos",
        f"✅ Scanned: {len(ok)}  ❌ Errors: {len(errors)}\n",
        "Needs attention (worst first):",
    ]
    for i, r in enumerate(ok_sorted[:10], 1):
        lines.append(
            f"{i}. {r['repo_name']} — {r['quality_score']}/10, "
            f"{r['issue_count']} issues [{r['language']}]"
        )
    lines.append("\nUse /github fix <repo> <description> to start fixing.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_dotenv(Path.home() / "beast" / "workspace" / ".env")

    if len(sys.argv) < 2:
        print("Usage: python3.14 luci_github.py [list|scan|scan <repo>|fix <repo> <desc>]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "list":
        repos = github_list_repos()
        print(f"Found {len(repos)} repos:")
        for r in repos:
            issues = " 🔴" if r["open_issues"] > 0 else ""
            print(f"  {r['name']} [{r['language']}] — {r['updated_at'][:10]}{issues}")

    elif cmd == "scan" and len(sys.argv) == 2:
        print("Starting full scan...")

        def cb(name, i, total, status):
            print(f"[{i}/{total}] {name}: {status}", flush=True)

        results = scan_all_repos(progress_callback=cb)
        print(format_scan_summary(results))
        out = Path.home() / "beast" / "workspace" / "reviews" / "scan_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nFull results saved to: {out}")

    elif cmd == "scan" and len(sys.argv) == 3:
        result = scan_one_repo(sys.argv[2])
        print(json.dumps(result, indent=2, default=str))
        if result["review_path"]:
            print("\n--- REVIEW ---")
            print(Path(result["review_path"]).read_text())

    elif cmd == "fix" and len(sys.argv) >= 4:
        repo = sys.argv[2]
        desc = " ".join(sys.argv[3:])
        print(f"Applying fix to {repo}: {desc}")
        result = apply_luci_fix(repo, desc)
        print(f"✅ PR opened: {result['pr_url']}")
        print(f"Branch: {result['branch']}")

    else:
        print("Unknown command. Use: list, scan, scan <repo>, fix <repo> <desc>")
