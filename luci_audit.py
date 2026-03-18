#!/usr/bin/env python3
"""
LUCI Self-Audit System
Scans the workspace, understands every file, cross-references
what's wired vs what exists, flags issues, confirms health.
"""

import ast
import json
import os
import re
import subprocess
import sys
import importlib.util
from pathlib import Path
from typing import Any

WORKSPACE = Path(os.path.expanduser("~/beast/workspace"))

# Files/dirs to skip entirely
SKIP_DIRS  = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache", "runs", "luci_faces", "piper", "builds", "tests"}
SKIP_FILES = {".env", "gmail_token.json", "gmail_credentials.json"}
SKIP_EXTS  = {".pyc", ".pyo", ".onnx", ".db", ".log", ".wav", ".mp3", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".db-shm", ".db-wal"}

# ── Core files we know should exist and what they provide ──────────────────
# Files that are standalone scripts or utilities — not imported, but intentional
KNOWN_STANDALONE = {
    'generate_luci_face.py': 'Standalone face generation utility',
    'repo/app.py':           'Test repo file for mini-BEAST',
    'mini_beast.py':         'Standalone CLI tool — run directly',
    'polymarket.py':         'Standalone CLI + importable module',
    'luci_audit.py':         'This audit script — run directly',
}

KNOWN_CORE = {
    "luci.py":              "Main agent — Telegram bot, routing, all core capabilities",
    "luci_web.py":          "Web UI server — FastAPI, chat endpoint, voice, overlays",
    "luci_features.json":   "Feature registry — auto-populated capability list",
    "persona_agent.txt":    "LUCI persona and identity",
    "beast_memory.json":    "Persistent memory store",
    ".env":                 "Environment variables and API keys",
    "polymarket.py":        "Polymarket integration — markets, positions, orders",
    "luci_trading_agent.py":"Trading agent — /trade commands, briefing block",
    "luci_trading_alpaca.py":"Alpaca execution layer — buy/sell/positions",
    "mini_beast.py":        "Hardened local dev agent (mini-BEAST)",
}

KNOWN_DIRS = {
    "luci_trading":  "Trading research engine — scanner, backtest, strategies",
    "luci_code":     "LUCI Code CLI — Claude Code equivalent",
    "docs":          "RAG document store",
    "runs":          "Agent run artifacts",
}


# ── File scanner ───────────────────────────────────────────────────────────
def scan_workspace() -> dict[str, Any]:
    files = {}
    for p in sorted(WORKSPACE.rglob("*")):
        if not p.is_file():
            continue
        # Skip by directory
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        if p.name in SKIP_FILES:
            continue
        if p.suffix.lower() in SKIP_EXTS:
            continue
        rel = str(p.relative_to(WORKSPACE))
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        files[rel] = {"path": str(p), "size": size, "ext": p.suffix.lower()}
    return files


# ── Python AST analysis ────────────────────────────────────────────────────
def analyze_python(path: Path) -> dict[str, Any]:
    result = {
        "imports": [],
        "functions": [],
        "classes": [],
        "top_level_calls": [],
        "syntax_ok": True,
        "syntax_error": None,
    }
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
        result["syntax_ok"] = True
    except SyntaxError as e:
        result["syntax_ok"] = False
        result["syntax_error"] = f"line {e.lineno}: {e.msg}"
        return result
    except Exception as e:
        result["syntax_ok"] = False
        result["syntax_error"] = str(e)
        return result

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append(alias.name)
            else:
                mod = node.module or ""
                result["imports"].append(mod)
        elif isinstance(node, ast.FunctionDef):
            result["functions"].append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            result["functions"].append(f"async:{node.name}")
        elif isinstance(node, ast.ClassDef):
            result["classes"].append(node.name)

    return result


# ── Cross-reference: what imports what ────────────────────────────────────
def build_import_graph(files: dict) -> dict[str, list[str]]:
    graph = {}
    workspace_modules = set()

    # Collect all local module names
    for rel in files:
        if rel.endswith(".py"):
            mod = Path(rel).stem
            workspace_modules.add(mod)

    for rel, info in files.items():
        if info["ext"] != ".py":
            continue
        analysis = analyze_python(Path(info["path"]))
        local_imports = []
        for imp in analysis["imports"]:
            base = imp.split(".")[0]
            if base in workspace_modules:
                local_imports.append(base)
        graph[rel] = local_imports

    return graph


# ── Service file checker ───────────────────────────────────────────────────
def check_services() -> dict[str, Any]:
    services = {}
    service_dir = Path.home() / ".config" / "systemd" / "user"
    if not service_dir.exists():
        return {}

    for sf in service_dir.glob("luci*.service"):
        try:
            result = subprocess.run(
                ["systemctl", "--user", "is-active", sf.stem],
                capture_output=True, text=True, timeout=5
            )
            status = result.stdout.strip()
        except Exception:
            status = "unknown"

        content = sf.read_text(errors="ignore")
        # Extract the actual binary from ExecStart (first token only)
        exec_match = re.search(r"ExecStart=(\S+)", content)
        target_file = exec_match.group(1) if exec_match else None
        # Strip environment variable wrappers like /usr/bin/env
        if target_file and target_file.endswith('/env'):
            env_match = re.search(r"ExecStart=\S+\s+(\S+)", content)
            target_file = env_match.group(1) if env_match else target_file
        target_exists = Path(target_file).exists() if target_file else False

        services[sf.stem] = {
            "status": status,
            "target_file": target_file,
            "target_exists": target_exists,
            "service_file": str(sf),
        }
    return services


# ── JSON file health ───────────────────────────────────────────────────────
def check_json_files(files: dict) -> dict[str, Any]:
    results = {}
    for rel, info in files.items():
        if info["ext"] not in (".json",):
            continue
        try:
            data = json.loads(Path(info["path"]).read_text(encoding="utf-8", errors="ignore"))
            results[rel] = {"valid": True, "type": type(data).__name__}
        except json.JSONDecodeError as e:
            results[rel] = {"valid": False, "error": str(e)}
    return results


# ── Env var checker ────────────────────────────────────────────────────────
def check_env() -> dict[str, Any]:
    env_path = WORKSPACE / ".env"
    if not env_path.exists():
        return {"exists": False, "keys": []}

    keys = []
    missing_values = []
    for line in env_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            keys.append(k.strip())
            if not v.strip():
                missing_values.append(k.strip())

    return {"exists": True, "keys": keys, "missing_values": missing_values}


# ── Dead file detector ─────────────────────────────────────────────────────
def find_dead_files(files: dict, import_graph: dict) -> list[dict]:
    """Files that exist but are never imported by anything."""
    all_imported = set()
    for imports in import_graph.values():
        all_imported.update(imports)

    dead = []
    for rel, info in files.items():
        if info["ext"] != ".py":
            continue
        if Path(rel).parts[0] in SKIP_DIRS:
            continue
        mod = Path(rel).stem
        # Check if it's imported anywhere
        imported = mod in all_imported
        # Check if it's a known entry point
        is_entry = rel in KNOWN_CORE or rel.startswith("luci_")
        # Check if it has a main block
        try:
            src = Path(info["path"]).read_text(errors="ignore")
            has_main = '__main__' in src or 'def main' in src
        except Exception:
            has_main = False

        is_standalone = rel in KNOWN_STANDALONE
        if not imported and not is_entry and not has_main and not is_standalone:
            dead.append({
                "file": rel,
                "size": info["size"],
                "note": "not imported by any other file and not a known entry point"
            })

    return dead


# ── Missing expected files ─────────────────────────────────────────────────
def find_missing_core(files: dict) -> list[dict]:
    missing = []
    for fname, desc in KNOWN_CORE.items():
        if fname not in files and not (WORKSPACE / fname).exists():
            missing.append({"file": fname, "description": desc})
    return missing


# ── Syntax errors ──────────────────────────────────────────────────────────
def find_syntax_errors(files: dict) -> list[dict]:
    errors = []
    for rel, info in files.items():
        if info["ext"] != ".py":
            continue
        result = analyze_python(Path(info["path"]))
        if not result["syntax_ok"]:
            errors.append({"file": rel, "error": result["syntax_error"]})
    return errors


# ── Main audit ─────────────────────────────────────────────────────────────
def run_audit(verbose: bool = False) -> dict[str, Any]:
    print("🔍 Scanning workspace...", flush=True)
    files = scan_workspace()
    print(f"   Found {len(files)} files", flush=True)

    print("🔗 Building import graph...", flush=True)
    import_graph = build_import_graph(files)

    print("⚙️  Checking services...", flush=True)
    services = check_services()

    print("📋 Checking JSON files...", flush=True)
    json_health = check_json_files(files)

    print("🔑 Checking .env...", flush=True)
    env = check_env()

    print("💀 Finding dead files...", flush=True)
    dead = find_dead_files(files, import_graph)

    print("❌ Finding syntax errors...", flush=True)
    syntax_errors = find_syntax_errors(files)

    print("🔎 Checking for missing core files...", flush=True)
    missing_core = find_missing_core(files)

    # ── Build report ──────────────────────────────────────────────────────
    issues = []
    recommendations = []

    # Syntax errors are critical
    for e in syntax_errors:
        issues.append({"severity": "CRITICAL", "category": "syntax", "file": e["file"], "detail": e["error"]})

    # Missing core files
    for m in missing_core:
        issues.append({"severity": "WARNING", "category": "missing_core", "file": m["file"], "detail": m["description"]})

    # Dead files
    for d in dead:
        issues.append({"severity": "INFO", "category": "dead_file", "file": d["file"], "detail": d["note"]})

    # Service health
    service_dir = Path.home() / ".config" / "systemd" / "user"
    for svc, info in services.items():
        # Oneshot or timer-driven services are expected to be inactive
        svc_content = open(info["service_file"]).read() if info.get("service_file") else ""
        is_oneshot = "oneshot" in svc_content.lower()
        is_timer_driven = (service_dir / (svc + ".timer")).exists()
        if info["status"] not in ("active", "activating") and not is_oneshot and not is_timer_driven:
            issues.append({"severity": "WARNING", "category": "service_down", "file": svc, "detail": f"status={info['status']}"})
        if not info["target_exists"] and info["target_file"]:
            issues.append({"severity": "CRITICAL", "category": "service_missing_file", "file": svc, "detail": f"ExecStart points to missing file: {info['target_file']}"})

    # JSON validity
    for rel, jh in json_health.items():
        if not jh["valid"]:
            issues.append({"severity": "CRITICAL", "category": "invalid_json", "file": rel, "detail": jh["error"]})

    # .env completeness
    if not env["exists"]:
        issues.append({"severity": "CRITICAL", "category": "missing_env", "file": ".env", "detail": "No .env file found"})
    elif env["missing_values"]:
        for k in env["missing_values"]:
            issues.append({"severity": "WARNING", "category": "empty_env_var", "file": ".env", "detail": f"{k} is set but has no value"})

    # Recommendations
    if not (WORKSPACE / "luci_features.json").exists():
        recommendations.append("Create luci_features.json — auto-populates LUCI's capability awareness")
    if syntax_errors:
        recommendations.append(f"Fix {len(syntax_errors)} syntax error(s) — these files cannot be loaded")
    if dead:
        recommendations.append(f"Review {len(dead)} potentially unused file(s) — confirm or clean up")

    # ── Health score ──────────────────────────────────────────────────────
    critical = sum(1 for i in issues if i["severity"] == "CRITICAL")
    warnings  = sum(1 for i in issues if i["severity"] == "WARNING")
    infos     = sum(1 for i in issues if i["severity"] == "INFO")

    if critical > 0:
        health = "DEGRADED"
    elif warnings > 2:
        health = "NEEDS_ATTENTION"
    elif warnings > 0:
        health = "MOSTLY_HEALTHY"
    else:
        health = "HEALTHY"

    report = {
        "health": health,
        "summary": {
            "total_files":       len(files),
            "python_files":      sum(1 for f in files if f.endswith(".py")),
            "services_checked":  len(services),
            "services_active":   sum(1 for s in services.values() if s["status"] == "active"),
            "critical_issues":   critical,
            "warnings":          warnings,
            "info_items":        infos,
        },
        "issues":          issues,
        "recommendations": recommendations,
        "services":        services,
        "env_keys_found":  env.get("keys", []),
    }

    if verbose:
        report["all_files"]    = list(files.keys())
        report["import_graph"] = import_graph
        report["json_health"]  = json_health

    return report


# ── Pretty printer ─────────────────────────────────────────────────────────
def print_report(report: dict[str, Any]) -> None:
    health_emoji = {
        "HEALTHY":           "✅",
        "MOSTLY_HEALTHY":    "🟡",
        "NEEDS_ATTENTION":   "🟠",
        "DEGRADED":          "🔴",
    }
    h = report["health"]
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"  LUCI WORKSPACE AUDIT  {health_emoji.get(h, '❓')} {h}")
    print(f"{'='*60}")
    print(f"  Files scanned:   {s['total_files']}  ({s['python_files']} Python)")
    print(f"  Services:        {s['services_active']}/{s['services_checked']} active")
    print(f"  Critical issues: {s['critical_issues']}")
    print(f"  Warnings:        {s['warnings']}")
    print(f"  Info items:      {s['info_items']}")
    print()

    issues = report["issues"]
    if issues:
        print("── ISSUES ──────────────────────────────────────────────")
        for issue in issues:
            sev = issue["severity"]
            emoji = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "ℹ️"}.get(sev, "?")
            print(f"  {emoji} [{sev}] {issue['category']}")
            print(f"     File:   {issue['file']}")
            print(f"     Detail: {issue['detail']}")
            print()

    if report["recommendations"]:
        print("── RECOMMENDATIONS ─────────────────────────────────────")
        for r in report["recommendations"]:
            print(f"  → {r}")
        print()

    print("── SERVICES ────────────────────────────────────────────")
    for svc, info in report["services"].items():
        status = info["status"]
        emoji = "✅" if status == "active" else "🔴"
        print(f"  {emoji} {svc}: {status}")
    print()


# ── Auto-update feature registry ──────────────────────────────────────────
def auto_update_registry(files: dict) -> None:
    """
    Scan Python files for route decorators, CommandHandlers, and
    function signatures to auto-detect new features and add them
    to luci_features.json.
    """
    registry_path = WORKSPACE / "luci_features.json"
    if not registry_path.exists():
        return

    data = json.loads(registry_path.read_text())
    existing_names = {f["name"].lower() for f in data["features"]}

    new_features = []

    for rel, info in files.items():
        if info["ext"] != ".py":
            continue
        try:
            src = Path(info["path"]).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Detect new Telegram CommandHandlers not yet in registry
        for m in re.finditer(r'CommandHandler\(["\'](\w+)["\']', src):
            cmd = m.group(1)
            feature_name = f"/{cmd} command"
            if feature_name.lower() not in existing_names:
                # Try to find the handler function's docstring
                fn_match = re.search(rf'async def cmd_{cmd}\([^)]*\)[^:]*:[\s\n]+"""([^"]+)"""', src)
                desc = fn_match.group(1).strip() if fn_match else f"Telegram /{cmd} command (auto-detected)"
                new_features.append({
                    "name": feature_name,
                    "commands": [f"/{cmd}"],
                    "description": desc,
                    "_auto_detected": True,
                    "_source": rel,
                })
                existing_names.add(feature_name.lower())

        # Detect new FastAPI routes not yet in registry
        for m in re.finditer(r'@app\.(get|post|put|delete)\(["\']([^"\']+)["\']', src):
            route = m.group(2)
            feature_name = f"API: {route}"
            if feature_name.lower() not in existing_names:
                new_features.append({
                    "name": feature_name,
                    "commands": [route],
                    "description": f"HTTP {m.group(1).upper()} endpoint at {route} (auto-detected)",
                    "_auto_detected": True,
                    "_source": rel,
                })
                existing_names.add(feature_name.lower())

    if new_features:
        data["features"].extend(new_features)
        registry_path.write_text(json.dumps(data, indent=2))
        print(f"📝 Auto-registered {len(new_features)} new feature(s) in luci_features.json")
        for f in new_features:
            print(f"   + {f['name']}: {f['description'][:60]}")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    auto    = "--auto-register" in sys.argv or "-a" in sys.argv

    report = run_audit(verbose=verbose)
    print_report(report)

    if auto:
        print("── AUTO-REGISTERING NEW FEATURES ───────────────────────")
        files = scan_workspace()
        auto_update_registry(files)

    # Save report
    report_path = WORKSPACE / "runs" / "last_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n📄 Full report saved to: {report_path}")
