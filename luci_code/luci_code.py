#!/usr/bin/env python3
"""
LUCI Code — Agentic coding CLI (Claude Code feature parity)
Usage: luci-code [path] [--model MODEL] [--no-confirm] [--auto]
"""
from __future__ import annotations
import sys
import os
import time
import argparse
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.agent import Agent, DEFAULT_MODEL
from core.tools import ToolResult
from ui.renderer import (
    console, print_header, print_tool_call, print_tool_result,
    print_response, print_help, print_error, LUCI_THEME
)

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

# ── Spinner ────────────────────────────────────────────────────────────────
_spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_spinner_active = False
_spinner_msg    = ""
_spinner_thread = None

def _spin():
    i = 0
    while _spinner_active:
        ch = _spinner_chars[i % len(_spinner_chars)]
        sys.stdout.write(f"\r  {ch} {_spinner_msg}   ")
        sys.stdout.flush()
        time.sleep(0.08)
        i += 1

def start_spinner(msg: str):
    global _spinner_active, _spinner_msg, _spinner_thread
    _spinner_msg    = msg
    _spinner_active = True
    _spinner_thread = threading.Thread(target=_spin, daemon=True)
    _spinner_thread.start()

def stop_spinner():
    global _spinner_active
    _spinner_active = False
    if _spinner_thread:
        _spinner_thread.join(timeout=0.5)
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()


def parse_args():
    p = argparse.ArgumentParser(
        prog="luci-code",
        description="LUCI Code — Agentic coding CLI (Claude Code feature parity)"
    )
    p.add_argument("path",      nargs="?", default=".",
                   help="Project directory (default: current)")
    p.add_argument("--model",   "-m", default=DEFAULT_MODEL,
                   help=f"Ollama model (default: {DEFAULT_MODEL})")
    p.add_argument("--no-confirm", action="store_true",
                   help="Skip bash command confirmation")
    p.add_argument("--auto",    action="store_true",
                   help="Auto-approve all decisions (dangerous)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show full tool output")
    p.add_argument("-c", "--command", type=str, default="",
                   help="Non-interactive single command")
    return p.parse_args()


def ask_decision(decision_text: str, auto: bool = False) -> str:
    """Show a decision panel and get user input."""
    stop_spinner()
    console.print()
    console.print("[bold yellow]━━━ LUCI needs your input ━━━[/]")
    for line in decision_text.strip().splitlines():
        console.print(f"  {line}")
    console.print("[bold yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
    if auto:
        console.print("  [luci.dim](auto-mode: proceeding)[/]")
        return "proceed"
    try:
        answer = input("  Your choice (or press Enter to proceed): ").strip()
        return answer if answer else "proceed"
    except (EOFError, KeyboardInterrupt):
        return "cancel"


def ask_plan(plan_text: str, auto: bool = False) -> bool:
    """Show a plan and get approval."""
    stop_spinner()
    console.print()
    console.print("[bold #D4AF37]━━━ LUCI's Plan ━━━[/]")
    for line in plan_text.strip().splitlines():
        console.print(f"  {line}")
    console.print("[bold #D4AF37]━━━━━━━━━━━━━━━━━━━[/]")
    if auto:
        console.print("  [luci.dim](auto-mode: executing plan)[/]")
        return True
    try:
        answer = input("  Proceed? [Y/n]: ").strip().lower()
        return answer not in ("n", "no")
    except (EOFError, KeyboardInterrupt):
        return False


def get_input(session=None, prompt_str: str = "❯ ") -> str:
    if session and HAS_PROMPT_TOOLKIT:
        try:
            return session.prompt(prompt_str)
        except (EOFError, KeyboardInterrupt):
            return "/exit"
    else:
        try:
            return input(prompt_str)
        except (EOFError, KeyboardInterrupt):
            return "/exit"


def _run_single_benchmark(model: str) -> dict:
    import requests as _req, json as _json
    payload = {
        "model": model,
        "messages": [
            {"role": "system",  "content": "You are LUCI. Be concise."},
            {"role": "user",    "content": "Count from 1 to 5."}
        ],
        "stream": True,
        "options": {"temperature": 0.1, "num_ctx": 512},
    }
    t_start = time.perf_counter()
    t_first = None
    total_tokens = 0
    try:
        resp = _req.post("http://127.0.0.1:11434/api/chat",
                         json=payload, stream=True, timeout=60)
        for line in resp.iter_lines():
            if not line:
                continue
            data  = _json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                if t_first is None:
                    t_first = time.perf_counter()
                total_tokens += 1
            if data.get("done"):
                break
        t_end = time.perf_counter()
        if total_tokens == 0:
            return {"ok": False, "error": "No tokens received"}
        ttft  = (t_first - t_start) if t_first else 0
        total = t_end - t_start
        tps   = total_tokens / total if total > 0 else 0
        return {"ok": True, "ttft": ttft, "total": total,
                "tokens": total_tokens, "tps": tps}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


def run_benchmark(models: list[str]):
    console.print("[luci.dim]Benchmarking...[/]\n")
    results = {}
    for model in models:
        console.print(f"  Testing [luci.tool]{model}[/]...", end=" ")
        sys.stdout.flush()
        stats = _run_single_benchmark(model)
        results[model] = stats
        console.print("done" if stats["ok"] else "FAILED")
    console.print()
    console.print("[luci.gold]─── Benchmark Results ────────────[/]")
    for model, s in results.items():
        console.print(f"  [bold]{model}[/]")
        if s["ok"]:
            rating = (
                "[luci.success]FAST[/]"      if s["ttft"] < 1.5 else
                "[luci.dim]ACCEPTABLE[/]"    if s["ttft"] < 4.0 else
                "[luci.error]SLOW[/]"
            )
            console.print(
                f"    ⚡ First token: [bold]{s['ttft']:.2f}s[/]  "
                f"⏱ Total: [bold]{s['total']:.2f}s[/]  "
                f"🚀 [bold]{s['tps']:.1f}[/] tok/s  {rating}"
            )
        else:
            console.print(f"    [luci.error]✗ {s['error']}[/]")
    console.print()


def main():
    args = parse_args()
    workspace = Path(args.path).resolve()
    if not workspace.exists():
        print(f"Error: {workspace} does not exist")
        sys.exit(1)

    os.chdir(workspace)
    verbose       = args.verbose
    auto          = args.auto
    current_model = args.model
    currently_streaming = False
    task_start_time     = None

    _token_buffer = []
    _in_xml = False

    def on_token(token: str):
        nonlocal currently_streaming, _token_buffer, _in_xml
        global _spinner_active
        if _spinner_active:
            stop_spinner()

        # Buffer tokens to suppress XML tags from display
        _token_buffer.append(token)
        buffered = "".join(_token_buffer)

        # If we are inside an XML block, keep buffering silently
        XML_OPENS = ("<tool>", "<args>", "<plan>", "<decision>", "<tool_result")
        XML_CLOSES = ("</args>", "</plan>", "</decision>", "</tool_result>")

        if any(tag in buffered for tag in XML_OPENS):
            _in_xml = True

        if _in_xml:
            # Check if the XML block closed
            if any(tag in buffered for tag in XML_CLOSES):
                # Consume the buffer silently, reset
                _token_buffer.clear()
                _in_xml = False
            # Don\'t print anything while in XML
            return

        # Safe to print — flush buffer
        to_print = "".join(_token_buffer)
        _token_buffer.clear()

        # Skip lines that are purely XML artifacts
        if to_print.strip() in ("<plan>", "</plan>", "<tool>", "</tool>",
                                 "<args>", "</args>", "<decision>", "</decision>"):
            return

        for ch in to_print:
            sys.stdout.write(ch)
            sys.stdout.flush()
        currently_streaming = True

    def on_tool(name: str, tool_args: dict, result: ToolResult, elapsed: float = 0.0):
        nonlocal currently_streaming
        if currently_streaming:
            print()
            currently_streaming = False
        stop_spinner()
        print_tool_call(name, tool_args, elapsed)
        print_tool_result(name, result, verbose=verbose)

    def on_status(msg: str):
        nonlocal currently_streaming
        if currently_streaming:
            print()
            currently_streaming = False
        start_spinner(msg)

    def on_decision(decision_text: str) -> str:
        return ask_decision(decision_text, auto=auto)

    def on_plan(plan_text: str) -> bool:
        return ask_plan(plan_text, auto=auto)

    def make_agent():
        return Agent(
            workspace=workspace,
            model=current_model,
            on_token=on_token,
            on_tool=on_tool,
            on_status=on_status,
            on_decision=on_decision,
            on_plan=on_plan,
            confirm_bash=not args.no_confirm and not auto,
            auto_commit=False,
        )

    agent = make_agent()
    print_header(str(workspace), current_model)

    # VRAM check
    try:
        import subprocess as _sp
        vram = _sp.run(
            "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
            shell=True, capture_output=True, text=True
        ).stdout.strip()
        free_mb = int(vram) if vram.isdigit() else 0
        if free_mb and free_mb < 6000:
            console.print(f"[luci.error]⚠ Low VRAM: {free_mb}MB free[/]")
        elif free_mb:
            console.print(f"[luci.dim]GPU: {free_mb}MB VRAM free[/]")
    except Exception:
        pass

    # Non-interactive mode
    if args.command:
        cmd_str = args.command.strip()
        if cmd_str == "/benchmark":
            run_benchmark([current_model])
            return
        on_status("Working...")
        response = agent.chat(cmd_str)
        stop_spinner()
        print_response(response)
        return

    # Interactive session
    history_file = Path.home() / ".luci_code_history"
    session = None
    if HAS_PROMPT_TOOLKIT:
        session = PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            style=Style.from_dict({"prompt": "#D4AF37 bold"}),
        )

    console.print("[luci.dim]Ready. Describe what you want to build.[/]\n")
    console.print("[luci.dim]Commands: /help /clear /context /model /undo /git /diff /benchmark /exit[/]\n")

    while True:
        try:
            cwd_display = str(agent.tools.workspace).replace(str(Path.home()), "~")
            user_input  = get_input(session, f"{cwd_display} ❯ ").strip()
        except KeyboardInterrupt:
            continue

        if not user_input:
            continue

        # cd built-in
        if user_input.startswith("cd ") or user_input == "cd":
            new_path = user_input[3:].strip() if user_input.startswith("cd ") \
                       else str(Path.home())
            new_path = str(Path(new_path).expanduser().resolve())
            result   = agent.tools.change_dir(new_path)
            if result.success:
                workspace = Path(new_path)
                agent.workspace = workspace
                from core.context import CodebaseContext
                agent.context = CodebaseContext(workspace)
                agent._setup_system()
                os.chdir(workspace)
                console.print(f"[luci.dim]→ {new_path}[/]")
            else:
                print_error(result.error)
            continue

        # Slash commands
        if user_input.startswith("/"):
            cmd   = user_input.lower().split()[0]
            parts = user_input.split(maxsplit=1)

            if cmd in ("/exit", "/quit", "/q"):
                console.print("[luci.gold]Goodbye.[/]")
                break
            elif cmd == "/help":
                print_help()
            elif cmd == "/clear":
                agent.reset()
                console.print("[luci.dim]Conversation cleared.[/]")
            elif cmd == "/context":
                console.print(agent.context.get_summary())
            elif cmd == "/verbose":
                verbose = not verbose
                console.print(f"[luci.dim]Verbose: {'on' if verbose else 'off'}[/]")
            elif cmd == "/auto":
                auto = not auto
                console.print(f"[luci.dim]Auto-approve: {'on' if auto else 'off'}[/]")
            elif cmd == "/model":
                if len(parts) > 1:
                    current_model = parts[1].strip()
                    agent = make_agent()
                    console.print(f"[luci.dim]Model: {current_model}[/]")
                else:
                    console.print(f"[luci.dim]Current: {current_model}[/]")
            elif cmd == "/undo":
                console.print("[luci.dim]Rolling back last changes...[/]")
                result = agent.rollback()
                console.print(result)
            elif cmd == "/git":
                r = agent.tools.git_status()
                console.print(r.output or "(clean)")
            elif cmd == "/diff":
                r = agent.tools.git_diff()
                console.print(r.output or "(no diff)")
            elif cmd == "/log":
                r = agent.tools.git_log()
                console.print(r.output)
            elif cmd == "/run":
                if len(parts) > 1:
                    r = agent.tools.bash(parts[1])
                    console.print(r.output)
                else:
                    console.print("[luci.error]Usage: /run <command>[/]")
            elif cmd == "/benchmark":
                run_benchmark([current_model])
            elif cmd == "/changed":
                if agent._changed_files:
                    console.print(
                        "[luci.dim]Changed this session:[/]\n" +
                        "\n".join(f"  {f}" for f in sorted(agent._changed_files))
                    )
                else:
                    console.print("[luci.dim]No changes this session.[/]")
            else:
                console.print(f"[luci.error]Unknown command: {cmd}[/]")
            continue

        # Normal message
        console.print()
        task_start_time = time.perf_counter()
        on_status("LUCI is thinking...")
        try:
            response = agent.chat(user_input)
            stop_spinner()
            if currently_streaming:
                print()
                currently_streaming = False
        except KeyboardInterrupt:
            stop_spinner()
            console.print("\n[luci.dim](interrupted)[/]")
            currently_streaming = False
            continue
        except Exception as e:
            stop_spinner()
            print_error(str(e))
            continue

        total_elapsed = time.perf_counter() - task_start_time
        console.print()
        console.print(f"[luci.dim]Task completed in {total_elapsed:.1f}s[/]")
        console.print()


if __name__ == "__main__":
    main()
