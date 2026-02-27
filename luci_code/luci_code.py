#!/usr/bin/env python3
"""
LUCI Code — Agentic coding CLI
Usage: luci-code [path] [--model MODEL] [--no-confirm]
"""
from __future__ import annotations
import sys
import os
import argparse
from pathlib import Path

# Add luci_code dir to path
sys.path.insert(0, str(Path(__file__).parent))

from core.agent import Agent, DEFAULT_MODEL
from core.tools import ToolResult
from ui.renderer import (
    console, print_header, print_tool_call, print_tool_result,
    print_response, print_help, print_error,
    LUCI_THEME
)

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


def parse_args():
    p = argparse.ArgumentParser(
        prog="luci-code",
        description="LUCI Code — Agentic coding CLI powered by local AI"
    )
    p.add_argument("path", nargs="?", default=".",
                   help="Project directory (default: current)")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL,
                   help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    p.add_argument("--no-confirm", action="store_true",
                   help="Skip bash command confirmation")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show full tool output")
    p.add_argument("-c", "--command", type=str, default="",
                   help="Run a single command non-interactively")
    return p.parse_args()


def _run_single_benchmark(model: str) -> dict:
    """Run one benchmark pass against Ollama. Returns stats dict."""
    import time, requests as _req, json as _json
    test_prompt = "Count from 1 to 10 and explain why numbers matter."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are LUCI. Be concise."},
            {"role": "user", "content": test_prompt}
        ],
        "stream": True,
        "options": {"temperature": 0.1, "num_ctx": 2048},
    }
    t_start = time.perf_counter()
    t_first = None
    total_tokens = 0
    try:
        resp = _req.post("http://127.0.0.1:11434/api/chat",
                         json=payload, stream=True, timeout=120)
        for line in resp.iter_lines():
            if not line:
                continue
            data = _json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                if t_first is None:
                    t_first = time.perf_counter()
                total_tokens += 1
            if data.get("done"):
                break
        t_end = time.perf_counter()
        if total_tokens == 0:
            return {"ok": False, "error": "No tokens received — model may not be pulled"}
        ttft = (t_first - t_start) if t_first else 0
        total = t_end - t_start
        tps = total_tokens / total if total > 0 else 0
        return {"ok": True, "ttft": ttft, "total": total, "tokens": total_tokens, "tps": tps}
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


def _print_benchmark(console, model: str, stats: dict):
    """Print benchmark results for one model."""
    if not stats["ok"]:
        console.print(f"  [luci.error]✗ {stats['error']}[/]")
        return
    ttft, total, tokens, tps = stats["ttft"], stats["total"], stats["tokens"], stats["tps"]
    if ttft < 1.5:
        rating = "[luci.success]FAST[/]"
    elif ttft < 4.0:
        rating = "[luci.dim]ACCEPTABLE[/]"
    else:
        rating = "[luci.error]SLOW[/]"
    console.print(f"  ⚡ First token:  [bold]{ttft:.2f}s[/]  "
                  f"⏱ Total: [bold]{total:.2f}s[/]  "
                  f"🚀 [bold]{tps:.1f}[/] tok/s  📊 {rating}")


def run_benchmark(console, models: list[str]):
    """Run benchmark against one or more models and display comparison."""
    from rich.panel import Panel
    from rich import box
    console.print("[luci.dim]Benchmarking — measuring first token latency + throughput...[/]\n")
    results = {}
    for model in models:
        console.print(f"  Testing [luci.tool]{model}[/]...", end=" ")
        import sys; sys.stdout.flush()
        stats = _run_single_benchmark(model)
        results[model] = stats
        console.print("done" if stats["ok"] else "FAILED")

    console.print()
    console.print("[luci.gold]─── Benchmark Results ───────────────────────[/]")
    for model, stats in results.items():
        console.print(f"  [bold]{model}[/]")
        _print_benchmark(console, model, stats)
    if len(results) == 2:
        vals = list(results.values())
        if vals[0]["ok"] and vals[1]["ok"]:
            ratio = vals[1]["ttft"] / vals[0]["ttft"] if vals[0]["ttft"] > 0 else 0
            console.print(f"\n  [luci.dim]Speed ratio: {ratio:.1f}x difference in first-token latency[/]")
    console.print()


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


def main():
    args = parse_args()
    workspace = Path(args.path).resolve()
    if not workspace.exists():
        print(f"Error: {workspace} does not exist")
        sys.exit(1)

    # Change to workspace directory
    os.chdir(workspace)

    verbose = args.verbose
    current_model = args.model

    # Streaming token buffer
    currently_streaming = False

    def on_token(token: str):
        nonlocal currently_streaming
        for ch in token:
            sys.stdout.write(ch)
            sys.stdout.flush()
        currently_streaming = True

    def on_tool(name: str, tool_args: dict, result: ToolResult):
        nonlocal currently_streaming
        if currently_streaming:
            print()  # newline after streaming
            currently_streaming = False
        print_tool_call(name, tool_args)
        print_tool_result(name, result, verbose=verbose)

    def make_agent():
        return Agent(
            workspace=workspace,
            model=current_model,
            on_token=on_token,
            on_tool=on_tool,
            confirm_bash=not args.no_confirm,
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
            console.print(
                f"[luci.error]⚠ Low VRAM: {free_mb}MB free. "
                f"Large models will use CPU. Using qwen2.5:14b.[/]"
            )
        elif free_mb:
            console.print(f"[luci.dim]GPU: {free_mb}MB VRAM free[/]")
    except Exception:
        pass

    # Non-interactive mode
    if args.command:
        cmd_str = args.command.strip()
        if cmd_str == "/benchmark":
            bench_models = ["chip-premium-dolphin:latest", current_model] \
                if current_model != "chip-premium-dolphin:latest" \
                else ["chip-premium-dolphin:latest"]
            run_benchmark(console, bench_models)
            return
        response = agent.chat(cmd_str)
        print_response(response)
        return

    # Interactive session
    history_file = Path.home() / ".luci_code_history"
    session = None
    if HAS_PROMPT_TOOLKIT:
        pt_style = Style.from_dict({"prompt": "#D4AF37 bold"})
        session = PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            style=pt_style,
        )

    console.print("[luci.dim]Ready. Describe what you want to do.[/]\n")

    while True:
        try:
            cwd_display = str(agent.tools.workspace).replace(str(Path.home()), "~")
            user_input = get_input(session, f"{cwd_display} ❯ ").strip()
        except KeyboardInterrupt:
            continue

        if not user_input:
            continue

        # Handle cd as a special built-in
        if user_input.startswith("cd ") or user_input == "cd":
            new_path = user_input[3:].strip() if user_input.startswith("cd ") else str(Path.home())
            new_path = str(Path(new_path).expanduser().resolve())
            result = agent.tools.change_dir(new_path)
            if result.success:
                workspace = Path(new_path)
                agent.workspace = workspace
                agent.tools.workspace = workspace
                from core.context import CodebaseContext
                agent.context = CodebaseContext(workspace)
                agent._setup_system()
                os.chdir(workspace)
                console.print(f"[luci.dim]→ {new_path}[/]")
            else:
                print_error(result.error)
            continue

        # Built-in commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
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

            elif cmd == "/model":
                if len(parts) > 1:
                    current_model = parts[1].strip()
                    agent = make_agent()
                    console.print(f"[luci.dim]Model: {current_model}[/]")
                else:
                    console.print(f"[luci.dim]Current model: {current_model}[/]")

            elif cmd == "/git":
                r = agent.tools.git_status()
                console.print(r.output or "(nothing changed)")

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
                bench_models = ["chip-premium-dolphin:latest", current_model] \
                    if current_model != "chip-premium-dolphin:latest" \
                    else ["chip-premium-dolphin:latest"]
                run_benchmark(console, bench_models)

            else:
                console.print(f"[luci.error]Unknown command: {cmd}[/]")
            continue

        # Normal message to agent
        console.print()
        try:
            response = agent.chat(user_input)
            if currently_streaming:
                print()
                currently_streaming = False
        except KeyboardInterrupt:
            console.print("\n[luci.dim](interrupted)[/]")
            currently_streaming = False
            continue
        except Exception as e:
            print_error(str(e))
            continue

        console.print()


if __name__ == "__main__":
    main()
