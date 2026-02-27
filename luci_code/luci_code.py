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

    # Non-interactive mode
    if args.command:
        response = agent.chat(args.command)
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
            user_input = get_input(session, "❯ ").strip()
        except KeyboardInterrupt:
            continue

        if not user_input:
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
