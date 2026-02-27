"""
LUCI Code UI — Rich terminal rendering.
"""
from __future__ import annotations
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
from rich.theme import Theme
from rich import box

LUCI_THEME = Theme({
    "luci.gold":     "bold #D4AF37",
    "luci.dim":      "#7a6a3a",
    "luci.tool":     "bold cyan",
    "luci.success":  "bold green",
    "luci.error":    "bold red",
    "luci.user":     "bold white",
    "luci.file":     "bold blue",
    "luci.bash":     "bold yellow",
})

console = Console(theme=LUCI_THEME, highlight=True)


def print_header(workspace: str, model: str):
    console.print(Panel(
        f"[luci.gold]LUCI Code[/] — [luci.dim]{workspace}[/]\n"
        f"[luci.dim]Model: {model} | Type [bold]/help[/] for commands[/]",
        border_style="#D4AF37",
        box=box.ROUNDED,
    ))


def print_tool_call(tool_name: str, args: dict):
    icons = {
        "read_file":   "📖",
        "create_file": "✨",
        "str_replace": "✏️ ",
        "bash":        "⚡",
        "glob":        "🔍",
        "grep":        "🔎",
        "git_diff":    "📊",
        "git_status":  "📋",
        "git_commit":  "💾",
        "git_push":    "🚀",
        "tree":        "🌲",
        "ls":          "📁",
    }
    icon = icons.get(tool_name, "🔧")
    label = {
        "read_file":   f"Reading {args.get('path', '')}",
        "create_file": f"Creating {args.get('path', '')}",
        "str_replace": f"Editing {args.get('path', '')}",
        "bash":        f"$ {args.get('command', '')[:60]}",
        "glob":        f"Finding {args.get('pattern', '')}",
        "grep":        f"Searching for {args.get('pattern', '')}",
        "git_diff":    "Git diff",
        "git_status":  "Git status",
        "git_commit":  f"Committing: {args.get('message', '')[:40]}",
        "git_push":    "Git push",
        "tree":        f"Tree {args.get('path', '.')}",
        "ls":          f"ls {args.get('path', '.')}",
    }.get(tool_name, f"{tool_name} {args}")
    console.print(f"  {icon} [luci.tool]{label}[/]")


def print_tool_result(tool_name: str, result, verbose: bool = False):
    if not result.success:
        console.print(f"  [luci.error]✗ {result.error}[/]")
        return
    if tool_name in ("str_replace", "create_file", "git_commit", "git_push"):
        console.print(f"  [luci.success]✓ {result.output.splitlines()[0][:80]}[/]")
    elif verbose and result.output:
        lines = result.output.splitlines()
        preview = "\n".join(lines[:20])
        if len(lines) > 20:
            preview += f"\n  ... ({len(lines)-20} more lines)"
        console.print(f"  [luci.dim]{preview}[/]")


def print_response(text: str):
    """Print LUCI's final response with markdown rendering."""
    import re
    clean = re.sub(r"<tool>.*?</tool>\s*<args>.*?</args>", "", text, flags=re.DOTALL)
    clean = re.sub(r"<tool_result[^>]*>.*?</tool_result>", "", clean, flags=re.DOTALL)
    clean = clean.strip()
    if clean:
        console.print()
        console.print(Markdown(clean))
        console.print()


def print_user_prompt() -> str:
    """Print the user input prompt."""
    console.print("[luci.gold]❯[/] ", end="")


def streaming_display():
    """Return a Live context for streaming output."""
    return Live(console=console, refresh_per_second=15)


def print_error(msg: str):
    console.print(f"[luci.error]Error: {msg}[/]")


def print_help():
    console.print(Panel(
        "[luci.gold]LUCI Code Commands[/]\n\n"
        "  [bold]/help[/]          Show this help\n"
        "  [bold]/clear[/]         Clear conversation history\n"
        "  [bold]/context[/]       Show project context\n"
        "  [bold]/model <name>[/]  Switch model\n"
        "  [bold]/git[/]           Show git status\n"
        "  [bold]/diff[/]          Show git diff\n"
        "  [bold]/log[/]           Show git log\n"
        "  [bold]/verbose[/]       Toggle verbose tool output\n"
        "  [bold]/exit[/]          Exit\n\n"
        "Just type your request naturally:\n"
        "  [luci.dim]add error handling to auth.py[/]\n"
        "  [luci.dim]write tests for the user module[/]\n"
        "  [luci.dim]why is this failing: python main.py[/]\n"
        "  [luci.dim]refactor the database layer to use async[/]",
        border_style="#D4AF37",
        box=box.ROUNDED,
        title="Help",
    ))
