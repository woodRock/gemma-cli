from __future__ import annotations
import os
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from gemma.chat import ChatSession

Skill = Callable[["ChatSession", list[str]], bool]
SKILLS: dict[str, tuple[str, Skill]] = {}


def skill(name: str, description: str):
    def decorator(fn: Skill) -> Skill:
        SKILLS[name] = (description, fn)
        return fn
    return decorator


@skill("help", "Show available commands")
def _help(session, args):
    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(title="Skills", border_style="cyan", title_style="bold cyan")
    table.add_column("Command", style="bold green", no_wrap=True)
    table.add_column("Description", style="dim")
    for name, (desc, _) in sorted(SKILLS.items()):
        table.add_row(f"/{name}", desc)
    console.print(table)
    return True


@skill("model", "Show or switch model  (/model [key])")
def _model(session, args):
    from rich.console import Console
    from rich.table import Table
    import gemma.models as m
    import gemma.config as cfg
    from gemma.inference import is_downloaded
    from gemma.config import MODEL_CACHE_DIR
    console = Console()

    if args:
        key = args[0]
        if key not in m.MODELS:
            console.print(f"[red]Unknown model:[/] {key}. Available: {', '.join(m.MODELS)}")
            return True
        session.config["model"] = key
        cfg.save(session.config)
        console.print(f"[bold cyan]◈[/] Switched to [bold]{m.get_model(key).name}[/]")
        console.print("[dim]  Model will load on next message.[/]")
        return True

    table = Table(title="Gemma 4 Models", border_style="cyan", title_style="bold cyan")
    table.add_column("", width=2)
    table.add_column("Key", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Description")
    table.add_column("Size", justify="right", style="dim")
    table.add_column("Local", justify="center")

    current = session.config.get("model", m.DEFAULT_MODEL)
    for key, model in m.MODELS.items():
        marker = "[bold green]●[/]" if key == current else " "
        cached = "[green]✓[/]" if is_downloaded(model.hf_id, MODEL_CACHE_DIR) else "[dim]—[/]"
        table.add_row(marker, key, model.name, model.description, f"{model.size_gb:.1f} GB", cached)

    console.print(table)
    console.print("[dim]Usage: /model <key>  ·  Use /pull to download a model first[/]")
    return True


@skill("pull", "Download a Gemma 4 model  (/pull [key])")
def _pull(session, args):
    from rich.console import Console
    import gemma.models as m
    from gemma.inference import pull, is_downloaded
    from gemma.config import MODEL_CACHE_DIR
    console = Console()

    key = args[0] if args else session.config.get("model", m.DEFAULT_MODEL)
    if key not in m.MODELS:
        console.print(f"[red]Unknown model:[/] {key}. Available: {', '.join(m.MODELS)}")
        return True

    model = m.get_model(key)
    if is_downloaded(model.hf_id, MODEL_CACHE_DIR):
        console.print(f"[bold green]✓[/] [bold]{model.name}[/] is already downloaded.")
        return True

    console.print(f"[dim]HuggingFace repo:[/] {model.hf_id}")
    console.print(f"[dim]Estimated size:  {model.size_gb:.1f} GB[/]")
    console.print()
    pull(model.hf_id, MODEL_CACHE_DIR)
    return True


@skill("device", "Show the active compute device")
def _device(session, args):
    from rich.console import Console
    from gemma.inference import detect_device, get_dtype
    import torch
    console = Console()
    device = detect_device()
    dtype = get_dtype(device)
    console.print(f"[bold cyan]◈[/] Device: [bold]{device}[/]   dtype: [dim]{dtype}[/]")
    if device == "cuda":
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            console.print(f"  GPU {i}: {name}  ({mem:.1f} GB)")
    return True


@skill("clear", "Clear screen and redraw splash")
def _clear(session, args):
    os.system("clear")
    from gemma.splash import show_splash
    show_splash(session.config)
    return True


@skill("reset", "Reset conversation history")
def _reset(session, args):
    from rich.console import Console
    session.reset()
    Console().print("[bold cyan]◈[/] [dim]Conversation reset.[/]")
    return True


@skill("tools", "List available file tools")
def _tools(session, args):
    from rich.console import Console
    from rich.table import Table
    from gemma.tools import TOOL_DECLARATIONS
    console = Console()
    table = Table(title="File Tools", border_style="cyan", title_style="bold cyan")
    table.add_column("Tool", style="bold green", no_wrap=True)
    table.add_column("Description", style="dim")
    for t in TOOL_DECLARATIONS:
        table.add_row(t["name"], t["description"])
    console.print(table)
    return True


@skill("config", "Show current configuration")
def _config(session, args):
    from rich.console import Console
    from rich.pretty import Pretty
    Console().print(Pretty(session.config))
    return True


@skill("exit", "Exit gemma")
def _exit(session, args):
    return False


@skill("quit", "Exit gemma")
def _quit(session, args):
    return False


def handle(session: "ChatSession", line: str):
    """Returns True to continue, False to quit, None if not a skill."""
    if not line.startswith("/"):
        return None
    parts = line[1:].split()
    if not parts:
        return None
    name = parts[0].lower()
    args = parts[1:]
    if name in SKILLS:
        return SKILLS[name][1](session, args)
    from rich.console import Console
    Console().print(f"[red]Unknown skill:[/] /{name}  — type [bold]/help[/] for commands")
    return True
