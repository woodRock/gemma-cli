import os
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from rich.console import Console

import gemma.config as cfg
from gemma.splash import show_splash
from gemma.chat import ChatSession
from gemma.skills import handle as handle_skill

console = Console()

PROMPT_STYLE = Style.from_dict({
    "prompt": "ansicyan bold",
})


def _bottom_toolbar(session: ChatSession):
    from gemma.models import get_model
    model = get_model(session.config.get("model", "e4b"))
    return HTML(
        f'<b><style fg="ansicyan"> ◈ {model.name}</style></b>'
        f'<style fg="ansibrightblack">  mlx  ·  /help · /model · /pull · /reset · /exit</style>'
    )


def main():
    os.system("clear")
    config = cfg.load()
    show_splash(config)

    session = ChatSession(config)

    cfg.ensure_dirs()
    prompt_session: PromptSession = PromptSession(
        history=FileHistory(str(cfg.HISTORY_FILE)),
        style=PROMPT_STYLE,
        bottom_toolbar=lambda: _bottom_toolbar(session),
        mouse_support=False,
        multiline=False,
    )

    while True:
        try:
            raw = prompt_session.prompt(HTML('<prompt>◈ </prompt>')).strip()
        except KeyboardInterrupt:
            console.print("\n[dim](Use /exit to quit)[/]")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye.[/]")
            break

        if not raw:
            continue

        result = handle_skill(session, raw)
        if result is False:
            console.print("[dim]Goodbye.[/]")
            break
        if result is True:
            continue

        try:
            session.send(raw)
        except KeyboardInterrupt:
            console.print("\n[dim](Interrupted)[/]")
        except Exception as e:
            import traceback
            console.print(f"[bold red]Error:[/] {e}")
            console.print("[dim]" + traceback.format_exc() + "[/]")


if __name__ == "__main__":
    main()
