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
    return HTML(f'<b><style fg="ansicyan"> ◈ {model.name}</style></b>'
                f'<style fg="ansibrightblack">  /help · /model · /reset · /exit</style>')


def main():
    os.system("clear")

    config = cfg.load()
    show_splash(config)

    api_key = cfg.get_api_key(config)
    if not api_key:
        console.print("[bold red]No API key found.[/]")
        console.print("[dim]Set [bold]GEMINI_API_KEY[/] in your environment or run:[/]")
        console.print("[dim]  export GEMINI_API_KEY=your-key[/]")
        console.print()

    session = ChatSession(config)

    history_path = cfg.HISTORY_FILE
    cfg.ensure_dirs()
    prompt_session: PromptSession = PromptSession(
        history=FileHistory(str(history_path)),
        style=PROMPT_STYLE,
        bottom_toolbar=lambda: _bottom_toolbar(session),
        mouse_support=False,
        multiline=False,
    )

    while True:
        try:
            raw = prompt_session.prompt(
                HTML('<prompt>◈ </prompt>'),
            ).strip()
        except KeyboardInterrupt:
            console.print("\n[dim](Use /exit to quit)[/]")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye.[/]")
            break

        if not raw:
            continue

        # Skill / slash-command handling
        result = handle_skill(session, raw)
        if result is False:
            console.print("[dim]Goodbye.[/]")
            break
        if result is True:
            continue

        # Regular message → send to model
        try:
            session.send(raw)
        except KeyboardInterrupt:
            console.print("\n[dim](Interrupted)[/]")
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")


if __name__ == "__main__":
    main()
