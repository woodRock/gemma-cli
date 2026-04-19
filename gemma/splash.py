from rich.console import Console
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.padding import Padding

# Block-letter GEMMA — hand-tuned for terminal width ~70
GEMMA_ASCII = """\
  ██████╗ ███████╗███╗   ███╗███╗   ███╗  █████╗
 ██╔════╝ ██╔════╝████╗ ████║████╗ ████║ ██╔══██╗
 ██║  ███╗█████╗  ██╔████╔██║██╔████╔██║ ███████║
 ██║   ██║██╔══╝  ██║╚██╔╝██║██║╚██╔╝██║ ██╔══██║
 ╚██████╔╝███████╗██║ ╚═╝ ██║██║ ╚═╝ ██║ ██║  ██║
  ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝ ╚═╝  ╚═╝"""

# Decorative gem crown that sits above the wordmark
GEM_CROWN = """\
          ◇       ◆       ◇
       ◆     ◇ ◈ ◇ ◈ ◇     ◆
     ◇   ◆  ◇           ◇  ◆   ◇
      ◆     ◇  ◈  ◆  ◈  ◇     ◆
       ◇       ◆  ◈  ◆       ◇"""


def _gradient_text(text: str, start_color: str, end_color: str) -> Text:
    """Apply a left→right colour gradient to a plain string using rich markup."""
    rich_text = Text()
    lines = text.splitlines()
    for line in lines:
        n = max(len(line), 1)
        for i, ch in enumerate(line):
            t = i / n
            # Simple linear interpolation on the two named colors; rich handles rendering
            # We alternate between start and end color bands across each line
            color = start_color if t < 0.5 else end_color
            rich_text.append(ch, style=f"bold {color}")
        rich_text.append("\n")
    return rich_text


def show_splash(config: dict):
    from gemma import __version__
    from gemma.models import get_model

    console = Console()
    model = get_model(config.get("model", "e4b"))

    console.print()

    # Gem crown
    crown = Text(GEM_CROWN, style="bold blue")
    console.print(Align.center(crown))

    console.print()

    # Main wordmark — cyan left half, blue right half for a gem-facet effect
    logo = _gradient_text(GEMMA_ASCII, "cyan", "blue")
    console.print(Align.center(logo))

    console.print()

    # Sub-line decoration
    bar = Text("◆ ◇ ◈ ◇ ◆ ◇ ◈ ◇ ◆ ◇ ◈ ◇ ◆ ◇ ◈ ◇ ◆", style="dim cyan")
    console.print(Align.center(bar))

    console.print()

    # Info row
    version_text = Text(f" v{__version__} ", style="bold cyan on grey15")
    model_text   = Text(f" {model.name}  ·  {model.description} ", style="bold white on grey15")
    google_text  = Text(" Google DeepMind ", style="dim white on grey15")
    console.print(Align.center(Columns([version_text, model_text, google_text], padding=(0, 1))))

    console.print()

    hint = Text(
        "  type a message to start  ·  /help for skills  ·  Ctrl+C or /exit to quit  ",
        style="dim",
    )
    console.print(Align.center(hint))
    console.print()
