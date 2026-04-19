from __future__ import annotations
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

console = Console()


class ChatSession:
    def __init__(self, config: dict):
        self.config = config
        self.history: list[dict] = []
        self._client = None
        self._setup_client()

    def _setup_client(self):
        from gemma.config import get_api_key
        api_key = get_api_key(self.config)
        if not api_key:
            console.print("[bold red]Error:[/] No API key found.")
            console.print("[dim]Set the [bold]GEMINI_API_KEY[/] environment variable, or add it to [bold]~/.gemma/config.json[/].[/]")
            sys.exit(1)
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
        except ImportError:
            console.print("[bold red]Error:[/] google-genai not installed.")
            console.print("[dim]Run: pip install google-genai[/]")
            sys.exit(1)

    def reset(self):
        self.history = []

    @property
    def _model_id(self) -> str:
        from gemma.models import get_model
        return get_model(self.config.get("model", "e4b")).id

    def send(self, message: str):
        from google import genai  # noqa: F401
        from google.genai import types
        from gemma.tools import TOOL_DECLARATIONS, execute

        self.history.append({"role": "user", "content": message})

        tool_declarations = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=t["parameters"],
            )
            for t in TOOL_DECLARATIONS
        ]
        gen_tools = [types.Tool(function_declarations=tool_declarations)]
        gen_config = types.GenerateContentConfig(tools=gen_tools, temperature=0.7)

        # Build content list from history
        def build_contents():
            contents = []
            for msg in self.history:
                role = msg["role"]
                if role == "user":
                    contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
                elif role == "model":
                    contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))
                elif role == "tool_call":
                    contents.append(types.Content(
                        role="model",
                        parts=[types.Part(function_call=types.FunctionCall(
                            name=msg["name"], args=msg["args"]
                        ))]
                    ))
                elif role == "tool_result":
                    contents.append(types.Content(
                        role="tool",
                        parts=[types.Part(function_response=types.FunctionResponse(
                            name=msg["name"],
                            response={"result": msg["result"]},
                        ))]
                    ))
            return contents

        # Agentic loop: run until no more tool calls
        while True:
            contents = build_contents()

            # Attempt streaming; fall back to non-streaming on error
            response_text = ""
            function_calls = []

            try:
                response_text, function_calls = self._stream(contents, gen_config)
            except Exception as e:
                console.print(f"[red]Stream error:[/] {e}")
                break

            if function_calls:
                for fc in function_calls:
                    name = fc.name
                    args = dict(fc.args) if fc.args else {}

                    if self.config.get("show_tool_calls", True):
                        console.print(f"\n[bold cyan]◈[/] [dim]tool:[/] [bold green]{name}[/]")
                        for k, v in args.items():
                            v_str = str(v)
                            if len(v_str) > 120:
                                v_str = v_str[:120] + "…"
                            console.print(f"  [dim]{k}:[/] {v_str}")

                    result = execute(name, args)

                    if self.config.get("show_tool_calls", True):
                        preview = result[:300] + "…" if len(result) > 300 else result
                        console.print(f"  [dim]→[/] {preview}\n")

                    self.history.append({"role": "tool_call", "name": name, "args": args})
                    self.history.append({"role": "tool_result", "name": name, "result": result})

            if response_text:
                self.history.append({"role": "model", "content": response_text})

            if not function_calls:
                break

    def _stream(self, contents, gen_config) -> tuple[str, list]:
        """Stream a response; return (accumulated_text, function_calls)."""
        accumulated = ""
        function_calls = []
        show_markdown = True

        printed_chars = 0
        print_buffer = ""

        import sys

        for chunk in self._client.models.generate_content_stream(
            model=self._model_id,
            contents=contents,
            config=gen_config,
        ):
            if chunk.text:
                accumulated += chunk.text
                print_buffer += chunk.text
                # Flush in small bursts so the terminal feels live
                sys.stdout.write(chunk.text)
                sys.stdout.flush()

            if chunk.candidates:
                for candidate in chunk.candidates:
                    if not candidate.content:
                        continue
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            function_calls.append(part.function_call)

        if accumulated:
            # Move to new line after raw stream, then re-render as markdown
            sys.stdout.write("\n")
            sys.stdout.flush()
            if show_markdown and not function_calls:
                # Clear the raw stream output and re-render as rich markdown
                # (only feasible in most terminals — use ANSI escape to move up)
                lines_printed = accumulated.count("\n") + 1
                # Instead of clearing (fragile), just print a separator and the markdown
                console.print()
                console.rule(style="dim cyan")
                console.print(Markdown(accumulated))

        return accumulated, function_calls
