from __future__ import annotations

import sys
from rich.console import Console
from rich.markdown import Markdown

from gemma.inference import LocalEngine, parse_tool_calls

console = Console()


class ChatSession:
    def __init__(self, config: dict):
        self.config = config
        self._engine: LocalEngine | None = None
        self._current_model_key: str = ""
        self.history: list[dict] = []   # OpenAI-style message dicts

    # ── engine management ──────────────────────────────────────────────────────

    def _get_engine(self) -> LocalEngine:
        from gemma.models import get_model
        key = self.config.get("model", "e4b")
        if self._engine is None or self._current_model_key != key:
            if self._engine is not None:
                self._engine.unload()
            model = get_model(key)
            self._engine = LocalEngine(model.hf_id)
            self._engine.load()
            self._current_model_key = key
        return self._engine

    def reset(self):
        self.history = []

    # ── sending a message ──────────────────────────────────────────────────────

    def send(self, user_message: str):
        from gemma.tools import TOOL_DECLARATIONS, execute

        self.history.append({"role": "user", "content": user_message})
        engine = self._get_engine()

        temperature = self.config.get("temperature", 0.7)
        max_new_tokens = self.config.get("max_new_tokens", 2048)
        show_tools = self.config.get("show_tool_calls", True)

        # Gemma tokenizers expect bare function dicts, not the OpenAI wrapper
        tools_schema = [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            }
            for t in TOOL_DECLARATIONS
        ]

        # Agentic loop — keep going until the model returns no more tool calls
        while True:
            raw_response = self._stream_response(engine, tools_schema, temperature, max_new_tokens)

            clean_text, tool_calls = parse_tool_calls(raw_response)

            if not tool_calls:
                # Final answer — render as markdown
                if clean_text:
                    console.print()
                    console.print(Markdown(clean_text))
                    self.history.append({"role": "assistant", "content": clean_text})
                break

            # There are tool calls to execute
            if clean_text:
                console.print()
                console.print(Markdown(clean_text))

            self.history.append({"role": "assistant", "content": raw_response})

            tool_results = []
            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("arguments", {})

                if show_tools:
                    console.print(f"\n[bold cyan]◈[/] [dim]tool:[/] [bold green]{name}[/]")
                    for k, v in args.items():
                        vs = str(v)
                        console.print(f"  [dim]{k}:[/] {vs[:120]}{'…' if len(vs) > 120 else ''}")

                result = execute(name, args)

                if show_tools:
                    preview = result[:300] + "…" if len(result) > 300 else result
                    console.print(f"  [dim]→[/] {preview}\n")

                tool_results.append(f"[{name}] {result}")

            # Feed all tool results back as a single user message
            combined = "\n\n".join(tool_results)
            self.history.append({"role": "user", "content": f"<tool_response>\n{combined}\n</tool_response>"})

    # ── streaming helper ───────────────────────────────────────────────────────

    def _stream_response(
        self,
        engine: LocalEngine,
        tools_schema: list,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Stream tokens to stdout and return the full raw response string."""
        accumulated = []
        try:
            for token in engine.stream(
                self.history,
                tools=tools_schema,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            ):
                sys.stdout.write(token)
                sys.stdout.flush()
                accumulated.append(token)
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/]")

        sys.stdout.write("\n")
        sys.stdout.flush()
        return "".join(accumulated)
