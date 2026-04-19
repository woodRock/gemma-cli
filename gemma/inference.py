"""
Local inference engine — runs Gemma 4 models via mlx-lm on Apple Silicon.
Models are downloaded and cached by mlx-lm in ~/.cache/huggingface/hub.
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Iterator, Optional

from rich.console import Console

console = Console()


# ── download helpers ──────────────────────────────────────────────────────────

def is_downloaded(hf_id: str) -> bool:
    """Check if the model is already in the HuggingFace hub cache."""
    try:
        from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
        result = try_to_load_from_cache(hf_id, "config.json")
        return result not in (None, _CACHED_NO_EXIST)
    except Exception:
        slug = hf_id.replace("/", "--")
        default = Path.home() / ".cache" / "huggingface" / "hub"
        return any(default.glob(f"models--{slug}/snapshots/*/config.json"))


def pull(hf_id: str):
    """Pre-download a model without loading it into memory."""
    console.print(f"\n[bold cyan]◈[/] Downloading [bold]{hf_id}[/]")
    console.print("[dim]  Cached in ~/.cache/huggingface/hub — only downloaded once.[/]\n")
    # Load and immediately discard — mlx-lm handles caching
    from mlx_lm import load as mlx_load
    _patch_extra_special_tokens()
    mlx_load(hf_id)
    console.print(f"\n[bold green]✓[/] Download complete.\n")


# ── tokenizer compatibility patch ─────────────────────────────────────────────

def _patch_extra_special_tokens():
    """
    Gemma 4 stores extra_special_tokens as a list in tokenizer_config.json.
    Transformers expects a dict and calls .keys() on it — patch to handle both.
    """
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        original = PreTrainedTokenizerBase._set_model_specific_special_tokens

        def _patched(self, special_tokens):
            if isinstance(special_tokens, list):
                return
            return original(self, special_tokens=special_tokens)

        PreTrainedTokenizerBase._set_model_specific_special_tokens = _patched
    except Exception:
        pass


# ── inference engine ──────────────────────────────────────────────────────────

class LocalEngine:
    """Loads a Gemma 4 model via mlx-lm and runs streaming generation."""

    def __init__(self, hf_id: str):
        self.hf_id = hf_id
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        try:
            from mlx_lm import load as mlx_load
        except ImportError:
            console.print("[bold red]Error:[/] mlx-lm is not installed.")
            console.print("[dim]Run: pip install mlx-lm[/]")
            raise SystemExit(1)

        _patch_extra_special_tokens()
        console.print(f"[bold cyan]◈[/] Loading [bold]{self.hf_id}[/] via mlx-lm")

        self.model, self.tokenizer = mlx_load(self.hf_id)

        self._loaded = True
        console.print(f"[bold green]✓[/] Model ready.\n")

    def unload(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False
        gc.collect()

    def stream(
        self,
        messages: list,
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ) -> Iterator[str]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self._build_prompt(messages, tools)
        sampler = make_sampler(temp=temperature)

        for chunk in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
        ):
            yield chunk.text

    def generate(
        self,
        messages: list,
        tools: Optional[list] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ) -> str:
        return "".join(self.stream(messages, tools=tools, temperature=temperature, max_new_tokens=max_new_tokens))

    def _build_prompt(self, messages: list, tools: Optional[list]) -> str:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if tools:
            try:
                return self.tokenizer.apply_chat_template(messages, tools=tools, **kwargs)
            except Exception:
                pass
        return self.tokenizer.apply_chat_template(messages, **kwargs)


# ── tool-call / thought parsing ───────────────────────────────────────────────
# Gemma 4 native format:
#   thoughts  →  <|channel>thought ... <channel|>
#   tool call →  <|tool_call>call:TOOL{key:<|"|>val<|"|>,...}<tool_call|>

_THOUGHT_RE   = re.compile(r"<\|channel\s*>thought(.*?)<channel\s*\|>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<\|tool_call\s*>call:(\w+)\{(.*?)\}<tool_call\s*\|>", re.DOTALL)
# Handles both quoted  →  key:<|"|>value<|"|>
#        and unquoted  →  key:value  (up to next comma or closing brace)
_PARAM_RE = re.compile(
    r'(\w+):'
    r'(?:<\|"\|>(.*?)<\|"\|>|([^,}\n<][^,}\n]*))',
    re.DOTALL,
)


def _clean_value(v: str) -> str:
    """Strip markdown backtick formatting that the model sometimes wraps values in."""
    return v.strip().strip("`")


def parse_tool_calls(text: str) -> tuple[str, list]:
    """Return (clean_text, tool_calls) from a raw Gemma 4 response."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        tool_name  = m.group(1)
        params_raw = m.group(2)
        # group(2) = quoted value, group(3) = unquoted value
        args = {
            pm.group(1): _clean_value(
                pm.group(2) if pm.group(2) is not None else pm.group(3).strip()
            )
            for pm in _PARAM_RE.finditer(params_raw)
        }
        calls.append({"name": tool_name, "arguments": args})

    clean = _TOOL_CALL_RE.sub("", text)
    clean = _THOUGHT_RE.sub("", clean).strip()
    return clean, calls


def extract_thoughts(text: str) -> list[str]:
    return [m.group(1).strip() for m in _THOUGHT_RE.finditer(text)]
