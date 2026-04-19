"""
Local inference engine — downloads Gemma 4 models from HuggingFace and runs
them on whatever hardware is available (CUDA → MPS → CPU).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from threading import Thread
from typing import Iterator, Optional

from rich.console import Console

console = Console()


# ── device detection ──────────────────────────────────────────────────────────

def detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"


def get_dtype(device: str):
    import torch
    if device == "cpu":
        return torch.float32
    return torch.bfloat16


# ── download helpers ───────────────────────────────────────────────────────────

def is_downloaded(hf_id: str, cache_dir: Path) -> bool:
    """Check if the model weights are already in the local cache."""
    try:
        from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
        result = try_to_load_from_cache(hf_id, "config.json", cache_dir=str(cache_dir))
        return result not in (None, _CACHED_NO_EXIST)
    except Exception:
        # Fall back to a quick directory scan
        slug = hf_id.replace("/", "--")
        return any(cache_dir.glob(f"models--{slug}/snapshots/*/config.json"))


def pull(hf_id: str, cache_dir: Path):
    """Download model weights from HuggingFace with a progress display."""
    from huggingface_hub import snapshot_download

    console.print(f"\n[bold cyan]◈[/] Downloading [bold]{hf_id}[/]")
    console.print("[dim]  This only happens once — weights are cached in ~/.gemma/model_cache[/]\n")

    snapshot_download(
        repo_id=hf_id,
        cache_dir=str(cache_dir),
        # skip framework-specific weight files we don't need
        ignore_patterns=[
            "*.msgpack", "flax_model*", "tf_model*",
            "rust_model*", "model.safetensors.index.json",
        ],
    )

    console.print(f"\n[bold green]✓[/] Download complete.\n")


# ── inference engine ───────────────────────────────────────────────────────────

class LocalEngine:
    """Loads a Gemma 4 model locally and runs streaming generation."""

    def __init__(self, hf_id: str, cache_dir: Path):
        self.hf_id = hf_id
        self.cache_dir = cache_dir
        self.device = detect_device()
        self.tokenizer = None
        self.model = None
        self._loaded = False

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def ensure_downloaded(self):
        if not is_downloaded(self.hf_id, self.cache_dir):
            pull(self.hf_id, self.cache_dir)

    def load(self):
        if self._loaded:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.ensure_downloaded()

        console.print(f"[bold cyan]◈[/] Loading [bold]{self.hf_id}[/] → [bold]{self.device}[/]")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id,
            cache_dir=str(self.cache_dir),
        )

        dtype = get_dtype(self.device)

        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_id,
                cache_dir=str(self.cache_dir),
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_id,
                cache_dir=str(self.cache_dir),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True
        console.print(f"[bold green]✓[/] Model ready.\n")

    def unload(self):
        """Free GPU/RAM — useful when switching models."""
        import torch
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    # ── generation ─────────────────────────────────────────────────────────────

    def stream(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ) -> Iterator[str]:
        """Yield text tokens one by one as they are generated."""
        import torch
        from transformers import TextIteratorStreamer

        base_kwargs: dict = dict(
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
        )

        # Try with tools first; fall back silently if the tokenizer rejects them
        input_ids = None
        if tools:
            try:
                input_ids = self.tokenizer.apply_chat_template(
                    messages, tools=tools, **base_kwargs
                )
            except Exception:
                pass  # unsupported format — retry without tools below

        if input_ids is None:
            input_ids = self.tokenizer.apply_chat_template(messages, **base_kwargs)

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(self.device)
        else:
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        for token in streamer:
            yield token

        thread.join()

    def generate(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ) -> str:
        """Non-streaming generation — returns full response string."""
        return "".join(self.stream(messages, tools=tools, temperature=temperature, max_new_tokens=max_new_tokens))


# ── tool-call parsing ──────────────────────────────────────────────────────────
# Gemma 4 uses a <tool_call>…</tool_call> block convention when function
# calling is active.  We parse that out of the raw generated text.

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """
    Returns (clean_text, tool_calls) where tool_calls is a list of
    {'name': str, 'arguments': dict} dicts.  clean_text has the
    <tool_call> blocks stripped out.
    """
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
            calls.append({
                "name": payload.get("name", ""),
                "arguments": payload.get("arguments", payload.get("args", {})),
            })
        except json.JSONDecodeError:
            pass
    clean = _TOOL_CALL_RE.sub("", text).strip()
    return clean, calls
