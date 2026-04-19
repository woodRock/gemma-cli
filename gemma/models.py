from dataclasses import dataclass


@dataclass
class GemmaModel:
    key: str
    hf_id: str              # HuggingFace repo ID (mlx-community quantized)
    name: str
    description: str
    context_window: int
    size_gb: float          # approximate download size (4-bit quantized)


MODELS: dict[str, GemmaModel] = {
    "e2b": GemmaModel(
        key="e2b",
        hf_id="mlx-community/gemma-4-e2b-it-4bit",
        name="Gemma 4 E2B",
        description="2B efficient — smallest & fastest",
        context_window=128_000,
        size_gb=1.4,
    ),
    "e4b": GemmaModel(
        key="e4b",
        hf_id="mlx-community/gemma-4-e4b-it-4bit",
        name="Gemma 4 E4B",
        description="4B efficient — blazingly fast",
        context_window=128_000,
        size_gb=2.5,
    ),
    "12b": GemmaModel(
        key="12b",
        hf_id="mlx-community/gemma-4-12b-it-4bit",
        name="Gemma 4 12B",
        description="12B balanced — fast & capable",
        context_window=128_000,
        size_gb=7.0,
    ),
    "27b": GemmaModel(
        key="27b",
        hf_id="mlx-community/gemma-4-27b-it-4bit",
        name="Gemma 4 27B",
        description="27B flagship — most capable",
        context_window=128_000,
        size_gb=15.0,
    ),
}

DEFAULT_MODEL = "e4b"


def get_model(key: str) -> GemmaModel:
    return MODELS.get(key, MODELS[DEFAULT_MODEL])
