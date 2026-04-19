from dataclasses import dataclass


@dataclass
class GemmaModel:
    key: str
    id: str
    name: str
    description: str
    context_window: int


MODELS: dict[str, GemmaModel] = {
    "e4b": GemmaModel(
        key="e4b",
        id="gemma-4-e4b-it",
        name="Gemma 4 E4B",
        description="4B efficient — blazingly fast",
        context_window=8192,
    ),
    "12b": GemmaModel(
        key="12b",
        id="gemma-4-12b-it",
        name="Gemma 4 12B",
        description="12B balanced — fast & capable",
        context_window=16384,
    ),
    "27b": GemmaModel(
        key="27b",
        id="gemma-4-27b-it",
        name="Gemma 4 27B",
        description="27B flagship — most capable",
        context_window=32768,
    ),
}

DEFAULT_MODEL = "e4b"


def get_model(key: str) -> GemmaModel:
    return MODELS.get(key, MODELS[DEFAULT_MODEL])
