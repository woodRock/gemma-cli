import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".gemma"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history"
SKILLS_DIR = CONFIG_DIR / "skills"
MODEL_CACHE_DIR = CONFIG_DIR / "model_cache"

DEFAULT_CONFIG: dict = {
    "model": "e4b",
    "stream": True,
    "show_tool_calls": True,
    "temperature": 0.7,
    "max_new_tokens": 2048,
}


def ensure_dirs():
    CONFIG_DIR.mkdir(exist_ok=True)
    SKILLS_DIR.mkdir(exist_ok=True)
    MODEL_CACHE_DIR.mkdir(exist_ok=True)


def load() -> dict:
    ensure_dirs()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save(config: dict):
    ensure_dirs()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
