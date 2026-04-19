import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".gemma"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history"
SKILLS_DIR = CONFIG_DIR / "skills"

DEFAULT_CONFIG: dict = {
    "model": "e4b",
    "api_key": None,
    "stream": True,
    "show_tool_calls": True,
}


def ensure_dirs():
    CONFIG_DIR.mkdir(exist_ok=True)
    SKILLS_DIR.mkdir(exist_ok=True)


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


def get_api_key(config=None):
    if config is None:
        config = load()
    return (
        config.get("api_key")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
