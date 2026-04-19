# gemma-cli

![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white)
![MLX](https://img.shields.io/badge/MLX-0.31+-black?logo=apple&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-orange?logo=apple)
![Models](https://img.shields.io/badge/models-Gemma%204-4285F4?logo=google&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Local](https://img.shields.io/badge/runs-100%25%20local-brightgreen)
![Status](https://img.shields.io/badge/status-active-success)

> A blazingly fast, fully local CLI for Google's **Gemma 4** model family — powered by [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. No cloud. No API key. Just run `gemma`.

---

## Features

- **100% local** — weights run on-device via Apple's MLX framework
- **Streaming output** — tokens print as they generate, no waiting
- **File tools** — read, write, edit, search and grep files mid-conversation
- **Skills system** — slash commands (`/model`, `/pull`, `/reset`, `/help`, ...)
- **Model selection** — switch between Gemma 4 E2B, E4B, 12B, and 27B
- **Auto-download** — first run fetches the model weights automatically
- **Persistent history** — conversation history saved across sessions

---

## Requirements

| Requirement | Version |
|---|---|
| macOS | Sequoia 15+ (Apple Silicon) |
| Python | 3.11+ |
| MLX | 0.31+ |
| mlx-lm | 0.31+ |
| Disk space | 2.5 GB (E4B) — 15 GB (27B) |

---

## Installation

### 1. Install Python 3.11

```bash
brew install python@3.11
```

Add to your `~/.zshrc` so `python3` and `pip3` point to 3.11:

```bash
export PATH="/opt/homebrew/opt/python@3.11/bin:/opt/homebrew/bin:$PATH"
alias python3=python3.11
alias pip3=pip3.11
```

Then reload:

```bash
source ~/.zshrc
```

### 2. Install dependencies

```bash
pip3.11 install mlx-lm huggingface-hub rich prompt-toolkit
```

### 3. Clone and install gemma-cli

```bash
git clone https://github.com/yourname/gemma-cli
cd gemma-cli
pip3.11 install -e .
```

### 4. Run

```bash
gemma
```

On first run, the selected model is automatically downloaded from [mlx-community](https://huggingface.co/mlx-community) and cached in `~/.cache/huggingface/hub`.

---

## Models

| Key | Model | Size | Description |
|---|---|---|---|
| `e2b` | Gemma 4 E2B | ~1.4 GB | Smallest & fastest |
| `e4b` | Gemma 4 E4B | ~2.5 GB | Blazingly fast *(default)* |
| `12b` | Gemma 4 12B | ~7.0 GB | Balanced |
| `27b` | Gemma 4 27B | ~15 GB | Most capable |

Switch models with `/model <key>` inside the CLI, or pre-download one with `/pull <key>`.

---

## Skills (slash commands)

| Command | Description |
|---|---|
| `/help` | Show all available skills |
| `/model [key]` | Show models or switch to a different one |
| `/pull [key]` | Download a model without loading it |
| `/reset` | Clear conversation history |
| `/clear` | Clear the screen and redraw splash |
| `/tools` | List available file tools |
| `/config` | Show current configuration |
| `/exit` | Quit |

---

## File Tools

The model can read, write and search your filesystem mid-conversation:

| Tool | Description |
|---|---|
| `read_file` | Read file contents (with optional line range) |
| `write_file` | Write or create a file |
| `edit_file` | Find-and-replace within a file |
| `list_dir` | List directory contents |
| `search_files` | Glob pattern file search |
| `grep_files` | Regex content search across files |
| `run_command` | Execute a shell command |

---

## Configuration

Config is stored at `~/.gemma/config.json`:

```json
{
  "model": "e4b",
  "stream": true,
  "show_tool_calls": true,
  "temperature": 0.7,
  "max_new_tokens": 2048
}
```

---

## How It Works

```
gemma (CLI)
  └── prompt_toolkit REPL
        └── ChatSession
              └── LocalEngine (mlx-lm)
                    └── mlx-community/gemma-4-*-it-4bit
                          └── Apple Silicon (MLX / Metal)
```

1. You type a message
2. It's passed to the local Gemma 4 model via mlx-lm
3. Tokens stream back in real time
4. If the model calls a file tool, it executes locally and loops
5. Final response is rendered as Markdown

---

## Troubleshooting

**`Model type gemma4 not supported`**
Your mlx-lm is too old. Run:
```bash
pip3.11 install --upgrade mlx-lm
```

**`gemma` runs the wrong Python**
Check your PATH. The entry point must point to Python 3.11:
```bash
head -1 $(which gemma)
# should show: #!/opt/homebrew/opt/python@3.11/bin/python3.11
```

**Model downloads twice**
mlx-lm manages its own cache at `~/.cache/huggingface/hub`. To clear a model:
```bash
rm -rf ~/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit
```

---

## License

MIT
