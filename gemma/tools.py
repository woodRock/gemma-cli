import subprocess
from glob import glob as _glob
from pathlib import Path
from typing import Any

TOOL_DECLARATIONS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "start_line": {"type": "integer", "description": "Line to start from (1-indexed)"},
                "end_line": {"type": "integer", "description": "Line to end at (inclusive)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating it if it doesn't exist",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace an exact string in a file with a new string",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit"},
                "old_string": {"type": "string", "description": "Exact string to find and replace"},
                "new_string": {"type": "string", "description": "Replacement string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files and directories at a path",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: current directory)"},
            },
        },
    },
    {
        "name": "search_files",
        "description": "Search for files matching a glob pattern",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern e.g. '**/*.py'"},
                "path": {"type": "string", "description": "Base directory to search in"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep_files",
        "description": "Search for a text pattern inside files",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Text or regex pattern"},
                "path": {"type": "string", "description": "File or directory to search"},
                "case_insensitive": {"type": "boolean", "description": "Case insensitive search"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command and return its output",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "cwd": {"type": "string", "description": "Working directory"},
            },
            "required": ["command"],
        },
    },
]


def execute(name: str, args: dict[str, Any]) -> str:
    handlers = {
        "read_file": _read_file,
        "write_file": _write_file,
        "edit_file": _edit_file,
        "list_dir": _list_dir,
        "search_files": _search_files,
        "grep_files": _grep_files,
        "run_command": _run_command,
    }
    handler = handlers.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    try:
        return handler(**args)
    except Exception as e:
        return f"Error: {e}"


def _read_file(path: str, start_line: int = None, end_line: int = None) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"File not found: {path}"
    lines = p.read_text(errors="replace").splitlines(keepends=True)
    if start_line or end_line:
        s = (start_line or 1) - 1
        e = end_line or len(lines)
        lines = lines[s:e]
    content = "".join(lines)
    if len(content) > 100_000:
        content = content[:100_000] + "\n... (truncated)"
    return content


def _write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Written {len(content)} bytes to {path}"


def _edit_file(path: str, old_string: str, new_string: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"File not found: {path}"
    content = p.read_text()
    if old_string not in content:
        return f"String not found in {path}"
    count = content.count(old_string)
    p.write_text(content.replace(old_string, new_string, 1))
    note = f" ({count - 1} others skipped)" if count > 1 else ""
    return f"Replaced 1 occurrence in {path}{note}"


def _list_dir(path: str = ".") -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"Path not found: {path}"
    entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    lines = [("/" if e.is_dir() else " ") + " " + e.name for e in entries[:200]]
    if sum(1 for _ in p.iterdir()) > 200:
        lines.append("... (truncated)")
    return "\n".join(lines) or "(empty directory)"


def _search_files(pattern: str, path: str = ".") -> str:
    base = Path(path).expanduser()
    matches = sorted(_glob(str(base / pattern), recursive=True))
    if not matches:
        return "No files found"
    return "\n".join(matches[:100])


def _grep_files(pattern: str, path: str = ".", case_insensitive: bool = False) -> str:
    args = ["grep", "-r", "-n"]
    if case_insensitive:
        args.append("-i")
    args.extend([pattern, path])
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        output = (result.stdout + result.stderr)[:50_000]
        return output if output else "No matches found"
    except subprocess.TimeoutExpired:
        return "Search timed out"


def _run_command(command: str, cwd: str = None) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30, cwd=cwd
        )
        output = (result.stdout + result.stderr)[:50_000]
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30s"
