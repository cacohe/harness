import re
import subprocess
from pathlib import Path

from src.schema.tool import Tool

_MAX_READ_LINES = 2000
_MAX_MATCHES = 200
_MAX_OUTPUT_CHARS = 20000
_DEFAULT_TIMEOUT_SECONDS = 30


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    omitted = len(text) - _MAX_OUTPUT_CHARS
    return f"{text[:_MAX_OUTPUT_CHARS]}\n...[{omitted} characters omitted]"


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _error(message: str) -> str:
    return f"Error: {message}"


def shell(command: str, timeout: int = _DEFAULT_TIMEOUT_SECONDS, cwd: str | None = None) -> str:
    """
    Run a shell command and return its output.
    Args:
        command: Command line to run.
        timeout: Seconds before the command is stopped.
        cwd: Working directory. Defaults to the current directory.
    Returns:
        Exit code, stdout, and stderr.
    """
    workdir = _resolve(cwd) if cwd else Path.cwd()
    if not workdir.is_dir():
        return _error(f"directory does not exist: {workdir}")
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return _truncate(
            f"Exit code: timeout\nStdout:\n{exc.stdout or ''}\nStderr:\n{exc.stderr or ''}"
        )
    return _truncate(
        f"Exit code: {completed.returncode}\nStdout:\n{completed.stdout}\nStderr:\n{completed.stderr}"
    )


def read_file(path: str, offset: int = 1, limit: int | None = None) -> str:
    """
    Read a text file with line numbers.
    Args:
        path: File path.
        offset: First line to return. Lines are numbered from 1.
        limit: Maximum number of lines to return.
    Returns:
        Numbered file lines.
    """
    file_path = _resolve(path)
    if not file_path.is_file():
        return _error(f"file does not exist: {file_path}")
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(offset, 1) - 1
    window = _MAX_READ_LINES if limit is None else limit
    selected = lines[start : start + window]
    body = "\n".join(f"{index}|{line}" for index, line in enumerate(selected, start=start + 1))
    if start + window < len(lines):
        body += f"\n...[{len(lines) - (start + window)} more lines]"
    return body


def write_file(path: str, content: str) -> str:
    """
    Create or overwrite a text file.
    Args:
        path: File path.
        content: Full file contents.
    Returns:
        The number of bytes written.
    """
    file_path = _resolve(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Wrote {file_path} ({len(content.encode('utf-8'))} bytes)"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """
    Replace one exact occurrence of text in a file.
    Args:
        path: File path.
        old_string: Exact text to replace. It must occur once.
        new_string: Replacement text.
    Returns:
        A confirmation, or an error when the text is missing or repeated.
    """
    file_path = _resolve(path)
    if not old_string:
        return _error("old_string must not be empty")
    if not file_path.is_file():
        return _error(f"file does not exist: {file_path}")
    text = file_path.read_text(encoding="utf-8", errors="replace")
    count = text.count(old_string)
    if count == 0:
        return _error("old_string was not found")
    if count > 1:
        return _error(f"old_string is not unique ({count} occurrences)")
    file_path.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
    return f"Edited {file_path}"


def list_dir(path: str = ".") -> str:
    """
    List the entries in a directory.
    Args:
        path: Directory path. Defaults to the current directory.
    Returns:
        One entry per line. Directory names end with a slash.
    """
    directory = _resolve(path)
    if not directory.is_dir():
        return _error(f"directory does not exist: {directory}")
    entries = []
    for entry in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        name = f"{entry.name}/" if entry.is_dir() else entry.name
        entries.append(name)
    return "\n".join(entries) if entries else "(empty)"


def grep(pattern: str, path: str = ".", glob: str | None = None) -> str:
    """
    Search file contents for a regular expression.
    Args:
        pattern: Python regular expression.
        path: File or directory to search.
        glob: Optional file-name pattern, such as "*.py".
    Returns:
        Matching path:line:text lines.
    """
    try:
        expression = re.compile(pattern)
    except re.error as exc:
        return _error(f"invalid pattern: {exc}")
    root = _resolve(path)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = [item for item in root.rglob(glob or "*") if item.is_file()]
    else:
        return _error(f"path does not exist: {root}")

    matches = []
    for file_path in files:
        if any(part in {".git", ".venv", "__pycache__"} for part in file_path.parts):
            continue
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, start=1):
            if expression.search(line):
                matches.append(f"{file_path}:{number}:{line}")
                if len(matches) >= _MAX_MATCHES:
                    return _truncate("\n".join(matches) + "\n...[match limit reached]")
    return "\n".join(matches) if matches else "(no matches)"


def glob_files(pattern: str, path: str = ".") -> str:
    """
    Find files by name pattern.
    Args:
        pattern: Glob pattern, such as "**/*.py".
        path: Directory to search.
    Returns:
        Matching file paths, one per line.
    """
    root = _resolve(path)
    if not root.is_dir():
        return _error(f"directory does not exist: {root}")
    matches = []
    for item in root.glob(pattern):
        if not item.is_file():
            continue
        if any(part in {".git", ".venv", "__pycache__"} for part in item.parts):
            continue
        matches.append(str(item))
        if len(matches) >= _MAX_MATCHES:
            break
    matches.sort()
    if len(matches) >= _MAX_MATCHES:
        return "\n".join(matches) + "\n...[match limit reached]"
    return "\n".join(matches) if matches else "(no matches)"


def _bind(function):
    """Call a tool function with the argument dict passed by the executor."""

    def executer(inputs: dict | None = None, **kwargs):
        payload = dict(inputs or {})
        payload.update(kwargs)
        return function(**payload)

    return executer


def _define(name: str, description: str, parameters: dict, function) -> Tool:
    properties = {}
    required = []
    for key, value in parameters.items():
        spec = dict(value)
        if spec.pop("required", True):
            required.append(key)
        properties[key] = spec
    return Tool(
        id=name,
        name=name,
        description=description,
        input_schema={"type": "object", "properties": properties, "required": required},
        executer=_bind(function),
    )


SHELL_TOOL = _define(
    "shell",
    "Run a shell command in the workspace and return its exit code, stdout, and stderr.",
    {
        "command": {"type": "string", "description": "Command line to run."},
        "timeout": {
            "type": "integer",
            "description": "Seconds before the command is stopped.",
            "required": False,
        },
        "cwd": {"type": "string", "description": "Working directory.", "required": False},
    },
    shell,
)

READ_FILE_TOOL = _define(
    "read_file",
    "Read a text file. Lines are numbered so a later edit can target exact text.",
    {
        "path": {"type": "string", "description": "File path."},
        "offset": {
            "type": "integer",
            "description": "First line to return, starting at 1.",
            "required": False,
        },
        "limit": {"type": "integer", "description": "Maximum number of lines.", "required": False},
    },
    read_file,
)

WRITE_FILE_TOOL = _define(
    "write_file",
    "Create or overwrite a text file with the full contents.",
    {
        "path": {"type": "string", "description": "File path."},
        "content": {"type": "string", "description": "Full file contents."},
    },
    write_file,
)

EDIT_FILE_TOOL = _define(
    "edit_file",
    "Replace one exact occurrence of text in a file. old_string must match once.",
    {
        "path": {"type": "string", "description": "File path."},
        "old_string": {"type": "string", "description": "Exact text to replace."},
        "new_string": {"type": "string", "description": "Replacement text."},
    },
    edit_file,
)

LIST_DIR_TOOL = _define(
    "list_dir",
    "List files and directories in a directory.",
    {"path": {"type": "string", "description": "Directory path.", "required": False}},
    list_dir,
)

GREP_TOOL = _define(
    "grep",
    "Search file contents with a regular expression.",
    {
        "pattern": {"type": "string", "description": "Python regular expression."},
        "path": {
            "type": "string",
            "description": "File or directory to search.",
            "required": False,
        },
        "glob": {
            "type": "string",
            "description": "File-name pattern, such as *.py.",
            "required": False,
        },
    },
    grep,
)

GLOB_TOOL = _define(
    "glob",
    "Find files by a glob pattern.",
    {
        "pattern": {"type": "string", "description": "Glob pattern, such as **/*.py."},
        "path": {"type": "string", "description": "Directory to search.", "required": False},
    },
    glob_files,
)


LOCAL_TOOLS = [
    SHELL_TOOL,
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    EDIT_FILE_TOOL,
    LIST_DIR_TOOL,
    GREP_TOOL,
    GLOB_TOOL,
]
