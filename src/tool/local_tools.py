import inspect
import re
from functools import wraps
from pathlib import Path

from src.schema.tool import SandboxKind, Tool, ToolOrigin
from src.workspace import Workspace, WorkspaceViolation

_MAX_READ_LINES = 2000
_MAX_MATCHES = 200
_MAX_OUTPUT_CHARS = 20000


class LocalToolIOError(RuntimeError):
    """A local I/O failure whose message contains only workspace-relative context."""

    @property
    def public_message(self) -> str:
        return f"LocalToolIOError: {self}"


def _sanitize_io_errors(function):
    function_signature = inspect.signature(function)

    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except OSError as error:
            arguments = function_signature.bind_partial(*args, **kwargs).arguments
            path = arguments.get("path", ".")
            detail = error.strerror or type(error).__name__
            raise LocalToolIOError(
                f"{function.__name__} failed for workspace path {path}: {detail}"
            ) from None

    return guarded


class LocalToolSuite:
    """Create local tools whose filesystem authority is one explicit workspace."""

    __slots__ = ("_workspace",)
    _workspace: Workspace

    def __init__(self, workspace: Workspace) -> None:
        object.__setattr__(self, "_workspace", workspace)

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("LocalToolSuite workspace bindings are immutable")

    def tools(self, *, include_shell: bool = False) -> list[Tool]:
        tools = [
            self._define(
                "read_file",
                "Read a workspace text file. Lines are numbered for exact edits.",
                {
                    "path": {"type": "string", "description": "Workspace-relative file path."},
                    "offset": {
                        "type": "integer",
                        "description": "First line to return, starting at 1.",
                        "required": False,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines.",
                        "required": False,
                    },
                },
                self.read_file,
                input_validator=self._path_validator(),
            ),
            self._define(
                "write_file",
                "Create or overwrite a workspace text file.",
                {
                    "path": {"type": "string", "description": "Workspace-relative file path."},
                    "content": {"type": "string", "description": "Full file contents."},
                },
                self.write_file,
                input_validator=self._path_validator(),
            ),
            self._define(
                "edit_file",
                "Replace one exact occurrence in a workspace text file.",
                {
                    "path": {"type": "string", "description": "Workspace-relative file path."},
                    "old_string": {"type": "string", "description": "Exact text to replace."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                },
                self.edit_file,
                input_validator=self._path_validator(),
            ),
            self._define(
                "list_dir",
                "List files and directories inside the workspace.",
                {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory path.",
                        "required": False,
                    }
                },
                self.list_dir,
                input_validator=self._path_validator(default="."),
            ),
            self._define(
                "grep",
                "Search workspace file contents with a regular expression.",
                {
                    "pattern": {"type": "string", "description": "Python regular expression."},
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file or directory.",
                        "required": False,
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional file-name pattern, such as *.py.",
                        "required": False,
                    },
                },
                self.grep,
                input_validator=self._grep_validator,
            ),
            self._define(
                "glob",
                "Find workspace files by a glob pattern.",
                {
                    "pattern": {"type": "string", "description": "Glob pattern."},
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory.",
                        "required": False,
                    },
                },
                self.glob_files,
                input_validator=self._glob_validator,
            ),
        ]
        if include_shell:
            tools.append(self._shell_tool())
        return tools

    @_sanitize_io_errors
    def read_file(self, path: str, offset: int = 1, limit: int | None = None) -> str:
        file_path = self.workspace.resolve(path)
        if not file_path.is_file():
            return _error(f"file does not exist: {path}")
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(offset, 1) - 1
        window = _MAX_READ_LINES if limit is None else limit
        selected = lines[start : start + window]
        body = "\n".join(f"{index}|{line}" for index, line in enumerate(selected, start=start + 1))
        if start + window < len(lines):
            body += f"\n...[{len(lines) - (start + window)} more lines]"
        return body

    @_sanitize_io_errors
    def write_file(self, path: str, content: str) -> str:
        file_path = self.workspace.resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        display = self.workspace.relative(file_path)
        return f"Wrote {display} ({len(content.encode('utf-8'))} bytes)"

    @_sanitize_io_errors
    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        file_path = self.workspace.resolve(path)
        if not old_string:
            return _error("old_string must not be empty")
        if not file_path.is_file():
            return _error(f"file does not exist: {path}")
        text = file_path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old_string)
        if count == 0:
            return _error("old_string was not found")
        if count > 1:
            return _error(f"old_string is not unique ({count} occurrences)")
        file_path.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
        return f"Edited {self.workspace.relative(file_path)}"

    @_sanitize_io_errors
    def list_dir(self, path: str = ".") -> str:
        directory = self.workspace.resolve(path)
        if not directory.is_dir():
            return _error(f"directory does not exist: {path}")
        entries = []
        for entry in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            try:
                self.workspace.relative(entry)
            except WorkspaceViolation:
                continue
            name = f"{entry.name}/" if entry.is_dir() else entry.name
            entries.append(name)
        return "\n".join(entries) if entries else "(empty)"

    @_sanitize_io_errors
    def grep(self, pattern: str, path: str = ".", glob: str | None = None) -> str:
        try:
            expression = re.compile(pattern)
        except re.error as error:
            return _error(f"invalid pattern: {error}")
        root = self.workspace.resolve(path)
        if root.is_file():
            files = [root]
        elif root.is_dir():
            self._validate_pattern(glob or "*")
            files = [item for item in root.rglob(glob or "*") if item.is_file()]
        else:
            return _error(f"path does not exist: {path}")

        matches = []
        for file_path in files:
            display = self._safe_display(file_path)
            if display is None or _is_ignored(file_path):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, start=1):
                if expression.search(line):
                    matches.append(f"{display}:{number}:{line}")
                    if len(matches) >= _MAX_MATCHES:
                        return _truncate("\n".join(matches) + "\n...[match limit reached]")
        return "\n".join(matches) if matches else "(no matches)"

    @_sanitize_io_errors
    def glob_files(self, pattern: str, path: str = ".") -> str:
        self._validate_pattern(pattern)
        root = self.workspace.resolve(path)
        if not root.is_dir():
            return _error(f"directory does not exist: {path}")
        matches = []
        for item in root.glob(pattern):
            if not item.is_file() or _is_ignored(item):
                continue
            display = self._safe_display(item)
            if display is None:
                continue
            matches.append(display)
            if len(matches) >= _MAX_MATCHES:
                break
        matches.sort()
        if len(matches) >= _MAX_MATCHES:
            return "\n".join(matches) + "\n...[match limit reached]"
        return "\n".join(matches) if matches else "(no matches)"

    def _safe_display(self, path: Path) -> str | None:
        try:
            return self.workspace.relative(path)
        except WorkspaceViolation:
            return None

    def _path_validator(self, *, default: str | None = None):
        def validate(inputs: dict) -> dict:
            validated = dict(inputs)
            raw = validated.get("path", default)
            if raw is not None:
                validated["path"] = self.workspace.relative(self.workspace.resolve(raw))
            return validated

        return validate

    def _grep_validator(self, inputs: dict) -> dict:
        validated = self._path_validator(default=".")(inputs)
        if validated.get("glob") is not None:
            self._validate_pattern(validated["glob"])
        return validated

    def _glob_validator(self, inputs: dict) -> dict:
        validated = self._path_validator(default=".")(inputs)
        self._validate_pattern(validated["pattern"])
        return validated

    def _validate_pattern(self, pattern: str) -> None:
        self.workspace.resolve(pattern)

    def _shell_tool(self) -> Tool:
        return self._define(
            "shell",
            "Run a command inside the configured workspace sandbox.",
            {
                "command": {"type": "string", "description": "Command line to run."},
                "timeout": {
                    "type": "integer",
                    "description": "Seconds before the command is stopped.",
                    "required": False,
                },
                "cwd": {
                    "type": "string",
                    "description": "Workspace-relative working directory.",
                    "required": False,
                },
            },
            _sandbox_required,
            requires_sandbox=True,
            sandbox_kind=SandboxKind.COMMAND,
            input_validator=self._shell_validator,
        )

    def _shell_validator(self, inputs: dict) -> dict:
        validated = dict(inputs)
        raw_cwd = validated.get("cwd") or "."
        validated["cwd"] = self.workspace.relative(self.workspace.resolve(raw_cwd))
        return validated

    def _define(
        self,
        name: str,
        description: str,
        parameters: dict,
        function,
        *,
        requires_sandbox: bool = False,
        sandbox_kind: SandboxKind | None = None,
        input_validator=None,
    ) -> Tool:
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
            requires_sandbox=requires_sandbox,
            sandbox_kind=sandbox_kind,
            input_validator=input_validator,
            workspace=self.workspace,
            origin=ToolOrigin.LOCAL,
        )


def _bind(function):
    def executer(inputs: dict | None = None, **kwargs):
        payload = dict(inputs or {})
        payload.update(kwargs)
        return function(**payload)

    return executer


def _sandbox_required(**_inputs):
    raise RuntimeError("shell execution requires a sandbox runner")


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    omitted = len(text) - _MAX_OUTPUT_CHARS
    return f"{text[:_MAX_OUTPUT_CHARS]}\n...[{omitted} characters omitted]"


def _error(message: str) -> str:
    return f"Error: {message}"


def _is_ignored(path: Path) -> bool:
    return any(part in {".git", ".venv", "__pycache__"} for part in path.parts)
