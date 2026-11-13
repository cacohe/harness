from __future__ import annotations

import hashlib
import os
from pathlib import Path, PureWindowsPath


class WorkspaceViolation(ValueError):
    """A path or workspace root violates the configured workspace boundary."""


class Workspace:
    """Resolve relative paths inside one immutable canonical workspace root."""

    def __init__(self, root: str | Path) -> None:
        try:
            resolved = Path(root).resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise WorkspaceViolation(
                f"workspace root is not an existing directory: {root}"
            ) from error
        if not resolved.is_dir():
            raise WorkspaceViolation(f"workspace root is not an existing directory: {root}")
        self.root = resolved
        normalized = os.path.normcase(str(resolved)).encode("utf-8")
        self.identity = hashlib.sha256(normalized).hexdigest()

    def resolve(self, path: str | Path = ".") -> Path:
        """Resolve one relative input and reject every path that escapes the root."""
        raw = os.fspath(path)
        host_path = Path(raw)
        windows_path = PureWindowsPath(raw)
        if (
            host_path.is_absolute()
            or bool(host_path.anchor)
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or bool(windows_path.root)
        ):
            raise WorkspaceViolation(f"absolute workspace path is not allowed: {raw}")
        if ".." in host_path.parts or ".." in windows_path.parts:
            raise WorkspaceViolation(f"parent traversal is not allowed: {raw}")
        try:
            resolved = (self.root / host_path).resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as error:
            raise WorkspaceViolation(f"workspace path cannot be resolved: {raw}") from error
        if not resolved.is_relative_to(self.root):
            raise WorkspaceViolation(f"workspace path escapes the root: {raw}")
        return resolved

    def relative(self, path: Path) -> str:
        """Render a canonical path without exposing the host workspace prefix."""
        try:
            resolved = path.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as error:
            raise WorkspaceViolation("workspace path cannot be rendered") from error
        if not resolved.is_relative_to(self.root):
            raise WorkspaceViolation("path is outside the workspace")
        rendered = resolved.relative_to(self.root).as_posix()
        return rendered or "."
