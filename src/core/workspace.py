from __future__ import annotations

import hashlib
import os
from pathlib import Path, PureWindowsPath


class WorkspaceViolation(ValueError):
    """A path or workspace root violates the configured workspace boundary."""


class Workspace:
    """Resolve relative paths inside one immutable canonical workspace root."""

    __slots__ = ("_root", "_identity")
    _root: Path
    _identity: str
    _WINDOWS_DEVICES = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "CONIN$",
        "CONOUT$",
        *(f"COM{suffix}" for suffix in "123456789¹²³"),
        *(f"LPT{suffix}" for suffix in "123456789¹²³"),
    }

    def __init__(self, root: str | Path) -> None:
        try:
            resolved = Path(root).resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise WorkspaceViolation(
                f"workspace root is not an existing directory: {root}"
            ) from error
        if not resolved.is_dir():
            raise WorkspaceViolation(f"workspace root is not an existing directory: {root}")
        object.__setattr__(self, "_root", resolved)
        normalized = os.path.normcase(str(resolved)).encode("utf-8")
        object.__setattr__(self, "_identity", hashlib.sha256(normalized).hexdigest())

    @property
    def root(self) -> Path:
        return self._root

    @property
    def identity(self) -> str:
        return self._identity

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Workspace capabilities are immutable")

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
        if any(self._is_reserved_windows_component(part) for part in windows_path.parts):
            raise WorkspaceViolation(f"reserved Windows path is not allowed: {raw}")
        try:
            resolved = (self.root / host_path).resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as error:
            raise WorkspaceViolation(f"workspace path cannot be resolved: {raw}") from error
        if not resolved.is_relative_to(self.root):
            raise WorkspaceViolation(f"workspace path escapes the root: {raw}")
        return resolved

    @classmethod
    def _is_reserved_windows_component(cls, component: str) -> bool:
        trimmed = component.rstrip(" .")
        if not trimmed or trimmed != component or ":" in component:
            return True
        stem = trimmed.split(".", 1)[0].upper()
        return stem in cls._WINDOWS_DEVICES

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
