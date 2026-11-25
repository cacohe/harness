from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from unicodedata import category

from src.adapters.artifacts.filesystem_platform import (
    ArtifactRepositoryError,
    Directory,
    directory_info,
    pin_root,
    regular_file,
)
from src.core.context.model import ModelMessage, ModelRole
from src.core.context.skills import (
    MAX_SKILL_CATALOG_ENTRIES,
    ResolvedSkill,
    SkillBinding,
    SkillCatalogEntry,
    skill_content_sha256,
    snapshot_skill_binding,
    snapshot_skill_catalog,
)
from src.core.ids import ProfileId

_MAX_CONFIGURED_SKILL_BYTES = 1_048_576
_CORE_MAX_CATALOG_BYTES = 65_536
_DOCUMENT_NAME = "SKILL.md"


class SkillProviderError(RuntimeError):
    """The Skill repository is unavailable, invalid, or no longer matches a binding."""


@dataclass(frozen=True, slots=True)
class _Document:
    entry: SkillCatalogEntry
    body: str
    digest: str


class FilesystemSkillProvider:
    def __init__(
        self,
        root: Path,
        *,
        max_skill_bytes: int = 262_144,
        max_catalog_entries: int = MAX_SKILL_CATALOG_ENTRIES,
        max_catalog_bytes: int = _CORE_MAX_CATALOG_BYTES,
    ) -> None:
        if not isinstance(root, Path) or not root.is_absolute():
            raise ValueError("Skill root must be an absolute Path")
        _positive_limit(max_skill_bytes, _MAX_CONFIGURED_SKILL_BYTES, "max_skill_bytes")
        _positive_limit(max_catalog_entries, MAX_SKILL_CATALOG_ENTRIES, "max_catalog_entries")
        _positive_limit(max_catalog_bytes, _CORE_MAX_CATALOG_BYTES, "max_catalog_bytes")
        try:
            root_info = directory_info(root.lstat())
            if root.resolve(strict=True) != root:
                raise ValueError("Skill root must not traverse a link")
        except (ArtifactRepositoryError, OSError):
            raise ValueError("Skill root must be an existing regular directory") from None
        self._root = root
        self._root_info = root_info
        self._max_skill_bytes = max_skill_bytes
        self._max_catalog_entries = max_catalog_entries
        self._max_catalog_bytes = max_catalog_bytes

    async def catalog(self, profile_id: ProfileId) -> tuple[SkillCatalogEntry, ...]:
        _profile(profile_id)
        return await asyncio.to_thread(self._catalog_sync)

    async def activate(self, name: str, *, profile_id: ProfileId) -> ResolvedSkill:
        _profile(profile_id)
        _logical_name(name)
        return await asyncio.to_thread(self._activate_sync, name)

    async def resolve(self, binding: SkillBinding, *, profile_id: ProfileId) -> ResolvedSkill:
        _profile(profile_id)
        try:
            pinned = snapshot_skill_binding(binding)
        except ValueError:
            raise SkillProviderError("Invalid Skill binding") from None
        _logical_name(pinned.name)
        return await asyncio.to_thread(self._resolve_sync, pinned)

    def _catalog_sync(self) -> tuple[SkillCatalogEntry, ...]:
        try:
            with pin_root(self._root, self._root_info) as root:
                directories = _skill_directories(root)
                if len(directories) > self._max_catalog_entries:
                    raise SkillProviderError("Skill catalog exceeds its entry limit")
                entries = tuple(
                    _read_document(
                        root,
                        name,
                        info,
                        self._max_skill_bytes,
                    ).entry
                    for name, info in directories
                )
            entries = tuple(sorted(entries, key=lambda entry: entry.name))
            if len({entry.name for entry in entries}) != len(entries):
                raise SkillProviderError("Skill catalog contains duplicate logical names")
            size = sum(
                len(value.encode("utf-8"))
                for entry in entries
                for value in (entry.name, entry.version, entry.description)
            )
            if size > self._max_catalog_bytes:
                raise SkillProviderError("Skill catalog exceeds its byte limit")
            return snapshot_skill_catalog(entries)
        except SkillProviderError:
            raise
        except (ArtifactRepositoryError, OSError, UnicodeError, ValueError):
            raise SkillProviderError("Skill catalog is invalid") from None

    def _activate_sync(self, name: str) -> ResolvedSkill:
        document = self._document(name)
        return _resolved(document)

    def _resolve_sync(self, binding: SkillBinding) -> ResolvedSkill:
        document = self._document(binding.name)
        if document.entry.version != binding.version or document.digest != binding.content_sha256:
            raise SkillProviderError("Pinned Skill content is no longer available")
        return _resolved(document)

    def _document(self, name: str) -> _Document:
        try:
            with pin_root(self._root, self._root_info) as root:
                info = directory_info(root.stat(name))
                return _read_document(root, name, info, self._max_skill_bytes)
        except SkillProviderError:
            raise
        except (ArtifactRepositoryError, OSError, UnicodeError, ValueError):
            raise SkillProviderError("Skill is unavailable or invalid") from None


def _skill_directories(root: Directory) -> tuple[tuple[str, os.stat_result], ...]:
    found: list[tuple[str, os.stat_result]] = []
    with root.entries() as entries:
        for entry in entries:
            _logical_name(entry.name)
            info = directory_info(root.stat(entry.name))
            found.append((entry.name, info))
    return tuple(sorted(found, key=lambda item: item[0]))


def _read_document(
    root: Directory,
    directory_name: str,
    expected_directory: os.stat_result,
    max_bytes: int,
) -> _Document:
    with root.child(directory_name, expected_directory) as directory:
        with directory.entries() as entries:
            names = tuple(entry.name for entry in entries)
        if names != (_DOCUMENT_NAME,):
            raise SkillProviderError("Skill directory must contain only SKILL.md")
        expected_file = regular_file(directory.stat(_DOCUMENT_NAME))
        with directory.read(_DOCUMENT_NAME, expected_file) as descriptor:
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(descriptor, min(65_536, max_bytes + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise SkillProviderError("Skill document exceeds its byte limit")
                chunks.append(chunk)
        encoded = b"".join(chunks)
    try:
        text = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise SkillProviderError("Skill document must be valid UTF-8") from None
    return _parse_document(directory_name, text)


def _parse_document(directory_name: str, text: str) -> _Document:
    if "\x00" in text:
        raise SkillProviderError("Skill document contains NUL")
    lines = text.splitlines()
    if len(lines) < 6 or lines[0] != "---" or lines[4] != "---":
        raise SkillProviderError("Skill frontmatter is malformed")
    values: dict[str, str] = {}
    for line in lines[1:4]:
        if ": " not in line:
            raise SkillProviderError("Skill frontmatter is malformed")
        key, value = line.split(": ", 1)
        if key not in {"name", "version", "description"} or key in values:
            raise SkillProviderError("Skill frontmatter fields are invalid")
        values[key] = value
    if set(values) != {"name", "version", "description"}:
        raise SkillProviderError("Skill frontmatter fields are incomplete")
    if values["name"] != directory_name:
        raise SkillProviderError("Skill declaration does not match its directory")
    _logical_name(values["name"])
    try:
        entry = SkillCatalogEntry(values["name"], values["description"], values["version"])
    except ValueError:
        raise SkillProviderError("Skill frontmatter values are invalid") from None
    body = "\n".join(lines[5:]).strip()
    if not body:
        raise SkillProviderError("Skill body must be non-blank")
    content = f"Skill: {entry.name}\nVersion: {entry.version}\n\n{body}"
    try:
        message = ModelMessage(ModelRole.SYSTEM, content)
    except ValueError:
        raise SkillProviderError("Skill body is invalid") from None
    assert message.content is not None
    digest = skill_content_sha256((message,))
    return _Document(entry, message.content, digest)


def _resolved(document: _Document) -> ResolvedSkill:
    binding = SkillBinding(
        document.entry.name,
        document.entry.version,
        document.digest,
    )
    return ResolvedSkill(binding, (ModelMessage(ModelRole.SYSTEM, document.body),))


def _logical_name(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or any(character in value for character in "/\\:")
        or any(category(character) == "Cc" for character in value)
    ):
        raise SkillProviderError("Skill name must be an opaque logical name")
    try:
        SkillBinding(value, "probe", "0" * 64)
    except ValueError:
        raise SkillProviderError("Skill name is invalid") from None
    return value


def _profile(value: object) -> ProfileId:
    if type(value) is not ProfileId:
        raise SkillProviderError("Skill profile is invalid")
    return ProfileId(value.value)


def _positive_limit(value: object, ceiling: int, label: str) -> int:
    if type(value) is not int or not 1 <= value <= ceiling:
        raise ValueError(f"{label} must be between 1 and {ceiling}")
    return value
