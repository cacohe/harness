from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from src.core.context.model import ModelMessage, ModelRole
from src.core.ids import ProfileId

MAX_SKILL_CATALOG_ENTRIES = 128
MAX_ACTIVE_SKILLS = 16
_MAX_SKILL_ID_BYTES = 128
_MAX_SKILL_DESCRIPTION_BYTES = 4_096
_MAX_SKILL_MESSAGES = 16


def _bounded_text(value: object, limit: int, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be non-blank canonical text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{label} must be valid UTF-8 text") from None
    if len(encoded) > limit:
        raise ValueError(f"{label} exceeds its UTF-8 byte limit")
    return value


@dataclass(frozen=True, slots=True)
class SkillCatalogEntry:
    name: str
    description: str
    version: str

    def __post_init__(self) -> None:
        _bounded_text(self.name, _MAX_SKILL_ID_BYTES, "Skill name")
        _bounded_text(self.description, _MAX_SKILL_DESCRIPTION_BYTES, "Skill description")
        _bounded_text(self.version, _MAX_SKILL_ID_BYTES, "Skill version")


@dataclass(frozen=True, slots=True)
class SkillBinding:
    name: str
    version: str
    content_sha256: str

    def __post_init__(self) -> None:
        _bounded_text(self.name, _MAX_SKILL_ID_BYTES, "Skill name")
        _bounded_text(self.version, _MAX_SKILL_ID_BYTES, "Skill version")
        if (
            type(self.content_sha256) is not str
            or len(self.content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.content_sha256)
        ):
            raise ValueError("Skill content digest must be lowercase SHA-256 hex")


@dataclass(frozen=True, slots=True)
class ResolvedSkill:
    binding: SkillBinding
    messages: tuple[ModelMessage, ...]

    def __post_init__(self) -> None:
        binding = snapshot_skill_binding(self.binding)
        if (
            type(self.messages) is not tuple
            or not 1 <= len(self.messages) <= _MAX_SKILL_MESSAGES
            or any(type(message) is not ModelMessage for message in self.messages)
            or any(message.role is not ModelRole.SYSTEM for message in self.messages)
        ):
            raise ValueError("Resolved Skill requires bounded system messages")
        messages = tuple(
            ModelMessage(
                object.__getattribute__(message, "role"),
                object.__getattribute__(message, "content"),
                object.__getattribute__(message, "tool_calls"),
                object.__getattribute__(message, "tool_call_id"),
            )
            for message in self.messages
        )
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "messages", messages)


class SkillProvider(Protocol):
    def catalog(self, profile_id: ProfileId) -> Awaitable[tuple[SkillCatalogEntry, ...]]: ...

    def activate(self, name: str, *, profile_id: ProfileId) -> Awaitable[ResolvedSkill]: ...

    def resolve(self, binding: SkillBinding, *, profile_id: ProfileId) -> Awaitable[ResolvedSkill]: ...


def snapshot_skill_catalog(value: object) -> tuple[SkillCatalogEntry, ...]:
    if type(value) is not tuple or len(value) > MAX_SKILL_CATALOG_ENTRIES:
        raise ValueError("Skill catalog must be a bounded tuple")
    entries = tuple(snapshot_skill_catalog_entry(entry) for entry in value)
    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("Skill catalog names must be unique")
    return entries


def snapshot_skill_catalog_entry(value: object) -> SkillCatalogEntry:
    if type(value) is not SkillCatalogEntry:
        raise ValueError("Skill catalog entry has an invalid type")
    try:
        return SkillCatalogEntry(
            object.__getattribute__(value, "name"),
            object.__getattribute__(value, "description"),
            object.__getattribute__(value, "version"),
        )
    except AttributeError:
        raise ValueError("Skill catalog entry is invalid") from None


def snapshot_skill_binding(value: object) -> SkillBinding:
    if type(value) is not SkillBinding:
        raise ValueError("Skill binding has an invalid type")
    try:
        return SkillBinding(
            object.__getattribute__(value, "name"),
            object.__getattribute__(value, "version"),
            object.__getattribute__(value, "content_sha256"),
        )
    except AttributeError:
        raise ValueError("Skill binding is invalid") from None


def snapshot_skill_bindings(value: object) -> tuple[SkillBinding, ...]:
    if type(value) is not tuple or len(value) > MAX_ACTIVE_SKILLS:
        raise ValueError("Active Skills must be a bounded tuple")
    bindings = tuple(snapshot_skill_binding(binding) for binding in value)
    if len({binding.name for binding in bindings}) != len(bindings):
        raise ValueError("Active Skill names must be unique")
    return bindings
