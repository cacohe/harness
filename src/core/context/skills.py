import hashlib
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from src.core.context._skill_values import (
    MAX_ACTIVE_SKILLS,
    MAX_SKILL_ID_BYTES,
    SkillBinding,
    bounded_skill_text,
    snapshot_skill_binding,
    snapshot_skill_bindings,
)
from src.core.context.model import ModelMessage, ModelRole
from src.core.ids import ProfileId

__all__ = [
    "MAX_ACTIVE_SKILLS",
    "MAX_SKILL_CATALOG_ENTRIES",
    "ResolvedSkill",
    "SkillBinding",
    "SkillCatalogEntry",
    "SkillProvider",
    "snapshot_skill_binding",
    "snapshot_skill_bindings",
    "snapshot_skill_catalog",
    "snapshot_skill_catalog_entry",
    "skill_content_sha256",
]

MAX_SKILL_CATALOG_ENTRIES = 128
_MAX_SKILL_DESCRIPTION_BYTES = 4_096
_MAX_SKILL_MESSAGES = 16


@dataclass(frozen=True, slots=True)
class SkillCatalogEntry:
    name: str
    description: str
    version: str

    def __post_init__(self) -> None:
        bounded_skill_text(self.name, MAX_SKILL_ID_BYTES, "Skill name")
        bounded_skill_text(self.description, _MAX_SKILL_DESCRIPTION_BYTES, "Skill description")
        bounded_skill_text(self.version, MAX_SKILL_ID_BYTES, "Skill version")


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
        if skill_content_sha256(messages) != binding.content_sha256:
            raise ValueError("Resolved Skill content does not match its binding digest")
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "messages", messages)


class SkillProvider(Protocol):
    def catalog(self, profile_id: ProfileId) -> Awaitable[tuple[SkillCatalogEntry, ...]]: ...

    def activate(self, name: str, *, profile_id: ProfileId) -> Awaitable[ResolvedSkill]: ...

    def resolve(
        self, binding: SkillBinding, *, profile_id: ProfileId
    ) -> Awaitable[ResolvedSkill]: ...


def skill_content_sha256(messages: tuple[ModelMessage, ...]) -> str:
    if type(messages) is not tuple or not messages:
        raise ValueError("Skill content requires messages")
    digest = hashlib.sha256()
    for message in messages:
        if type(message) is not ModelMessage or message.role is not ModelRole.SYSTEM:
            raise ValueError("Skill content requires system messages")
        assert message.content is not None
        encoded = message.content.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


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
