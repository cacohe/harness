from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from src.schema.skills import Skill
from src.schema.tool import Tool
from src.tool.local_tools import LocalToolSuite
from src.utils.tool_names import provider_tool_name
from src.workspace import Workspace

_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", re.DOTALL)


class SkillDocumentError(ValueError):
    """A directory does not contain a valid, readable SKILL.md document."""


class SkillDirectory:
    """Load one skill from a directory bound to an immutable workspace."""

    __slots__ = ("_workspace",)
    _workspace: Workspace

    def __init__(self, root: str | Path) -> None:
        object.__setattr__(self, "_workspace", Workspace(root))

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SkillDirectory workspace binding is immutable")

    def load(self, *, include_shell: bool = False) -> Skill:
        """Parse this directory's SKILL.md into a data-only Skill."""
        return self._parse_document(self._read_document(), include_shell=include_shell)

    def load_bundle(self, *, include_shell: bool = False) -> tuple[Skill, list[Tool]]:
        """Load one Skill and derive its resource tools from the same document read."""
        skill = self._parse_document(self._read_document(), include_shell=include_shell)
        return skill, self._tools_for(skill.name, include_shell=include_shell)

    def _parse_document(self, text: str, *, include_shell: bool) -> Skill:
        if not (text.startswith("---\n") or text.startswith("---\r\n")):
            raise SkillDocumentError("SKILL.md must start with YAML frontmatter")
        match = _FRONTMATTER.match(text)
        if match is None:
            raise SkillDocumentError("SKILL.md is missing its closing frontmatter delimiter")
        metadata = self._load_metadata(f"{match.group(1)}\n")
        name = self._required_text(metadata, "name")
        description = self._required_text(metadata, "description")
        body = text[match.end() :]
        return Skill(
            name=name,
            description=description,
            content=self._with_resource_note(body, name, include_shell=include_shell),
        )

    def tools(self, *, include_shell: bool = False) -> list[Tool]:
        """Return read-only and optional sandbox-required tools for this skill."""
        skill_name = self.load().name
        return self._tools_for(skill_name, include_shell=include_shell)

    def _tools_for(self, skill_name: str, *, include_shell: bool) -> list[Tool]:
        wanted = {"read_file"}
        if include_shell:
            wanted.add("shell")
        tools = []
        for tool in LocalToolSuite(self.workspace).tools(include_shell=include_shell):
            if tool.name not in wanted:
                continue
            name = provider_tool_name("skill", skill_name, tool.name)
            description = f"[{skill_name}] {tool.description}"
            tools.append(
                replace(
                    tool,
                    id=name,
                    name=name,
                    description=description,
                    requires_approval=tool.requires_approval or tool.name == "shell",
                )
            )
        return tools

    def _read_document(self) -> str:
        document = self.workspace.resolve("SKILL.md")
        try:
            return document.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise SkillDocumentError("SKILL.md is not valid UTF-8") from error
        except OSError as error:
            detail = error.strerror or type(error).__name__
            raise SkillDocumentError(f"could not read SKILL.md: {detail}") from None

    @staticmethod
    def _load_metadata(source: str) -> dict[str, Any]:
        try:
            metadata = yaml.safe_load(source)
        except (yaml.YAMLError, ValueError, TypeError, OverflowError) as error:
            raise SkillDocumentError("SKILL.md contains invalid YAML frontmatter") from error
        if not isinstance(metadata, dict):
            raise SkillDocumentError("SKILL.md frontmatter must be a mapping")
        return metadata

    @staticmethod
    def _required_text(metadata: dict[str, Any], field: str) -> str:
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SkillDocumentError(
                f"SKILL.md frontmatter field {field!r} must be a non-empty string"
            )
        return value.strip() if "\n" not in value else value

    @staticmethod
    def _with_resource_note(body: str, name: str, *, include_shell: bool) -> str:
        read_name = provider_tool_name("skill", name, "read_file")
        lines = [
            "Skill resources:",
            f"- Use `{read_name}` to read files relative to this skill directory.",
        ]
        if include_shell:
            shell_name = provider_tool_name("skill", name, "shell")
            lines.append(
                f"- Use `{shell_name}` to run commands in this skill's configured sandbox."
            )
        authored = body.rstrip()
        note = "\n".join(lines)
        return f"{authored}\n\n{note}" if authored else note
