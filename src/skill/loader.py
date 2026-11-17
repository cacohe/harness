from pathlib import Path

from src.schema.skills import Skill
from src.schema.tool import Tool
from src.skill.directory import SkillDirectory


def load_skill_directories(
    roots: list[str | Path], *, include_shell: bool = False
) -> tuple[list[Skill], list[Tool]]:
    """Load unique Skills and their resource tools in declaration order."""
    skills: list[Skill] = []
    tools: list[Tool] = []
    names: set[str] = set()
    for root in roots:
        skill, resource_tools = SkillDirectory(root).load_bundle(include_shell=include_shell)
        if skill.name in names:
            raise ValueError(f"duplicate skill name: {skill.name!r}")
        names.add(skill.name)
        skills.append(skill)
        tools.extend(resource_tools)
    return skills, tools
