from pathlib import Path

import pytest
from src.skill import load_skill_directories


def _write_skill(root: Path, name: str) -> Path:
    root.mkdir()
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} description.\n---\nUse the resources.\n",
        encoding="utf-8",
    )
    return root


def test_load_skill_directories_returns_models_and_bound_tools(tmp_path):
    roots = [
        _write_skill(tmp_path / "first", "first"),
        _write_skill(tmp_path / "second", "second"),
    ]

    skills, tools = load_skill_directories(roots, include_shell=False)

    assert [skill.name for skill in skills] == ["first", "second"]
    assert [tool.name for tool in tools] == [
        "skill__first__read_file",
        "skill__second__read_file",
    ]


def test_load_skill_directories_rejects_duplicate_names(tmp_path):
    roots = [
        _write_skill(tmp_path / "first", "duplicate"),
        _write_skill(tmp_path / "second", "duplicate"),
    ]

    with pytest.raises(ValueError, match="duplicate skill name"):
        load_skill_directories(roots, include_shell=False)
