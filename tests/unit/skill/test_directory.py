from pathlib import Path

import pytest
from src.core.workspace import Workspace, WorkspaceViolation
from src.skill.directory import SkillDirectory, SkillDocumentError


def _skill_dir(tmp_path: Path, document: bytes | str | None) -> Path:
    root = tmp_path / "skill"
    root.mkdir()
    if isinstance(document, str):
        (root / "SKILL.md").write_text(document, encoding="utf-8", newline="")
    elif document is not None:
        (root / "SKILL.md").write_bytes(document)
    return root


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_load_parses_frontmatter_and_unicode_body(tmp_path, newline):
    text = newline.join(
        [
            "---",
            "name: research helper",
            "description: |",
            "  Finds reliable sources.",
            "  Summarizes evidence.",
            "---",
            "# 工作流",
            "先阅读 references/source.md。",
            "",
        ]
    )
    directory = SkillDirectory(_skill_dir(tmp_path, text))

    skill = directory.load()

    assert skill.name == "research helper"
    assert skill.description == "Finds reliable sources.\nSummarizes evidence.\n"
    assert skill.content.startswith("# 工作流\n先阅读 references/source.md。")
    assert "name: research helper" not in skill.content


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("---\nname: demo\ndescription: ok\n", "closing frontmatter"),
        ("---\nname: [\ndescription: ok\n---\nbody", "invalid YAML"),
        ("---\nname: demo\ndescription: 2026-99-99\n---\nbody", "invalid YAML"),
        ("---\nname: demo\ndescription: !!int invalid\n---\nbody", "invalid YAML"),
        ("---\n- name\n- description\n---\nbody", "mapping"),
        ("---\ndescription: ok\n---\nbody", "name"),
        ("---\nname: '  '\ndescription: ok\n---\nbody", "name"),
        ("---\nname: 42\ndescription: ok\n---\nbody", "name"),
        ("---\nname: demo\n---\nbody", "description"),
        ("---\nname: demo\ndescription: []\n---\nbody", "description"),
        ("body without frontmatter", "frontmatter"),
    ],
)
def test_load_rejects_malformed_skill_documents_without_leaking_root(tmp_path, document, message):
    root = _skill_dir(tmp_path, document)

    with pytest.raises(SkillDocumentError, match=message) as caught:
        SkillDirectory(root).load()

    assert str(root) not in str(caught.value)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (None, "read SKILL.md"),
        (b"---\nname: demo\ndescription: ok\n---\n\xff", "UTF-8"),
    ],
)
def test_load_wraps_document_read_errors_with_relative_context(tmp_path, document, message):
    root = _skill_dir(tmp_path, document)

    with pytest.raises(SkillDocumentError, match=message) as caught:
        SkillDirectory(root).load()

    assert str(root) not in str(caught.value)


def test_resource_tools_are_read_only_by_default_and_named_in_skill_content(tmp_path):
    root = _skill_dir(
        tmp_path,
        "---\nname: research helper\ndescription: Finds sources.\n---\nFollow the workflow.\n",
    )
    (root / "references").mkdir()
    (root / "references" / "source.md").write_text("trusted", encoding="utf-8")
    directory = SkillDirectory(root)

    tools = directory.tools()
    skill = directory.load()

    assert [tool.name for tool in tools] == ["skill__research_helper__read_file"]
    [read_file] = tools
    assert read_file.workspace is directory.workspace
    assert read_file.requires_sandbox is False
    assert read_file.executer({"path": "references/source.md"}) == "1|trusted"
    assert "skill__research_helper__read_file" in skill.content
    assert "skill__research_helper__shell" not in skill.content


@pytest.mark.parametrize(
    "path",
    ["../outside.txt", r"..\outside.txt", r"C:\outside.txt", "NUL.txt"],
)
def test_resource_read_tool_rejects_paths_outside_the_skill(tmp_path, path):
    root = _skill_dir(tmp_path, "---\nname: demo\ndescription: Demo.\n---\nbody")
    [read_file] = SkillDirectory(root).tools()

    with pytest.raises(WorkspaceViolation):
        assert read_file.input_validator is not None
        read_file.input_validator({"path": path})


def test_resource_read_tool_rejects_an_escaping_symlink(tmp_path):
    root = _skill_dir(tmp_path, "---\nname: demo\ndescription: Demo.\n---\nbody")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        (root / "linked.txt").symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")
    [read_file] = SkillDirectory(root).tools()

    with pytest.raises(WorkspaceViolation):
        assert read_file.input_validator is not None
        read_file.input_validator({"path": "linked.txt"})


def test_optional_shell_tool_is_sandbox_required_and_scoped_to_skill(tmp_path):
    root = _skill_dir(
        tmp_path,
        "---\nname: script runner\ndescription: Runs scripts.\n---\nUse scripts/check.sh.\n",
    )
    (root / "scripts").mkdir()
    directory = SkillDirectory(root)

    tools = directory.tools(include_shell=True)
    skill = directory.load(include_shell=True)

    assert [tool.name for tool in tools] == [
        "skill__script_runner__read_file",
        "skill__script_runner__shell",
    ]
    shell = tools[1]
    assert shell.workspace is directory.workspace
    assert shell.requires_sandbox is True
    assert shell.input_validator is not None
    assert shell.input_validator({"command": "bash scripts/check.sh"}) == {
        "command": "bash scripts/check.sh",
        "cwd": ".",
    }
    assert shell.input_validator({"command": "dir", "cwd": "scripts"}) == {
        "command": "dir",
        "cwd": "scripts",
    }
    assert "skill__script_runner__shell" in skill.content
    with pytest.raises(RuntimeError, match="sandbox"):
        shell.executer({"command": "bash scripts/check.sh"})


@pytest.mark.parametrize("attribute", ["workspace", "_workspace"])
def test_skill_directory_workspace_binding_is_immutable(tmp_path, attribute):
    root = _skill_dir(tmp_path, "---\nname: demo\ndescription: Demo.\n---\nbody")
    directory = SkillDirectory(root)
    other = tmp_path / "other"
    other.mkdir()

    with pytest.raises(AttributeError, match="immutable"):
        setattr(directory, attribute, Workspace(other))
