from pathlib import Path

import pytest
from src.core.workspace import Workspace, WorkspaceViolation
from src.tool.local_tools import LocalToolSuite


def _suite(tmp_path: Path) -> tuple[Path, LocalToolSuite]:
    root = tmp_path / "workspace"
    root.mkdir()
    return root, LocalToolSuite(Workspace(root))


def test_local_suite_exposes_six_file_tools_without_shell(tmp_path):
    _, suite = _suite(tmp_path)

    tools = suite.tools()

    assert {tool.name for tool in tools} == {
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "grep",
        "glob",
    }
    assert all(tool.workspace is suite.workspace for tool in tools)
    assert all(not tool.requires_sandbox for tool in tools)


@pytest.mark.parametrize("attribute", ["workspace", "_workspace"])
def test_local_suite_workspace_binding_is_immutable(tmp_path, attribute):
    _, suite = _suite(tmp_path)
    other = tmp_path / "other"
    other.mkdir()

    with pytest.raises(AttributeError, match="immutable"):
        setattr(suite, attribute, Workspace(other))


def test_bound_write_read_edit_list_grep_and_glob_stay_relative(tmp_path):
    root, suite = _suite(tmp_path)

    assert suite.write_file("notes/plan.txt", "buy a phone\nbuy a case\n") == (
        "Wrote notes/plan.txt (23 bytes)"
    )
    assert "1|buy a phone" in suite.read_file("notes/plan.txt")
    assert suite.edit_file("notes/plan.txt", "buy a phone", "buy a laptop") == (
        "Edited notes/plan.txt"
    )
    assert "buy a laptop" in suite.read_file("notes/plan.txt")
    assert suite.list_dir("notes") == "plan.txt"

    grep_output = suite.grep("laptop", "notes")
    glob_output = suite.glob_files("**/*.txt")

    assert grep_output == "notes/plan.txt:1:buy a laptop"
    assert glob_output == "notes/plan.txt"
    assert str(root) not in grep_output
    assert str(root) not in glob_output


@pytest.mark.parametrize("path", ["../outside.txt", r"..\outside.txt", r"C:\outside.txt"])
def test_bound_tools_reject_absolute_and_parent_paths(tmp_path, path):
    _, suite = _suite(tmp_path)

    with pytest.raises(WorkspaceViolation):
        suite.read_file(path)
    with pytest.raises(WorkspaceViolation):
        suite.write_file(path, "blocked")


def test_search_skips_outside_symlink_targets(tmp_path):
    root, suite = _suite(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret marker", encoding="utf-8")
    _symlink_or_skip(root / "linked.txt", outside)

    grep_output = suite.grep("secret", ".")
    glob_output = suite.glob_files("**/*.txt")

    assert grep_output == "(no matches)"
    assert glob_output == "(no matches)"
    assert str(outside) not in grep_output
    assert str(outside) not in glob_output


@pytest.mark.parametrize("pattern", ["../*.txt", r"..\*.txt", r"C:\*.txt", "/*.txt"])
def test_glob_rejects_absolute_and_parent_patterns(tmp_path, pattern):
    _, suite = _suite(tmp_path)

    with pytest.raises(WorkspaceViolation):
        suite.glob_files(pattern)


def test_shell_tool_is_opt_in_and_requires_sandbox(tmp_path):
    _, suite = _suite(tmp_path)

    [shell] = [tool for tool in suite.tools(include_shell=True) if tool.name == "shell"]

    assert shell.requires_sandbox is True
    assert shell.workspace is suite.workspace
    assert all(tool.name != "shell" for tool in suite.tools())


def test_shell_validator_normalizes_omitted_and_nested_cwd(tmp_path):
    root, suite = _suite(tmp_path)
    (root / "nested").mkdir()
    [shell] = [tool for tool in suite.tools(include_shell=True) if tool.name == "shell"]

    assert shell.input_validator({"command": "dir"}) == {"command": "dir", "cwd": "."}
    assert shell.input_validator({"command": "dir", "cwd": "nested"}) == {
        "command": "dir",
        "cwd": "nested",
    }


def test_shell_callable_never_executes_a_host_process(tmp_path):
    _, suite = _suite(tmp_path)
    [shell] = [tool for tool in suite.tools(include_shell=True) if tool.name == "shell"]

    with pytest.raises(RuntimeError, match="sandbox"):
        shell.executer({"command": "dir"})


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")
