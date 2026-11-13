from pathlib import Path

import pytest
from src.core.workspace import Workspace, WorkspaceViolation


def test_workspace_requires_an_existing_directory(tmp_path):
    missing = tmp_path / "missing"
    regular_file = tmp_path / "file.txt"
    regular_file.write_text("content", encoding="utf-8")

    with pytest.raises(WorkspaceViolation):
        Workspace(missing)
    with pytest.raises(WorkspaceViolation):
        Workspace(regular_file)


def test_workspace_resolves_relative_paths_and_has_stable_identity(tmp_path):
    root = tmp_path / "workspace"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()

    first = Workspace(root)
    second = Workspace(str(root))

    assert first.root == root.resolve()
    assert first.resolve("nested/file.txt") == (root / "nested" / "file.txt").resolve()
    assert first.identity == second.identity
    assert first.identity != Workspace(other).identity


@pytest.mark.parametrize("attribute", ["root", "identity", "_root", "_identity"])
def test_workspace_capability_is_immutable(tmp_path, attribute):
    workspace = Workspace(tmp_path)

    with pytest.raises(AttributeError, match="immutable"):
        setattr(workspace, attribute, tmp_path)


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        r"..\outside",
        r"C:\outside",
        r"\\server\share\outside",
        "/outside",
    ],
)
def test_workspace_rejects_absolute_and_parent_paths(tmp_path, path):
    root = tmp_path / "workspace"
    root.mkdir()

    with pytest.raises(WorkspaceViolation):
        Workspace(root).resolve(path)


@pytest.mark.parametrize(
    "path",
    ["NUL", "nul.txt", "nested/COM1.log", "AUX ", "file:stream"],
)
def test_workspace_rejects_windows_reserved_components(tmp_path, path):
    with pytest.raises(WorkspaceViolation, match="reserved Windows path"):
        Workspace(tmp_path).resolve(path)


def test_workspace_relative_never_returns_a_host_absolute_path(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = Workspace(root)

    assert workspace.relative(workspace.resolve("nested/file.txt")) == "nested/file.txt"
    with pytest.raises(WorkspaceViolation):
        workspace.relative(tmp_path / "outside.txt")


def test_workspace_rejects_file_symlink_outside_root(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = root / "link.txt"
    _symlink_or_skip(link, outside)

    with pytest.raises(WorkspaceViolation):
        Workspace(root).resolve("link.txt")


def test_workspace_rejects_missing_leaf_under_outside_symlink(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "linked"
    _symlink_or_skip(link, outside, target_is_directory=True)

    with pytest.raises(WorkspaceViolation):
        Workspace(root).resolve("linked/new.txt")


def _symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")
