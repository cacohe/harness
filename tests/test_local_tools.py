import shutil
from pathlib import Path

from src.tool.local_tools import (
    LOCAL_TOOLS,
    edit_file,
    grep,
    list_dir,
    read_file,
    shell,
    write_file,
)


def test_local_tools_cover_the_agent_file_and_shell_set():
    assert {tool.name for tool in LOCAL_TOOLS} == {
        "shell",
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "grep",
        "glob",
    }


def test_write_read_edit_and_search(monkeypatch):
    root = Path(__file__).resolve().parents[1] / ".tmp-local-tools"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    try:
        monkeypatch.chdir(root)
        write_file("notes/plan.txt", "buy a phone\nbuy a case\n")
        assert "1|buy a phone" in read_file("notes/plan.txt")

        assert edit_file("notes/plan.txt", "buy a phone", "buy a laptop") == "Edited " + str(
            (root / "notes" / "plan.txt").resolve()
        )
        assert "buy a laptop" in read_file("notes/plan.txt")
        assert "old_string is not unique" in edit_file("notes/plan.txt", "buy a", "sell a")

        assert "plan.txt" in list_dir("notes")
        assert "plan.txt" in grep("laptop", "notes")
        listed = next(tool for tool in LOCAL_TOOLS if tool.name == "glob").executer(
            {"pattern": "**/*.txt", "path": str(root)}
        )
        assert "plan.txt" in listed
        assert "Exit code: 0" in shell('python -c "print(123)"', cwd=str(root))
    finally:
        shutil.rmtree(root, ignore_errors=True)
