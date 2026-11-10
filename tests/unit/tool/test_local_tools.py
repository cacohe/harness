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
    root = Path(__file__).resolve().parents[3] / ".tmp-local-tools"
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


def test_shell_decodes_utf8_and_gbk_output(monkeypatch):
    class Completed:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = b""

    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return Completed(captured["payload"])

    monkeypatch.setattr("src.tool.local_tools.subprocess.run", fake_run)

    captured["payload"] = "目录".encode("utf-8")
    assert "目录" in shell("dir")
    assert "text" not in captured

    captured["payload"] = "中".encode("gbk")
    assert "中" in shell("dir")

    captured["payload"] = b"ok\xaf"
    rendered = shell("dir")
    assert "Exit code: 0" in rendered
    assert "ok" in rendered
