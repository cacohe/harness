import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_importing_submissions_does_not_load_runtime_dependencies() -> None:
    program = """
import sys
from importlib.abc import MetaPathFinder


class RejectRuntimeImports(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        forbidden = (
            "src.application.agent", "src.adapters", "litellm", "sqlite3",
        )
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError(f"unexpected runtime import: {fullname}")


sys.meta_path.insert(0, RejectRuntimeImports())
from src.application.submissions import SubmissionService
"""
    completed = subprocess.run(
        [sys.executable, "-S", "-c", program],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
