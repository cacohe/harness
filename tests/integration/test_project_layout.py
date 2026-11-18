import ast
import importlib.util
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LEGACY_SOURCE_PATHS = (
    "context",
    "demo",
    "llm",
    "mcp",
    "memory",
    "observability",
    "persistence",
    "sandbox",
    "schema",
    "skill",
    "tool",
    "utils",
)
EXPECTED_CORE_MODULES = {
    "__init__.py",
    "approval.py",
    "artifacts.py",
    "commands.py",
    "context.py",
    "errors.py",
    "events.py",
    "ids.py",
    "journal.py",
    "json_values.py",
    "model.py",
    "receipt_snapshot.py",
    "run.py",
    "run_content.py",
    "run_projection.py",
    "run_receipt.py",
    "state.py",
    "state_validation.py",
    "tools.py",
}
FORBIDDEN_CORE_IMPORTS = (
    "src.adapters",
    "src.application",
    "src" + ".config",
    "src" + ".logger",
    "src" + ".context",
    "src" + ".demo",
    "src" + ".llm",
    "src" + ".mcp",
    "src" + ".memory",
    "src" + ".observability",
    "src" + ".persistence",
    "src" + ".sandbox",
    "src" + ".schema",
    "src" + ".skill",
    "src" + ".tool",
    "fastmcp",
    "litellm",
    "sqlite3",
    "docker",
    "pydantic",
)
LEGACY_SYMBOLS = (
    "Snapshot" + "ExecutionStore",
    "Task" + "Repository",
    "src.core." + "loop",
)


def _resolved_imports(path: Path, root: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(root).with_suffix("")
    package = ".".join(relative.parts[:-1])
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                name = "." * node.level + (node.module or "")
                try:
                    module = importlib.util.resolve_name(name, package)
                except (ImportError, ValueError):
                    module = name
            else:
                module = node.module or ""
            imports.extend((node.lineno, f"{module}.{alias.name}") for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and (
                (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "importlib"
                    and node.func.attr == "import_module"
                )
                or (isinstance(node.func, ast.Name) and node.func.id == "import_module")
            )
        ):
            imports.append((node.lineno, node.args[0].value))
    return imports


def _core_violations(root: Path) -> list[str]:
    violations = []
    for path in (root / "src" / "core").rglob("*.py"):
        for line, module in _resolved_imports(path, root):
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in FORBIDDEN_CORE_IMPORTS
            ):
                violations.append(f"{path.relative_to(root)}:{line}:{module}")
    return violations


@pytest.mark.architecture
def test_core_contains_only_harness_mechanism_and_extension_ports():
    assert {path.name for path in (ROOT / "src" / "core").glob("*.py")} == EXPECTED_CORE_MODULES
    assert _core_violations(ROOT) == []


@pytest.mark.architecture
def test_legacy_architecture_has_no_source_files():
    assert [name for name in LEGACY_SOURCE_PATHS if any((ROOT / "src" / name).rglob("*.py"))] == []
    searchable = [ROOT / "src", ROOT / "tests", ROOT / "README.md"]
    violations = []
    for target in searchable:
        paths = target.rglob("*.py") if target.is_dir() else (target,)
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for symbol in LEGACY_SYMBOLS:
                if symbol in text:
                    violations.append(f"{path.relative_to(ROOT)}:{symbol}")
    assert violations == []


@pytest.mark.parametrize(
    "statement",
    [
        "import src.adapters.tools",
        "from src.application import runtime",
        "from src import config",
        "import src" + ".schema.execution",
        "import src" + ".persistence.sqlite",
        "import litellm",
        "import sqlite3",
        "importlib.import_module('src" + ".demo.agent')",
    ],
)
@pytest.mark.architecture
def test_core_gate_rejects_concrete_and_legacy_dependencies(tmp_path: Path, statement: str):
    core = tmp_path / "src" / "core"
    core.mkdir(parents=True)
    source = "import importlib\n" if "importlib." in statement else ""
    (core / "probe.py").write_text(f"{source}{statement}\n", encoding="utf-8")
    assert _core_violations(tmp_path)


def test_uv_manages_a_non_packaged_project():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["uv"]["package"] is False
    assert "build-system" not in config
    assert "dev" in config["dependency-groups"]
