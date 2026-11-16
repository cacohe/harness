import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LEGACY_TOP_LEVEL_MODULES = {
    "config",
    "context",
    "core",
    "demo",
    "llm",
    "logger",
    "mcp",
    "memory",
    "tool",
}


def test_project_imports_are_qualified_by_src():
    violations = []
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 0 or not node.module:
                continue
            if node.module.split(".", 1)[0] in LEGACY_TOP_LEVEL_MODULES:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.module}")
    assert violations == []


def _schema_core_violations(root: Path) -> list[str]:
    violations = []
    for path in (root / "src" / "schema").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module == "src":
                    modules.extend(f"src.{alias.name}" for alias in node.names)
                else:
                    modules.append(node.module)

            for module in modules:
                if module == "src.core" or module.startswith("src.core."):
                    violations.append(f"{path.relative_to(root)}:{node.lineno}:{module}")
    return violations


def test_schema_layer_does_not_depend_on_core_runtime():
    violations = _schema_core_violations(ROOT)
    assert violations == []


@pytest.mark.parametrize(
    "statement",
    [
        "import src.core.workspace",
        "import src.core.workspace as runtime",
        "from src import core",
        "from src import core as runtime",
    ],
)
def test_schema_dependency_guard_rejects_every_absolute_core_import(tmp_path: Path, statement: str):
    schema = tmp_path / "src" / "schema"
    schema.mkdir(parents=True)
    (schema / "probe.py").write_text(f"{statement}\n", encoding="utf-8")

    assert _schema_core_violations(tmp_path) != []


def test_uv_manages_a_non_packaged_project():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["uv"]["package"] is False
    assert "build-system" not in config
    assert "dev" in config["dependency-groups"]
    assert any(item.startswith("pydantic") for item in config["project"]["dependencies"])
