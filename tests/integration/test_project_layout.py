import ast
import tomllib
from pathlib import Path

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


def test_uv_manages_a_non_packaged_project():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["uv"]["package"] is False
    assert "build-system" not in config
    assert "dev" in config["dependency-groups"]
    assert any(item.startswith("pydantic") for item in config["project"]["dependencies"])
