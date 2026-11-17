import ast
import importlib.util
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


FORBIDDEN_IMPORTS = {
    "schema": (
        "src.core",
        "src.persistence",
        "src.demo",
        "src.llm",
        "src.mcp",
        "src.sandbox",
        "fastmcp",
        "litellm",
        "docker",
    ),
    "core": (
        "src.persistence",
        "src.demo",
        "src.llm.litellm",
        "src.mcp",
        "src.sandbox",
        "fastmcp",
        "litellm",
        "docker",
    ),
}

# Every exception must explain why dependency inversion cannot represent it.
ARCHITECTURE_ALLOWLIST: dict[tuple[str, str], str] = {}


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
            imports.append((node.lineno, module))
            imports.extend(
                (node.lineno, f"{module}.{alias.name}") for alias in node.names if alias.name != "*"
            )
        elif (
            isinstance(node, ast.Call)
            and len(node.args) >= 1
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


def _architecture_violations(root: Path) -> list[str]:
    violations = []
    for layer, prefixes in FORBIDDEN_IMPORTS.items():
        for path in (root / "src" / layer).rglob("*.py"):
            for line, module in _resolved_imports(path, root):
                forbidden = next(
                    (
                        prefix
                        for prefix in prefixes
                        if module == prefix or module.startswith(f"{prefix}.")
                    ),
                    None,
                )
                if forbidden is None:
                    continue
                key = (str(path.relative_to(root)).replace("\\", "/"), module)
                if key in ARCHITECTURE_ALLOWLIST:
                    assert ARCHITECTURE_ALLOWLIST[key].strip()
                    continue
                violations.append(f"{key[0]}:{line}:{module}")
    return violations


@pytest.mark.architecture
def test_schema_and_core_dependency_direction_is_enforced():
    assert _architecture_violations(ROOT) == []


@pytest.mark.parametrize(
    "layer,statement",
    [
        ("schema", "import src.core.task"),
        ("schema", "from ..core import task"),
        ("schema", "importlib.import_module('src.persistence.sqlite')"),
        ("core", "import src.persistence.sqlite"),
        ("core", "from ..persistence import sqlite"),
        ("core", "from src.llm import litellm"),
        ("core", "import fastmcp"),
        ("core", "import docker"),
        ("core", "importlib.import_module('src.demo.agent')"),
    ],
)
@pytest.mark.architecture
def test_architecture_gate_rejects_concrete_dependency_bypasses(
    tmp_path: Path, layer: str, statement: str
):
    target = tmp_path / "src" / layer
    target.mkdir(parents=True)
    source = "import importlib\n" if "importlib." in statement else ""
    (target / "probe.py").write_text(f"{source}{statement}\n", encoding="utf-8")

    assert _architecture_violations(tmp_path)


@pytest.mark.parametrize(
    "layer,statement",
    [
        ("schema", "from pydantic import BaseModel"),
        ("core", "from src.schema.execution import SessionEvent"),
        ("core", "from typing import Protocol"),
        ("demo", "from src.persistence.sqlite import SqliteExecutionStore"),
    ],
)
@pytest.mark.architecture
def test_architecture_gate_allows_schema_values_and_core_ports(
    tmp_path: Path, layer: str, statement: str
):
    target = tmp_path / "src" / layer
    target.mkdir(parents=True)
    (target / "probe.py").write_text(f"{statement}\n", encoding="utf-8")

    assert _architecture_violations(tmp_path) == []


def test_schema_layer_does_not_depend_on_core_runtime():
    violations = _schema_core_violations(ROOT)
    assert violations == []


def test_internal_modules_use_canonical_neutral_imports():
    compatibility_modules = {
        ROOT / "src" / "core" / "async_utils.py",
        ROOT / "src" / "core" / "workspace.py",
    }
    violations = []
    for path in (ROOT / "src").rglob("*.py"):
        if path in compatibility_modules:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "src.core.workspace":
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module == "src.core.workspace":
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.module}")
                if node.module == "src.core.async_utils" and any(
                    alias.name == "iterate_maybe_async" for alias in node.names
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:iterate_maybe_async")
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
