import ast
import importlib.util
import inspect
import tomllib
from pathlib import Path

import pytest
from src.application.agent import AgentSession
from src.application.composition import create_development_agent, create_production_agent
from src.core.journal import RunJournal

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
    "_run_driver.py",
    "approval.py",
    "artifact_validation.py",
    "artifacts.py",
    "commands.py",
    "context.py",
    "context_support.py",
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
LEGACY_DOCUMENT_TERMS = (
    "Session" + "Event",
    "AgentSession." + "chat",
    "AgentSession." + "achat",
    "Agent." + "aclose",
    "Snapshot" + "ExecutionStore",
    "Task" + "Repository",
    "SQLite schema " + "v2",
    "Migration from " + "v1",
)
ALLOWED_CORE_IMPORT_CYCLES = {
    ("src.core.state", "src.core.state_validation"),
}


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


def _present_legacy_paths(root: Path) -> list[str]:
    return [name for name in LEGACY_SOURCE_PATHS if (root / "src" / name).exists()]


def _core_line_counts(root: Path) -> dict[str, int]:
    return {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in (root / "src" / "core").glob("*.py")
    }


def _core_runtime_dependencies(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dependencies: set[str] = set()

    class RuntimeImportVisitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            is_type_checking = isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
            if is_type_checking:
                for child in node.orelse:
                    self.visit(child)
                return
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            dependencies.update(
                alias.name for alias in node.names if alias.name.startswith("src.core.")
            )

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.level == 0 and node.module and node.module.startswith("src.core."):
                dependencies.add(node.module)

    RuntimeImportVisitor().visit(tree)
    return dependencies


def _core_import_cycles(root: Path) -> list[tuple[str, ...]]:
    graph = {
        f"src.core.{path.stem}": _core_runtime_dependencies(path)
        for path in (root / "src" / "core").glob("*.py")
    }
    cycles: set[tuple[str, ...]] = set()

    def visit(module: str, path: tuple[str, ...]) -> None:
        if module in path:
            start = path.index(module)
            cycle = path[start:] + (module,)
            rotations = [cycle[index:-1] + cycle[:index] for index in range(len(cycle) - 1)]
            cycles.add(min(rotations))
            return
        for dependency in graph.get(module, set()):
            if dependency in graph:
                visit(dependency, path + (module,))

    for module in graph:
        visit(module, ())
    return sorted(cycles)


@pytest.mark.architecture
def test_core_contains_only_harness_mechanism_and_extension_ports():
    assert {path.name for path in (ROOT / "src" / "core").glob("*.py")} == EXPECTED_CORE_MODULES
    assert _core_violations(ROOT) == []


@pytest.mark.architecture
def test_core_modules_stay_within_reviewable_size_limits():
    line_counts = _core_line_counts(ROOT)
    assert line_counts["run.py"] <= 350
    assert {name: count for name, count in line_counts.items() if count > 500} == {}


@pytest.mark.architecture
def test_core_runtime_import_graph_is_acyclic():
    cycles = set(_core_import_cycles(ROOT))
    assert cycles - ALLOWED_CORE_IMPORT_CYCLES == set()


@pytest.mark.architecture
def test_legacy_architecture_paths_are_physically_absent():
    assert _present_legacy_paths(ROOT) == []
    searchable = [
        ROOT / "src",
        ROOT / "tests",
        ROOT / "README.md",
        ROOT / "docs" / "reliable-runtime.md",
    ]
    violations = []
    for target in searchable:
        paths = target.rglob("*.py") if target.is_dir() else (target,)
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for symbol in LEGACY_SYMBOLS:
                if symbol in text:
                    violations.append(f"{path.relative_to(ROOT)}:{symbol}")
    assert violations == []


def test_legacy_path_gate_rejects_placeholder_namespaces(tmp_path: Path):
    sandbox = tmp_path / "src" / "sandbox"
    sandbox.mkdir(parents=True)
    (sandbox / ".gitkeep").write_text("", encoding="utf-8")

    assert _present_legacy_paths(tmp_path) == ["sandbox"]


def test_reliable_runtime_documents_only_the_async_single_track_api():
    text = (ROOT / "docs" / "reliable-runtime.md").read_text(encoding="utf-8")
    assert [term for term in LEGACY_DOCUMENT_TERMS if term in text] == []
    assert inspect.iscoroutinefunction(create_development_agent)
    assert inspect.iscoroutinefunction(create_production_agent)
    for method in ("submit", "decide_tool_call", "cancel", "close"):
        assert inspect.iscoroutinefunction(getattr(AgentSession, method))
    for removed in ("chat", "achat", "aclose"):
        assert not hasattr(AgentSession, removed)
    assert list(inspect.signature(RunJournal.commit).parameters) == [
        "self",
        "batch",
        "expected_version",
    ]
    assert "RunJournal.commit(batch, expected_version)" in text


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
