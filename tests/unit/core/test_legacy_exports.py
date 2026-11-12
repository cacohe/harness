import importlib

import pytest
from src.core import persistence, tool


def test_obsolete_type_aliases_are_not_public_api():
    assert not hasattr(tool, "ToolRegistry")
    assert not hasattr(persistence, "CheckpointStore")
    assert not hasattr(persistence, "ArtifactStore")
    assert not hasattr(persistence, "TraceStore")


@pytest.mark.parametrize("module_name", ["src.core.sandbox", "src.schema.hook"])
def test_obsolete_compatibility_modules_cannot_be_imported(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
