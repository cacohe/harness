from typing import Any


class CheckpointStore: ...


class ArtifactStore: ...


class TraceStore: ...


class HarnessPersistence:
    """
    Harness persistence interface.
    """

    def __init__(self):
        self.checkpoint_store = CheckpointStore()
        self.artifact_store = ArtifactStore()
        self.trace_store = TraceStore()

    def save(self, key: str, data: Any, *, kind: str, metadata: dict | None = None): ...
    def load(self, key: str, *, kind: str) -> Any: ...
    def exists(self, key: str, *, kind: str) -> bool: ...
    def delete(self, key: str, *, kind: str) -> None: ...
