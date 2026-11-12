from src.core import event as core_event
from src.core.context import ContextBuilder
from src.core.guard import HarnessGuard
from src.core.llm import LlmGateway
from src.core.memory import MemoryRetriever, MemoryStore


class DuckContextBuilder:
    def build_planning_context(self, system_prompt, task, **kwargs): ...
    def build_infer_context(self, system_prompt, task, tools, skills, **kwargs): ...
    def build_final_context(self, system_prompt, task, **kwargs): ...


class DuckLlmGateway:
    def planning(self, context, stream=True, response_format=None): ...
    def infer(self, context, stream=True, response_format=None): ...


class DuckGuard:
    def guard_input(self, task): ...


class DuckMemoryStore:
    def add(self, item): ...
    def update(self, item): ...
    def remove(self, item_id): ...
    def get(self, item_id): ...
    def all(self): ...


class DuckMemoryRetriever:
    def retrieve(self, query, items, top_k): ...


class DuckEventSink:
    def emit(self, event, channel=None): ...


def test_core_ports_accept_structural_implementations():
    assert isinstance(DuckContextBuilder(), ContextBuilder)
    assert isinstance(DuckLlmGateway(), LlmGateway)
    assert isinstance(DuckGuard(), HarnessGuard)
    assert isinstance(DuckMemoryStore(), MemoryStore)
    assert isinstance(DuckMemoryRetriever(), MemoryRetriever)


def test_event_sink_accepts_structural_implementation():
    assert hasattr(core_event, "EventSink")
    assert isinstance(DuckEventSink(), core_event.EventSink)


def test_guard_module_provides_an_explicit_noop_default():
    from src.core import guard

    assert hasattr(guard, "NoOpGuard")
    guard.NoOpGuard().guard_input(object())
