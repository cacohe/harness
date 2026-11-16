from src.application.agent import Agent
from src.core import event as core_event
from src.core.context import ContextBuilder
from src.core.guard import HarnessGuard
from src.core.llm import AsyncLlmGateway, LlmGateway
from src.core.loop import Harness
from src.core.memory import MemoryRetriever, MemoryStore
from src.core.tool import AsyncToolRunner, ToolExecutor
from src.tool.tool_executor import DefaultToolExecutor


class DuckContextBuilder:
    def build_infer_context(self, system_prompt, task, tools, skills, **kwargs): ...
    def build_final_context(self, system_prompt, task, **kwargs): ...


class DuckLlmGateway:
    def infer(self, context, stream=True, response_format=None): ...


class DuckAsyncLlmGateway:
    async def ainfer(self, context, stream=True, response_format=None): ...


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


class DuckAsyncToolRunner:
    async def arun(self, tool, inputs): ...


def test_core_ports_accept_structural_implementations():
    assert isinstance(DuckContextBuilder(), ContextBuilder)
    assert isinstance(DuckLlmGateway(), LlmGateway)
    assert isinstance(DuckAsyncLlmGateway(), AsyncLlmGateway)
    assert isinstance(DuckGuard(), HarnessGuard)
    assert isinstance(DuckMemoryStore(), MemoryStore)
    assert isinstance(DuckMemoryRetriever(), MemoryRetriever)
    assert isinstance(DuckAsyncToolRunner(), AsyncToolRunner)


def test_event_sink_accepts_structural_implementation():
    assert hasattr(core_event, "EventSink")
    assert isinstance(DuckEventSink(), core_event.EventSink)


def test_guard_module_provides_an_explicit_noop_default():
    from src.core import guard

    assert hasattr(guard, "NoOpGuard")
    guard.NoOpGuard().guard_input(object())


def test_public_composition_annotations_accept_native_async_ports():
    from types import UnionType
    from typing import get_args, get_type_hints

    harness_gateway = get_type_hints(Harness.__init__)["llm_gateway"]
    agent_gateway = get_type_hints(Agent.__init__)["llm_gateway"]
    core_runner = get_type_hints(ToolExecutor.__init__)["runner"]
    default_runner = get_type_hints(DefaultToolExecutor.__init__)["runner"]

    assert isinstance(harness_gateway, UnionType)
    assert AsyncLlmGateway in get_args(harness_gateway)
    assert AsyncLlmGateway in get_args(agent_gateway)
    assert AsyncToolRunner in get_args(core_runner)
    assert AsyncToolRunner in get_args(default_runner)
