import asyncio
import importlib
import importlib.util
import threading
import time

import pytest
from src.core.async_utils import call_maybe_async, iterate_maybe_async, run_sync


def test_async_iterator_uses_a_layer_neutral_canonical_module():
    spec = importlib.util.find_spec("src.utils.async_utils")

    assert spec is not None
    module = importlib.import_module("src.utils.async_utils")
    assert iterate_maybe_async is module.iterate_maybe_async


def test_run_sync_returns_async_result_outside_an_event_loop():
    assert run_sync(lambda: asyncio.sleep(0, result="done"), async_name="ado_work") == "done"


def test_run_sync_rejects_an_active_event_loop_before_creating_the_awaitable():
    created = False

    def factory():
        nonlocal created
        created = True
        return asyncio.sleep(0)

    async def scenario():
        with pytest.raises(RuntimeError, match="await ado_work"):
            run_sync(factory, async_name="ado_work")

    asyncio.run(scenario())

    assert created is False


def test_call_maybe_async_prefers_the_native_async_callable():
    sync_called = False

    async def native(value):
        return value * 2

    def legacy(value):
        nonlocal sync_called
        sync_called = True
        return value

    result = asyncio.run(call_maybe_async(native, legacy, 3))

    assert result == 6
    assert sync_called is False


def test_call_maybe_async_runs_a_legacy_callable_on_a_worker_thread():
    caller_thread = threading.get_ident()

    def legacy():
        return threading.get_ident()

    worker_thread = asyncio.run(call_maybe_async(None, legacy))

    assert worker_thread != caller_thread


def test_iterate_maybe_async_advances_a_sync_iterator_without_blocking_the_loop():
    ticks: list[str] = []

    def blocking_values():
        time.sleep(0.05)
        yield "value"

    async def ticker():
        await asyncio.sleep(0.01)
        ticks.append("tick")

    async def scenario():
        ticker_task = asyncio.create_task(ticker())
        values = [item async for item in iterate_maybe_async(blocking_values())]
        await ticker_task
        return values

    assert asyncio.run(scenario()) == ["value"]
    assert ticks == ["tick"]


def test_iterate_maybe_async_consumes_an_async_iterator_directly():
    async def values():
        yield "first"
        yield "second"

    async def scenario():
        return [item async for item in iterate_maybe_async(values())]

    assert asyncio.run(scenario()) == ["first", "second"]
