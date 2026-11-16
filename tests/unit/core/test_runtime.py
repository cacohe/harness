import asyncio

import pytest
from src.core.runtime import (
    RuntimeClosedError,
    RuntimeCloseTimeoutError,
    RuntimeCoordinator,
)


class Resource:
    def __init__(self, name, calls, *, fail_start=False, close_gate=None):
        self.name = name
        self.calls = calls
        self.fail_start = fail_start
        self.close_gate = close_gate

    async def astart(self):
        self.calls.append(("start", self.name))
        if self.fail_start:
            raise RuntimeError("start failed")

    async def aflush(self):
        self.calls.append(("flush", self.name))

    async def aclose(self):
        self.calls.append(("close", self.name))
        if self.close_gate is not None:
            await self.close_gate.wait()


def test_resources_start_once_and_flush_then_close_in_reverse():
    async def scenario():
        calls = []
        runtime = RuntimeCoordinator([Resource("one", calls), Resource("two", calls)])
        await asyncio.gather(runtime.start(), runtime.start())
        await runtime.aclose(1)
        await runtime.aclose(1)
        return runtime, calls

    runtime, calls = asyncio.run(scenario())

    assert calls == [
        ("start", "one"),
        ("start", "two"),
        ("flush", "two"),
        ("flush", "one"),
        ("close", "two"),
        ("close", "one"),
    ]
    assert runtime.state == "closed"


def test_partial_startup_closes_only_started_resources_in_reverse():
    async def scenario():
        calls = []
        runtime = RuntimeCoordinator(
            [Resource("one", calls), Resource("broken", calls, fail_start=True)]
        )
        with pytest.raises(RuntimeError, match="start failed"):
            await runtime.start()
        return runtime, calls

    runtime, calls = asyncio.run(scenario())

    assert calls == [("start", "one"), ("start", "broken"), ("close", "one")]
    assert runtime.state == "closed"


def test_close_cancels_operations_and_rejects_new_ones():
    async def scenario():
        runtime = RuntimeCoordinator()
        await runtime.start()
        started = asyncio.Event()

        async def operation():
            started.set()
            await asyncio.Event().wait()

        task = runtime.register_operation(asyncio.create_task(operation()))
        await started.wait()
        await runtime.aclose(1)
        rejected = asyncio.create_task(asyncio.sleep(0))
        with pytest.raises(RuntimeClosedError):
            runtime.register_operation(rejected)
        rejected.cancel()
        await asyncio.gather(rejected, return_exceptions=True)
        return task

    task = asyncio.run(scenario())
    assert task.cancelled()


def test_timeout_and_cancelled_waiter_do_not_cancel_shielded_cleanup():
    async def scenario():
        calls = []
        gate = asyncio.Event()
        runtime = RuntimeCoordinator([Resource("slow", calls, close_gate=gate)])
        await runtime.start()
        with pytest.raises(RuntimeCloseTimeoutError):
            await runtime.aclose(0.01)
        assert runtime.state == "closing"
        waiter = asyncio.create_task(runtime.aclose(1))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        gate.set()
        await runtime.aclose(1)
        return runtime, calls

    runtime, calls = asyncio.run(scenario())
    assert runtime.state == "closed"
    assert calls.count(("close", "slow")) == 1
