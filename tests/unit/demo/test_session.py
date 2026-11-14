import asyncio

import pytest
from src.demo.session import AgentSession
from src.schema.task import (
    TaskFailReason,
    TaskPauseReason,
    TaskResult,
    TaskRunningReason,
)
from src.schema.tool import ToolExecutionResult


class RecordingRepository:
    def __init__(self):
        self.saved = []

    def save(self, task):
        self.saved.append(task.to_dict())

    def get(self, task_id):
        return None


class CompletingHarness:
    def __init__(self):
        self.inputs = []

    async def arun(self, task, cancellation_token=None):
        self.inputs.append(task.original_input)
        task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
        task.set_result(TaskResult(output=f"answer-{len(self.inputs)}"))
        task.transition_to_completed()
        return task.result


class PausingHarness:
    def __init__(self):
        self.calls = 0

    async def arun(self, task, cancellation_token=None):
        self.calls += 1
        if self.calls == 1:
            task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
            task.set_pending_tool_calls(
                [
                    ToolExecutionResult(
                        tool_id="call-1",
                        tool_name="shell",
                        needs_approval=True,
                    )
                ]
            )
            task.transition_to_paused(TaskPauseReason.TASK_PAUSED_WAITING_APPROVAL)
        else:
            task.transition_to_running(TaskRunningReason.TASK_RUNNING_RESUME)
            task.set_pending_tool_calls([])
            task.set_result(TaskResult(output="resumed"))
            task.transition_to_completed()
        return task.result


def test_session_completed_turns_use_only_their_own_transcript():
    harness = CompletingHarness()
    session = AgentSession(harness, RecordingRepository(), lambda: None)

    first = session.chat("one")
    second = session.chat("two")

    assert first.result.output == "answer-1"
    assert second.result.output == "answer-2"
    assert harness.inputs == ["one", "user: one\nassistant: answer-1\nuser: two"]
    assert session.transcript == [
        ("user", "one"),
        ("assistant", "answer-1"),
        ("user", "two"),
        ("assistant", "answer-2"),
    ]
    assert session.pending_run is None


def test_session_owns_paused_run_approval_and_resume():
    repository = RecordingRepository()
    session = AgentSession(PausingHarness(), repository, lambda: None)

    paused = session.chat("run it")

    assert session.pending_task is paused
    assert session.pending_message == "run it"
    assert [item.tool_id for item in session.approval_requests()] == ["call-1"]
    with pytest.raises(ValueError, match="waiting for approval"):
        session.chat("overwrite")

    session.decide_tool_call("call-1", approved=False)
    completed = session.resume()

    assert repository.saved[-1]["id"] == paused.id
    assert completed is paused
    assert completed.result.output == "resumed"
    assert session.pending_run is None
    assert session.transcript == [("user", "run it"), ("assistant", "resumed")]


def test_session_rejects_decision_without_own_pending_run():
    session = AgentSession(CompletingHarness(), RecordingRepository(), lambda: None)

    assert session.approval_requests() == []
    with pytest.raises(ValueError, match="nothing is waiting"):
        session.decide_tool_call("call-1", approved=True)


def test_failed_turn_is_not_committed_to_session_transcript():
    class FailingHarness:
        async def arun(self, task, cancellation_token=None):
            task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
            task.set_result(TaskResult(output="failed"))
            task.transition_to_failed(TaskFailReason.TASK_FAILED_CANCEL)
            return task.result

    session = AgentSession(FailingHarness(), RecordingRepository(), lambda: None)

    task = asyncio.run(session.achat("cancelled"))

    assert task.is_failed()
    assert session.transcript == []
    assert session.pending_run is None


def test_same_session_reentry_fails_without_blocking_the_first_turn():
    class BlockingHarness:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def arun(self, task, cancellation_token=None):
            self.started.set()
            await self.release.wait()
            task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
            task.set_result(TaskResult(output="first"))
            task.transition_to_completed()
            return task.result

    async def scenario():
        harness = BlockingHarness()
        session = AgentSession(harness, RecordingRepository(), lambda: None)
        first = asyncio.create_task(session.achat("one"))
        await harness.started.wait()

        with pytest.raises(RuntimeError, match="already in progress"):
            await session.achat("two")

        harness.release.set()
        return session, await first

    session, task = asyncio.run(scenario())

    assert task.result.output == "first"
    assert session.transcript == [("user", "one"), ("assistant", "first")]


def test_operation_guard_is_released_after_error():
    class RaisesOnceHarness(CompletingHarness):
        async def arun(self, task, cancellation_token=None):
            if not self.inputs:
                self.inputs.append(task.original_input)
                raise RuntimeError("gateway offline")
            return await super().arun(task, cancellation_token=cancellation_token)

    session = AgentSession(RaisesOnceHarness(), RecordingRepository(), lambda: None)

    with pytest.raises(RuntimeError, match="gateway offline"):
        session.chat("first")
    recovered = session.chat("second")

    assert recovered.is_completed()


def test_operation_guard_is_released_after_caller_cancellation():
    class CancellableHarness(CompletingHarness):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def arun(self, task, cancellation_token=None):
            if not self.inputs:
                self.inputs.append(task.original_input)
                self.started.set()
                await self.release.wait()
            return await super().arun(task, cancellation_token=cancellation_token)

    async def scenario():
        harness = CancellableHarness()
        session = AgentSession(harness, RecordingRepository(), lambda: None)
        cancelled = asyncio.create_task(session.achat("cancel"))
        await harness.started.wait()
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        harness.release.set()
        recovered = await session.achat("recover")
        return session, recovered

    session, recovered = asyncio.run(scenario())

    assert recovered.is_completed()
    assert session.transcript == [("user", "recover"), ("assistant", "answer-2")]
