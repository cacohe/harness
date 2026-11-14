from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from src.core.async_utils import run_sync
from src.core.cancellation import CancellationToken
from src.core.persistence import TaskRepository
from src.core.task import Task
from src.schema.task import TaskResult
from src.schema.tool import ToolExecutionResult


class HarnessRunner(Protocol):
    async def arun(
        self, task: Task, cancellation_token: CancellationToken | None = None
    ) -> TaskResult: ...


@dataclass
class TaskRun:
    """One session-owned turn that may pause and later resume."""

    task: Task
    message: str


class AgentSession:
    """Own one ordered conversation on top of shared Agent dependencies."""

    def __init__(
        self,
        harness: HarnessRunner,
        task_repository: TaskRepository,
        ensure_open: Callable[[], None],
    ) -> None:
        self.id = str(uuid.uuid4())
        self.harness = harness
        self.task_repository = task_repository
        self._ensure_open = ensure_open
        self.transcript: list[tuple[str, str]] = []
        self.pending_run: TaskRun | None = None
        self._operation_lock = threading.Lock()

    @property
    def pending_task(self) -> Task | None:
        return self.pending_run.task if self.pending_run is not None else None

    @property
    def pending_message(self) -> str | None:
        return self.pending_run.message if self.pending_run is not None else None

    def chat(
        self,
        message: str,
        cancellation_token: CancellationToken | None = None,
    ) -> Task:
        return run_sync(
            lambda: self.achat(message, cancellation_token=cancellation_token),
            async_name="achat",
        )

    async def achat(
        self,
        message: str,
        cancellation_token: CancellationToken | None = None,
    ) -> Task:
        self._begin_operation()
        try:
            if self.pending_run is not None:
                raise ValueError("session has a task waiting for approval")
            task = Task(original_input=self._with_transcript(message))
            await self.harness.arun(task, cancellation_token=cancellation_token)
            if task.is_paused():
                self.pending_run = TaskRun(task=task, message=message)
            else:
                self._remember(message, task)
            return task
        finally:
            self._operation_lock.release()

    def approval_requests(self) -> list[ToolExecutionResult]:
        task = self.pending_task
        return task.approval_requests() if task is not None else []

    def decide_tool_call(self, tool_id: str, approved: bool) -> None:
        self._begin_operation()
        try:
            task = self.pending_task
            if task is None:
                raise ValueError("nothing is waiting for approval")
            task.decide_tool_call(tool_id, approved)
            self.task_repository.save(task)
        finally:
            self._operation_lock.release()

    def resume(self, cancellation_token: CancellationToken | None = None) -> Task:
        return run_sync(
            lambda: self.aresume(cancellation_token=cancellation_token),
            async_name="aresume",
        )

    async def aresume(self, cancellation_token: CancellationToken | None = None) -> Task:
        self._begin_operation()
        run = self.pending_run
        try:
            if run is None:
                raise ValueError("nothing is waiting for approval")
            task = run.task
            await self.harness.arun(task, cancellation_token=cancellation_token)
            if not task.is_paused():
                self._remember(run.message, task)
            return task
        finally:
            if run is not None and not run.task.is_paused():
                self.pending_run = None
            self._operation_lock.release()

    def _begin_operation(self) -> None:
        self._ensure_open()
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("session operation already in progress")

    def _remember(self, message: str, task: Task) -> None:
        if task.is_completed() and task.result is not None:
            self.transcript.append(("user", message))
            self.transcript.append(("assistant", task.result.output))

    def _with_transcript(self, message: str) -> str:
        if not self.transcript:
            return message
        prior = "\n".join(f"{role}: {content}" for role, content in self.transcript)
        return f"{prior}\nuser: {message}"
