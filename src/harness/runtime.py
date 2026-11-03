from __future__ import annotations

import threading
import time
import uuid
from dataclasses import replace

from harness.approval import grant_covers, normalize_scope
from harness.config import ResolvedConfig, resolve_config, resolve_output_mode
from harness.defaults import DefaultGoalAdjuster, EmptyMemoryStore, ListObserver, StructuralDiscriminator
from harness.environment import CurrentEnvironment, IsolatedEnvironment
from harness.hooks import HookBus, HookPoint
from harness.queue import InputQueue
from harness.scheduler import ToolScheduler
from harness.state import InMemoryStateStore, transition
from harness.types import (
    TERMINAL_STATUSES,
    ApprovalRequirement,
    BuildContextArgs,
    HarnessInput,
    MemoryQuery,
    Message,
    OutputPhase,
    RunResult,
    RunSnapshot,
    RunStatus,
    ToolCall,
    ToolResult,
    TraceEvent,
)


class Harness:
    """Runs the agent loop. Ports supplied by the host do the actual work."""

    def __init__(
        self,
        config,
        *,
        validator,
        context_builder,
        llm,
        approval_policy,
        approver,
        safety,
        executor,
        discriminator=None,
        goal_adjuster=None,
        state_store=None,
        observer=None,
        hooks=None,
        memory=None,
        environment=None,
        scheduler=None,
    ) -> None:
        self.config = config
        self.resolved: ResolvedConfig = resolve_config(config)
        self.validator = validator
        self.context_builder = context_builder
        self.llm = llm
        self.approval_policy = approval_policy
        self.approver = approver
        self.safety = safety
        self.executor = executor
        self.discriminator = discriminator or StructuralDiscriminator()
        self.goal_adjuster = goal_adjuster or DefaultGoalAdjuster()
        self.state_store = state_store or InMemoryStateStore()
        self.observer = observer or ListObserver()
        self.hooks = hooks or HookBus()
        self.memory = memory or EmptyMemoryStore()
        self.scheduler = scheduler or ToolScheduler()
        if environment is not None:
            self.environment = environment
        elif self.resolved.execution_kind == "isolated":
            self.environment = IsolatedEnvironment()
        else:
            self.environment = CurrentEnvironment(executor)
        self.queue: InputQueue[HarnessInput] = InputQueue()
        self._grants = []
        self._handle = None
        self._cancelled = False
        self._pending = 0
        self._async_ready: list[ToolResult] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def submit(self, content: str, role: str = "user") -> HarnessInput:
        item = HarnessInput(
            id=uuid.uuid4().hex,
            role=role,
            content=content,
            created_at=time.time(),
        )
        self.queue.push(item)
        return item

    def cancel(self) -> None:
        self._cancelled = True

    def run(self, content: str, *, goal: str | None = None) -> RunResult:
        self.submit(content)
        snapshot = RunSnapshot(
            run_id=uuid.uuid4().hex,
            status=RunStatus.IDLE,
            goal=content if goal is None else goal,
            strategy="",
            step=0,
            messages=[],
        )
        try:
            snapshot, output = self._loop(snapshot)
        finally:
            if self._handle is not None:
                self.environment.dispose(self._handle)
                self._handle = None
        stored = self.state_store.load(snapshot.run_id) or snapshot
        return RunResult(
            run_id=snapshot.run_id,
            status=snapshot.status,
            output=output,
            error=snapshot.error,
            snapshot=stored,
        )

    def _loop(self, snapshot: RunSnapshot) -> tuple[RunSnapshot, str | None]:
        snapshot = self._move(snapshot, RunStatus.QUEUED)
        output: str | None = None
        while snapshot.status not in TERMINAL_STATUSES:
            if self._cancelled:
                snapshot = self._move(snapshot, RunStatus.CANCELLED)
                break
            if snapshot.step >= self.resolved.max_steps:
                snapshot = self._fail(snapshot, "max steps exceeded")
                break

            incoming = self.queue.drain()
            ready = self._drain_async()
            update = self.goal_adjuster.adjust(snapshot, incoming)
            snapshot = replace(snapshot, goal=update.goal, strategy=update.strategy)
            for item in incoming:
                snapshot.messages.append(Message(role=item.role, content=item.content))
            for result in ready:
                snapshot.messages.append(_tool_message(result))
            self.hooks.emit(HookPoint.BEFORE_STEP, {"snapshot": snapshot, "inputs": incoming})

            snapshot = self._move(snapshot, RunStatus.VALIDATING)
            validation = self.validator.validate(incoming, snapshot)
            if not validation.ok:
                snapshot = self._fail(snapshot, validation.reason)
                break
            self.hooks.emit(HookPoint.AFTER_VALIDATE, {"snapshot": snapshot, "inputs": incoming})
            if self._cancelled:
                snapshot = self._move(snapshot, RunStatus.CANCELLED)
                break

            snapshot = self._move(snapshot, RunStatus.BUILDING_CONTEXT)
            query_text = incoming[-1].content if incoming else snapshot.goal
            memories = tuple(self.memory.search(MemoryQuery(goal=snapshot.goal, text=query_text)))
            context = self.context_builder.build(
                BuildContextArgs(
                    state=snapshot,
                    system_prompt=self.resolved.system_prompt,
                    memories=memories,
                    tools=self.resolved.tools,
                )
            )
            mode = resolve_output_mode(self.resolved, OutputPhase.EXECUTION)
            hooked = self.hooks.emit(
                HookPoint.BEFORE_INFER,
                {"context": context, "mode": mode, "snapshot": snapshot},
            )
            context = hooked.get("context", context)
            mode = hooked.get("mode", mode)

            snapshot = self._move(snapshot, RunStatus.INFERRING)
            llm_output = self.llm.complete(context, mode)
            self._emit("model", snapshot, {"text": llm_output.text, "tool_calls": [_call_payload(call) for call in llm_output.tool_calls]})
            self.hooks.emit(HookPoint.AFTER_INFER, {"snapshot": snapshot, "output": llm_output})

            snapshot = self._move(snapshot, RunStatus.DECIDING)
            decision = self.discriminator.decide(llm_output, snapshot)
            if decision.type == "fail":
                snapshot = self._fail(snapshot, decision.reason)
                break
            if decision.type == "finish":
                self._wait_pending()
                late = self._drain_async()
                if late:
                    for result in late:
                        snapshot.messages.append(_tool_message(result))
                    snapshot = replace(snapshot, step=snapshot.step + 1)
                    snapshot = self._move(snapshot, RunStatus.QUEUED)
                    continue
                final_mode = resolve_output_mode(self.resolved, OutputPhase.FINAL)
                event_mode = resolve_output_mode(self.resolved, OutputPhase.EVENT)
                emitted = self.hooks.emit(
                    HookPoint.BEFORE_OUTPUT,
                    {"text": decision.text, "mode": final_mode, "event_mode": event_mode},
                )
                text = emitted.get("text", decision.text)
                snapshot = self._move(snapshot, RunStatus.EMITTING)
                self._emit(
                    "output",
                    snapshot,
                    {"text": text, "mode": final_mode.value, "event_mode": event_mode.value},
                )
                snapshot = replace(snapshot, output=text, step=snapshot.step + 1)
                snapshot = self._move(snapshot, RunStatus.FINISHED)
                output = text
                break

            snapshot, results, halted = self._authorize_and_run(snapshot, decision.calls)
            if halted:
                break
            snapshot.messages.append(
                Message(role="assistant", content=llm_output.text or "", tool_calls=tuple(decision.calls))
            )
            for result in results:
                snapshot.messages.append(_tool_message(result))
            snapshot = replace(snapshot, step=snapshot.step + 1, last_tool_results=list(results))
            snapshot = self._move(snapshot, RunStatus.QUEUED)
        return snapshot, output

    def _authorize_and_run(self, snapshot: RunSnapshot, calls: tuple[ToolCall, ...]):
        results: list[ToolResult | None] = [None] * len(calls)
        to_run: list[tuple[int, ToolCall]] = []
        for index, call in enumerate(calls):
            decided = self.approval_policy.decide(call, self.resolved.permission_policy, snapshot)
            self.hooks.emit(HookPoint.ON_APPROVAL, {"call": call, "decision": decided, "snapshot": snapshot})
            self._emit(
                "approval",
                snapshot,
                {"call_id": call.id, "requirement": decided.requirement.value, "scope": decided.scope.kind},
            )
            if decided.requirement is ApprovalRequirement.FORBIDDEN:
                results[index] = ToolResult(call_id=call.id, ok=False, output="forbidden", error="forbidden")
                continue
            needs_review = decided.requirement in (
                ApprovalRequirement.NEEDS_APPROVAL,
                ApprovalRequirement.NEEDS_HUMAN_APPROVAL,
            )
            if needs_review and not self._covered(call):
                snapshot = self._move(snapshot, RunStatus.AWAITING_APPROVAL)
                verdict = self.approver.review(decided.action, decided.requirement)
                self._emit("approval_verdict", snapshot, {"call_id": call.id, "verdict": verdict})
                if verdict != "approved":
                    results[index] = ToolResult(call_id=call.id, ok=False, output="denied", error="denied")
                    continue
                scope = normalize_scope(decided.scope, call)
                if scope.kind in ("session", "exact"):
                    self._grants.append(scope)
            snapshot = self._move(snapshot, RunStatus.CHECKING_SAFETY)
            safety = self.safety.check(decided.action, self.environment)
            self._emit("safety", snapshot, {"call_id": call.id, "verdict": safety})
            if safety == "deny":
                results[index] = ToolResult(call_id=call.id, ok=False, output="safety denied", error="safety denied")
                continue
            to_run.append((index, call))

        if to_run:
            snapshot = self._move(snapshot, RunStatus.EXECUTING_TOOLS)
            ordered = [call for _, call in to_run]
            self.hooks.emit(HookPoint.BEFORE_TOOL, {"calls": ordered, "snapshot": snapshot})
            try:
                handle = self._ensure_handle()
            except NotImplementedError as exc:
                snapshot = self._fail(snapshot, str(exc))
                return snapshot, [], True
            asynchronous = self.resolved.tool_schedule.completion == "async"
            if asynchronous:
                with self._lock:
                    self._pending += len(ordered)
            executed = self.scheduler.execute(
                ordered,
                self.resolved.tool_schedule,
                self.environment,
                handle,
                on_complete=self._on_tool_complete if asynchronous else None,
            )
            self.hooks.emit(HookPoint.AFTER_TOOL, {"results": executed, "snapshot": snapshot})
            self._emit("tool", snapshot, {"call_ids": [call.id for call in ordered]})
            for (index, _call), result in zip(to_run, executed):
                results[index] = result
        return snapshot, [result for result in results if result is not None], False

    def _covered(self, call: ToolCall) -> bool:
        return any(grant_covers(grant, call) for grant in self._grants)

    def _ensure_handle(self):
        if self._handle is None:
            self._handle = self.environment.prepare()
        return self._handle

    def _on_tool_complete(self, result: ToolResult) -> None:
        with self._cv:
            self._async_ready.append(result)
            self._pending -= 1
            self._cv.notify_all()

    def _wait_pending(self) -> None:
        with self._cv:
            while self._pending > 0:
                if not self._cv.wait(timeout=5):
                    break

    def _drain_async(self) -> list[ToolResult]:
        with self._cv:
            ready = list(self._async_ready)
            self._async_ready.clear()
            return ready

    def _move(self, snapshot: RunSnapshot, status: RunStatus) -> RunSnapshot:
        if snapshot.status is status:
            return snapshot
        moved = transition(snapshot, status)
        self.state_store.save(moved)
        self.hooks.emit(HookPoint.ON_STATE_CHANGE, {"snapshot": moved, "status": status})
        self._emit("state", moved, {"status": status.value})
        return replace(moved, messages=list(moved.messages), last_tool_results=list(moved.last_tool_results))

    def _fail(self, snapshot: RunSnapshot, reason: str) -> RunSnapshot:
        snapshot = replace(snapshot, error=reason)
        return self._move(snapshot, RunStatus.FAILED)

    def _emit(self, kind: str, snapshot: RunSnapshot, payload: dict) -> None:
        self.observer.emit(
            TraceEvent(
                run_id=snapshot.run_id,
                step=snapshot.step,
                status=snapshot.status,
                kind=kind,
                payload=payload,
            )
        )


def _tool_message(result: ToolResult) -> Message:
    return Message(role="tool", content=result.output, tool_call_id=result.call_id)


def _call_payload(call: ToolCall) -> dict:
    return {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
