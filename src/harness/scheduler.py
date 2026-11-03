from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from harness.types import EnvHandle, ToolCall, ToolResult, ToolSchedule


class ToolScheduler:
    def execute(
        self,
        calls: Sequence[ToolCall],
        schedule: ToolSchedule,
        environment,
        handle: EnvHandle,
        on_complete: Callable[[ToolResult], None] | None = None,
    ) -> list[ToolResult]:
        if schedule.completion == "async":
            return self._start_async(calls, schedule, environment, handle, on_complete)
        if schedule.concurrency == "parallel" and len(calls) > 1:
            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                return list(pool.map(lambda call: environment.invoke(handle, call), calls))
        return [environment.invoke(handle, call) for call in calls]

    def _start_async(self, calls, schedule, environment, handle, on_complete) -> list[ToolResult]:
        def run_one(call: ToolCall) -> None:
            try:
                result = environment.invoke(handle, call)
            except Exception as exc:
                result = ToolResult(call_id=call.id, ok=False, output="", error=str(exc))
            if on_complete is not None:
                on_complete(result)

        if schedule.concurrency == "parallel":
            for call in calls:
                threading.Thread(target=run_one, args=(call,), daemon=True).start()
        else:
            def run_all() -> None:
                for call in calls:
                    run_one(call)

            threading.Thread(target=run_all, daemon=True).start()
        return [
            ToolResult(call_id=call.id, ok=True, output="accepted", accepted=True)
            for call in calls
        ]
