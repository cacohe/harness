from __future__ import annotations

import asyncio
from threading import Event, Lock


class CancellationToken:
    """A thread-safe, cooperative cancellation signal for one harness run."""

    def __init__(self) -> None:
        self._event = Event()
        self._waiters_lock = Lock()
        self._async_waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future[bool]]] = []

    def cancel(self) -> None:
        self._event.set()
        with self._waiters_lock:
            waiters = self._async_waiters
            self._async_waiters = []
        for loop, waiter in waiters:
            loop.call_soon_threadsafe(self._complete_waiter, waiter)

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)

    async def await_cancelled(self, timeout: float | None = None) -> bool:
        """Wait without blocking the event loop until cancelled or the timeout expires."""
        if self._event.is_set():
            return True

        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        entry = (loop, waiter)
        with self._waiters_lock:
            if self._event.is_set():
                return True
            self._async_waiters.append(entry)
        try:
            if timeout is None:
                return await waiter
            return await asyncio.wait_for(waiter, timeout)
        except TimeoutError:
            return False
        finally:
            with self._waiters_lock:
                if entry in self._async_waiters:
                    self._async_waiters.remove(entry)

    @staticmethod
    def _complete_waiter(waiter: asyncio.Future[bool]) -> None:
        if not waiter.done():
            waiter.set_result(True)
