"""Small adapters shared by async-first runtime components."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, TypeVar, cast

T = TypeVar("T")


async def _await_value(awaitable: Awaitable[T]) -> T:
    return await awaitable


def run_sync(awaitable_factory: Callable[[], Awaitable[T]], *, async_name: str) -> T:
    """Run one async entry point when the caller does not already own an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_value(awaitable_factory()))
    raise RuntimeError(f"an event loop is already running; await {async_name} instead")


async def call_maybe_async(
    async_callable: Callable[..., Any] | None,
    sync_callable: Callable[..., Any] | None,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Prefer a native async callable, otherwise offload its synchronous counterpart."""
    if async_callable is not None:
        result = async_callable(*args, **kwargs)
    else:
        if sync_callable is None:
            raise TypeError("neither an async nor synchronous callable is available")
        result = await asyncio.to_thread(sync_callable, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _next_or_end(iterator: Iterator[Any]) -> tuple[bool, Any]:
    try:
        return False, next(iterator)
    except StopIteration:
        return True, None


async def iterate_maybe_async(value: AsyncIterator[T] | Iterator[T]) -> AsyncIterator[T]:
    """Yield an async iterator directly or advance a sync iterator on worker threads."""
    if isinstance(value, AsyncIterator):
        async for item in value:
            yield item
        return

    iterator = iter(value)
    while True:
        done, item = await asyncio.to_thread(lambda: _next_or_end(iterator))
        if done:
            return
        yield cast(T, item)
