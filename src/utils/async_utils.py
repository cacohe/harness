"""Layer-neutral adapters for consuming synchronous or asynchronous iterators."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any, TypeVar, cast

T = TypeVar("T")


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
