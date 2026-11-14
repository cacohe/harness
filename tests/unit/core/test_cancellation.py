import asyncio
import threading

from src.core.cancellation import CancellationToken


def test_token_starts_active_and_cancel_is_idempotent():
    token = CancellationToken()

    assert token.is_cancelled() is False
    token.cancel()
    token.cancel()

    assert token.is_cancelled() is True
    assert token.wait(0) is True


def test_cancel_wakes_a_waiting_thread():
    token = CancellationToken()
    waiting = threading.Event()
    results = []

    def wait_for_cancel():
        waiting.set()
        results.append(token.wait(1))

    worker = threading.Thread(target=wait_for_cancel)
    worker.start()
    assert waiting.wait(1)

    token.cancel()
    worker.join(1)

    assert worker.is_alive() is False
    assert results == [True]


def test_reusing_a_cancelled_token_remains_cancelled():
    token = CancellationToken()
    token.cancel()

    assert token.wait(0) is True
    assert token.is_cancelled() is True


def test_cancel_wakes_async_waiters():
    async def scenario():
        token = CancellationToken()
        waiting = asyncio.create_task(token.await_cancelled(1))
        await asyncio.sleep(0)

        token.cancel()

        assert await waiting is True

    asyncio.run(scenario())


def test_async_wait_returns_false_after_timeout():
    async def scenario():
        token = CancellationToken()
        assert await token.await_cancelled(0.01) is False

    asyncio.run(scenario())


def test_async_wait_on_an_already_cancelled_token_returns_immediately():
    async def scenario():
        token = CancellationToken()
        token.cancel()
        assert await token.await_cancelled() is True

    asyncio.run(scenario())
