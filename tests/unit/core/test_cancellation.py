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
