import pytest

from harness import (
    HookBus,
    HookPoint,
    IllegalTransition,
    InMemoryStateStore,
    InputQueue,
    RunSnapshot,
    RunStatus,
    transition,
)

from support import config


def _snapshot(status):
    return RunSnapshot(
        run_id="run",
        status=status,
        goal="goal",
        strategy="",
        step=0,
        messages=[],
    )


def test_state_machine_allows_the_main_step_and_rejects_skips():
    moved = transition(_snapshot(RunStatus.VALIDATING), RunStatus.BUILDING_CONTEXT)

    assert moved.status is RunStatus.BUILDING_CONTEXT
    with pytest.raises(IllegalTransition):
        transition(_snapshot(RunStatus.VALIDATING), RunStatus.INFERRING)
    with pytest.raises(IllegalTransition):
        transition(_snapshot(RunStatus.FINISHED), RunStatus.QUEUED)


def test_memory_store_keeps_each_saved_snapshot():
    store = InMemoryStateStore()
    first = _snapshot(RunStatus.QUEUED)
    second = _snapshot(RunStatus.FINISHED)

    store.save(first)
    store.save(second)

    assert [item.status for item in store.history] == [RunStatus.QUEUED, RunStatus.FINISHED]
    assert store.load("run").status is RunStatus.FINISHED
    assert store.load("missing") is None


def test_input_queue_drains_in_fifo_order_and_is_empty_after():
    queue = InputQueue()
    queue.push("a")
    queue.push("b")

    assert queue.drain() == ["a", "b"]
    assert queue.drain() == []


def test_hooks_run_in_registration_order_and_can_replace_context():
    bus = HookBus()
    seen = []

    def first(ctx):
        seen.append("first")
        return {**ctx, "n": 1}

    def second(ctx):
        seen.append(ctx["n"])
        return None

    bus.register(HookPoint.BEFORE_STEP, first)
    bus.register(HookPoint.BEFORE_STEP, second)

    assert bus.emit(HookPoint.BEFORE_STEP, {"n": 0}) == {"n": 1}
    assert seen == ["first", 1]
    assert config().max_steps == 8
