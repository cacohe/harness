import asyncio
import inspect
import time

from src.adapters.observability import CompositeObserver, EventObserver, MetricsObserver
from src.application.composition import create_production_agent
from src.application.observability import Observation


def test_composite_isolates_failure_and_metrics_bound_cardinality(capsys) -> None:
    class Broken:
        async def observe(self, _observation):
            raise RuntimeError("no")

    async def scenario():
        metrics = MetricsObserver(max_series=1)
        composite = CompositeObserver((Broken(), metrics))
        await composite.observe(Observation("accepted", "s", "r", "running", 1, 1))
        await composite.observe(Observation("advanced", "s", "r", "completed", 2, 2))
        snapshot = await metrics.snapshot()
        assert snapshot.counts == (("accepted", "running", 1),)
        assert snapshot.overflow == 1

    asyncio.run(scenario())
    assert capsys.readouterr().out == ""


def test_composite_preserves_order_and_event_has_no_content() -> None:
    seen = []

    class Recorder:
        def __init__(self, name):
            self.name = name

        async def observe(self, observation):
            seen.append((self.name, observation.kind))

    events = []

    async def scenario():
        observation = Observation("advanced", "s", "r", "running", 2, 3)
        await CompositeObserver(
            (Recorder("one"), Recorder("two"), EventObserver(events.append))
        ).observe(observation)

    asyncio.run(scenario())
    assert seen == [("one", "advanced"), ("two", "advanced")]
    assert set(events[0]) == {
        "kind",
        "session_id",
        "run_id",
        "status",
        "journal_version",
        "journal_sequence",
    }


def test_metrics_is_async_safe_and_default_composition_has_no_observer() -> None:
    async def scenario():
        metrics = MetricsObserver(max_series=2)
        observation = Observation("advanced", "s", "r", "running", 1, 1)
        await asyncio.gather(*(metrics.observe(observation) for _ in range(100)))
        snapshot = await metrics.snapshot()
        assert snapshot.counts == (("advanced", "running", 100),)

    asyncio.run(scenario())
    assert inspect.signature(create_production_agent).parameters["observer"].default is None


def test_sync_event_sink_runs_off_event_loop() -> None:
    def sink(_event):
        time.sleep(0.03)

    async def scenario():
        task = asyncio.create_task(
            EventObserver(sink).observe(Observation("x", "s", "r", "running", 1, 1))
        )
        await asyncio.sleep(0.005)
        assert not task.done()
        await task

    asyncio.run(scenario())
