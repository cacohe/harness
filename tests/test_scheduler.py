import threading

from harness import (
    CurrentEnvironment,
    OutputMode,
    RunStatus,
    StructuralDiscriminator,
    ToolResult,
    ToolSchedule,
    ToolScheduler,
)

from support import RecordingExecutor, call, config, make_harness, text_output, tool_output


def test_structural_discriminator_finishes_on_text_uses_tools_or_fails_empty():
    discriminator = StructuralDiscriminator()
    state = None

    assert discriminator.decide(text_output("done"), state).text == "done"
    tools = discriminator.decide(tool_output(call("echo")), state)
    assert tools.type == "tools"
    assert tools.calls[0].name == "echo"
    failure = discriminator.decide(text_output(None), state)
    assert failure.type == "fail"
    assert failure.reason == "empty model output"


def test_serial_sync_scheduler_keeps_call_order():
    executor = RecordingExecutor()
    environment = CurrentEnvironment(executor)
    handle = environment.prepare()
    calls = [call("a", "1"), call("b", "2")]

    results = ToolScheduler().execute(calls, ToolSchedule(), environment, handle)

    assert executor.calls == calls
    assert [result.output for result in results] == ["ran:a", "ran:b"]
    environment.dispose(handle)


def test_parallel_sync_scheduler_overlaps_calls():
    barrier = threading.Barrier(2, timeout=2)

    def execute(tool_call, handle):
        barrier.wait()
        return ToolResult(call_id=tool_call.id, ok=True, output=tool_call.name)

    executor = RecordingExecutor(execute=execute)
    environment = CurrentEnvironment(executor)
    handle = environment.prepare()

    results = ToolScheduler().execute(
        [call("a", "1"), call("b", "2")],
        ToolSchedule(concurrency="parallel"),
        environment,
        handle,
    )

    assert {result.output for result in results} == {"a", "b"}


def test_async_scheduler_returns_placeholder_before_the_executor_finishes():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    holder = {}

    def execute(tool_call, handle):
        started.set()
        assert release.wait(2)
        return ToolResult(call_id=tool_call.id, ok=True, output="real-result")

    def on_complete(result):
        holder["result"] = result
        finished.set()

    executor = RecordingExecutor(execute=execute)
    environment = CurrentEnvironment(executor)
    handle = environment.prepare()

    results = ToolScheduler().execute(
        [call("echo", "1")],
        ToolSchedule(completion="async"),
        environment,
        handle,
        on_complete=on_complete,
    )

    assert results[0].accepted is True
    assert results[0].output == "accepted"
    assert started.wait(2)
    assert "result" not in holder
    release.set()
    assert finished.wait(2)
    assert holder["result"].output == "real-result"


def test_empty_model_output_fails_the_run():
    gateway_outputs = [text_output(None)]
    from support import ScriptedGateway

    gateway = ScriptedGateway(gateway_outputs)
    result = make_harness(gateway).run("hello")

    assert result.status is RunStatus.FAILED
    assert result.error == "empty model output"


def test_explicit_output_mode_reaches_the_gateway_and_the_final_event():
    from harness import AgentKind, ListObserver

    from support import ScriptedGateway

    observer = ListObserver()
    gateway = ScriptedGateway([text_output("hello")])
    result = make_harness(
        gateway,
        observer=observer,
        harness_config=config(output_mode=OutputMode.BUFFERED, agent_kind=AgentKind.CONVERSATIONAL),
    ).run("hello")

    assert result.output == "hello"
    assert gateway.modes == [OutputMode.BUFFERED]
    output_event = next(event for event in observer.events if event.kind == "output")
    assert output_event.payload["mode"] == "buffered"
    assert output_event.payload["event_mode"] == "buffered"
