import threading

from harness import (
    AgentKind,
    ApprovalRequirement,
    ExecCommand,
    HookPoint,
    InMemoryStateStore,
    ListObserver,
    OutputMode,
    RunStatus,
    ToolSchedule,
)

from support import (
    EchoBuilder,
    FixedMemory,
    RecordingApprover,
    RecordingExecutor,
    RecordingSafety,
    RejectAll,
    ScriptedGateway,
    StaticApproval,
    call,
    config,
    make_harness,
    remembered,
    text_output,
    tool_output,
)


def test_text_answer_finishes_and_records_transitions():
    store = InMemoryStateStore()
    observer = ListObserver()
    gateway = ScriptedGateway([text_output("hello")])
    result = make_harness(gateway, state_store=store, observer=observer).run("ship it")

    assert result.status is RunStatus.FINISHED
    assert result.output == "hello"
    assert result.error is None
    statuses = [item.status for item in store.history]
    assert statuses[0] is RunStatus.QUEUED
    assert RunStatus.VALIDATING in statuses
    assert RunStatus.INFERRING in statuses
    assert statuses[-1] is RunStatus.FINISHED
    assert store.load(result.run_id).status is RunStatus.FINISHED
    kinds = [event.kind for event in observer.events]
    assert "model" in kinds
    assert "output" in kinds
    assert all(event.run_id == result.run_id for event in observer.events)
    assert gateway.contexts[0].system_prompt
    assert gateway.contexts[0].messages[0].content == "ship it"
    assert gateway.contexts[0].goal == "ship it"


def test_task_agent_buffers_execution_and_streams_the_final_event():
    observer = ListObserver()
    gateway = ScriptedGateway([text_output("hello")])
    make_harness(gateway, observer=observer, harness_config=config(agent_kind=AgentKind.TASK)).run("ship it")

    assert gateway.modes == [OutputMode.BUFFERED]
    payload = next(event.payload for event in observer.events if event.kind == "output")
    assert payload["mode"] == "stream"
    assert payload["event_mode"] == "stream"


def test_validator_rejection_stops_before_the_model():
    gateway = ScriptedGateway([text_output("hello")])
    result = make_harness(gateway, validator=RejectAll("nope")).run("ship it")

    assert result.status is RunStatus.FAILED
    assert result.error == "nope"
    assert gateway.modes == []


def test_tool_result_is_sent_back_to_the_model():
    executor = RecordingExecutor()
    gateway = ScriptedGateway(
        [tool_output(call("echo", "1")), text_output("done")]
    )
    result = make_harness(gateway, executor=executor).run("ship it")

    assert result.status is RunStatus.FINISHED
    assert result.output == "done"
    assert [item.name for item in executor.calls] == ["echo"]
    roles = [message.role for message in gateway.contexts[1].messages]
    assert roles == ["user", "assistant", "tool"]
    assert gateway.contexts[1].messages[2].content == "ran:echo"
    assert gateway.contexts[1].messages[2].tool_call_id == "1"


def test_forbidden_and_safety_denial_do_not_execute():
    executor = RecordingExecutor()
    gateway = ScriptedGateway(
        [
            tool_output(call("rm", "1"), call("echo", "2")),
            text_output("done"),
        ]
    )

    class SplitApproval:
        def decide(self, tool_call, policy, state):
            if tool_call.name == "rm":
                return StaticApproval(ApprovalRequirement.FORBIDDEN).decide(tool_call, policy, state)
            return StaticApproval(ApprovalRequirement.SKIP).decide(tool_call, policy, state)

    class DenyEcho:
        def check(self, action, environment):
            if action.name == "echo":
                return "deny"
            return "allow"

    result = make_harness(
        gateway,
        executor=executor,
        approval=SplitApproval(),
        safety=DenyEcho(),
    ).run("ship it")

    assert result.status is RunStatus.FINISHED
    assert executor.calls == []
    tool_messages = [message.content for message in gateway.contexts[1].messages if message.role == "tool"]
    assert tool_messages == ["forbidden", "safety denied"]


def test_skip_still_asks_safety_and_human_denial_passes_the_requirement():
    executor = RecordingExecutor()
    approver = RecordingApprover(verdict="denied")
    safety = RecordingSafety()
    action = ExecCommand(command="ls")
    gateway = ScriptedGateway([tool_output(call("ls", "1")), text_output("done")])
    result = make_harness(
        gateway,
        executor=executor,
        approver=approver,
        safety=safety,
        approval=StaticApproval(ApprovalRequirement.NEEDS_HUMAN_APPROVAL, action=action),
    ).run("ship it")

    assert result.status is RunStatus.FINISHED
    assert executor.calls == []
    assert safety.actions == []
    assert approver.reviews == [(action, ApprovalRequirement.NEEDS_HUMAN_APPROVAL)]
    assert gateway.contexts[1].messages[-1].content == "denied"


def test_skip_runs_safety_before_the_executor():
    executor = RecordingExecutor()
    safety = RecordingSafety()
    action = ExecCommand(command="ls")
    gateway = ScriptedGateway([tool_output(call("ls", "1")), text_output("done")])
    make_harness(
        gateway,
        executor=executor,
        safety=safety,
        approval=StaticApproval(ApprovalRequirement.SKIP, action=action),
    ).run("ship it")

    assert [item.command for item in safety.actions] == ["ls"]
    assert [item.name for item in executor.calls] == ["ls"]


def test_session_grant_skips_later_approval_but_not_safety():
    executor = RecordingExecutor()
    approver = RecordingApprover()
    safety = RecordingSafety()
    gateway = ScriptedGateway(
        [
            tool_output(call("echo", "1", value=1), call("echo", "2", value=2)),
            text_output("done"),
        ]
    )
    make_harness(
        gateway,
        executor=executor,
        approver=approver,
        safety=safety,
        approval=StaticApproval(ApprovalRequirement.NEEDS_APPROVAL, scope_kind="session"),
    ).run("ship it")

    assert len(approver.reviews) == 1
    assert len(safety.actions) == 2
    assert [item.name for item in executor.calls] == ["echo", "echo"]


def test_exact_grant_matches_arguments_and_once_does_not():
    approver = RecordingApprover()
    gateway = ScriptedGateway(
        [
            tool_output(call("echo", "1", value=1)),
            tool_output(call("echo", "2", value=1)),
            tool_output(call("echo", "3", value=2)),
            text_output("done"),
        ]
    )
    make_harness(
        gateway,
        approver=approver,
        approval=StaticApproval(ApprovalRequirement.NEEDS_APPROVAL, scope_kind="exact"),
        harness_config=config(max_steps=5),
    ).run("ship it")

    assert len(approver.reviews) == 2

    once_approver = RecordingApprover()
    once_gateway = ScriptedGateway(
        [
            tool_output(call("echo", "1", value=1)),
            tool_output(call("echo", "2", value=1)),
            text_output("done"),
        ]
    )
    make_harness(
        once_gateway,
        approver=once_approver,
        approval=StaticApproval(ApprovalRequirement.NEEDS_APPROVAL, scope_kind="once"),
    ).run("ship it")

    assert len(once_approver.reviews) == 2


def test_submit_during_run_is_applied_on_the_next_turn_without_changing_the_default_goal():
    builder = EchoBuilder()
    gateway = ScriptedGateway([tool_output(call("echo", "1")), text_output("done")])
    harness = make_harness(gateway, context_builder=builder)
    submitted = {"done": False}

    def before_infer(ctx):
        if not submitted["done"]:
            harness.submit("more")
            submitted["done"] = True
        return None

    harness.hooks.register(HookPoint.BEFORE_INFER, before_infer)
    harness.run("ship it", goal="keep")

    assert gateway.contexts[1].goal == "keep"
    assert gateway.contexts[1].strategy == ""
    assert [message.content for message in gateway.contexts[1].messages if message.role == "user"] == [
        "ship it",
        "more",
    ]


def test_custom_goal_adjuster_sees_queued_input():
    from harness import GoalUpdate

    gateway = ScriptedGateway([tool_output(call("echo", "1")), text_output("done")])

    class Adjust:
        def adjust(self, state, incoming):
            if any(item.content == "more" for item in incoming):
                return GoalUpdate(goal="new-goal", strategy="replan")
            return GoalUpdate(goal=state.goal, strategy=state.strategy)

    harness = make_harness(gateway, goal_adjuster=Adjust())
    once = {"done": False}

    def before_infer(ctx):
        if not once["done"]:
            harness.submit("more")
            once["done"] = True
        return ctx

    harness.hooks.register(HookPoint.BEFORE_INFER, before_infer)
    harness.run("ship it", goal="keep")

    assert gateway.contexts[1].goal == "new-goal"
    assert gateway.contexts[1].strategy == "replan"


def test_before_infer_hook_can_replace_the_context():
    from harness import LlmContext

    gateway = ScriptedGateway([text_output("hello")])
    harness = make_harness(gateway)

    def before_infer(ctx):
        return {**ctx, "context": LlmContext(system_prompt="replaced", messages=(), tools=(), goal="x", strategy="")}

    harness.hooks.register(HookPoint.BEFORE_INFER, before_infer)
    harness.run("ship it")

    assert gateway.contexts[0].system_prompt == "replaced"


def test_max_steps_stops_a_tool_loop():
    gateway = ScriptedGateway([tool_output(call("echo", "1")), tool_output(call("echo", "2")), text_output("done")])
    result = make_harness(gateway, harness_config=config(max_steps=2)).run("ship it")

    assert result.status is RunStatus.FAILED
    assert result.error == "max steps exceeded"
    assert len(gateway.modes) == 2


def test_cancel_during_validation_skips_the_model():
    gateway = ScriptedGateway([text_output("hello")])
    harness = make_harness(gateway)

    def after_validate(ctx):
        harness.cancel()
        return None

    harness.hooks.register(HookPoint.AFTER_VALIDATE, after_validate)
    result = harness.run("ship it")

    assert result.status is RunStatus.CANCELLED
    assert gateway.modes == []


def test_memory_search_result_is_given_to_the_context_builder():
    builder = EchoBuilder()
    memory = FixedMemory((remembered("note"),))
    gateway = ScriptedGateway([text_output("hello")])
    make_harness(gateway, context_builder=builder, memory=memory).run("ship it", goal="keep")

    assert memory.queries[0].goal == "keep"
    assert gateway.contexts[0].memories[0].content == "note"
    assert builder.args[0].memories[0].content == "note"


def test_isolated_environment_is_a_todo_and_does_not_execute():
    executor = RecordingExecutor()
    gateway = ScriptedGateway([tool_output(call("echo", "1"))])
    result = make_harness(
        gateway,
        executor=executor,
        harness_config=config(execution_kind="isolated"),
    ).run("ship it")

    assert result.status is RunStatus.FAILED
    assert result.error.startswith("TODO")
    assert executor.calls == []


def test_async_finish_waits_for_the_real_tool_result():
    release = threading.Event()

    def execute(tool_call, handle):
        assert release.wait(2)
        from harness import ToolResult

        return ToolResult(call_id=tool_call.id, ok=True, output="real-result")

    executor = RecordingExecutor(execute=execute)

    class Gateway:
        def __init__(self):
            self.contexts = []
            self.modes = []

        def complete(self, context, mode):
            self.modes.append(mode)
            self.contexts.append(context)
            contents = [message.content for message in context.messages]
            if "real-result" in contents:
                return text_output("done")
            if "accepted" in contents:
                release.set()
                return text_output("early")
            return tool_output(call("echo", "1"))

    gateway = Gateway()
    result = make_harness(
        gateway,
        executor=executor,
        harness_config=config(tool_schedule=ToolSchedule(completion="async"), max_steps=4),
    ).run("ship it")

    assert result.output == "done"
    assert result.status is RunStatus.FINISHED
    assert any("real-result" in [message.content for message in context.messages] for context in gateway.contexts)
