import pytest

from harness import (
    IsolatedEnvironment,
    MemoryQuery,
    TodoApprover,
    TodoApprovalPolicy,
    TodoContextBuilder,
    TodoExecutor,
    TodoLlmGateway,
    TodoMemoryStore,
    TodoSafetyAugment,
    TodoStateStore,
    TodoTraceExporter,
    TodoValidator,
    ToolCall,
)


def test_unimplemented_ports_are_marked_todo():
    call = ToolCall(id="1", name="echo", arguments={})
    cases = [
        lambda: TodoValidator().validate((), None),
        lambda: TodoContextBuilder().build(None),
        lambda: TodoLlmGateway().complete(None, None),
        lambda: TodoApprovalPolicy().decide(call, None, None),
        lambda: TodoApprover().review(None, None),
        lambda: TodoSafetyAugment().check(None, None),
        lambda: TodoExecutor().execute(call, None),
        lambda: TodoMemoryStore().search(MemoryQuery(goal="g", text="t")),
        lambda: TodoMemoryStore().save(None),
        lambda: TodoStateStore().save(None),
        lambda: TodoStateStore().load("run"),
        lambda: TodoTraceExporter().export(()),
        lambda: TodoTraceExporter().replay("run"),
        lambda: IsolatedEnvironment().prepare(),
    ]

    for case in cases:
        with pytest.raises(NotImplementedError, match="TODO"):
            case()
