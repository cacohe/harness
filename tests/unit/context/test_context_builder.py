from src.context.context_builder import DefaultContextBuilder
from src.core.task import Task
from src.schema.context import Context
from src.schema.task import TaskPlan


def _task() -> Task:
    return Task(original_input="buy a phone")


def test_later_steps_keep_the_goal_and_drop_stale_plan_steps():
    task = _task()
    task.plan = TaskPlan(goal=["look"], current="read", remaining=["summarize"])

    first = DefaultContextBuilder().build_infer_context("Be brief.", task, [], [])
    assert "summarize" in first.messages[-1].content

    task.history = [first.messages[1]]
    later = DefaultContextBuilder().build_infer_context("Be brief.", task, [], [])
    assert "look" in later.messages[-1].content
    assert "summarize" not in later.messages[-1].content


def test_builder_reads_history_from_the_task():
    task = _task()
    task.plan = TaskPlan(goal=[])

    context = DefaultContextBuilder().build_infer_context("Be brief.", task, [], [])

    assert isinstance(context, Context)
    assert context.messages[0].role == "system"
    assert context.messages[1].content == "buy a phone"
    assert context.tools == []
