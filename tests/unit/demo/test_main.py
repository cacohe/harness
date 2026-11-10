from src.core.task import Task
from src.demo.main import repl
from src.schema.task import TaskFailReason, TaskResult, TaskRunningReason


class ScriptedAgent:
    def __init__(self, results):
        self.results = list(results)
        self.messages = []

    def chat(self, message):
        self.messages.append(message)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        task = Task(original_input=message)
        if isinstance(result, TaskFailReason):
            task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
            task.transition_to_failed(result)
            return task
        task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
        task.set_result(TaskResult(result))
        task.transition_to_completed()
        return task


def _run(agent, lines):
    printed = []
    pending = iter(lines)

    def read(_prompt):
        try:
            return next(pending)
        except StopIteration:
            raise EOFError from None

    repl(agent, input_fn=read, output_fn=printed.append)
    return printed


def test_repl_prints_each_answer_until_exit():
    agent = ScriptedAgent(["hello", "there"])

    printed = _run(agent, ["hi", "again", "/exit", "ignored"])

    assert agent.messages == ["hi", "again"]
    assert printed[0].startswith("agent-harness")
    assert "hello" not in printed
    assert "there" not in printed


def test_repl_skips_blank_lines_and_reports_failures():
    agent = ScriptedAgent([RuntimeError("down"), TaskFailReason.TASK_FAILED_TIMEOUT])

    printed = _run(agent, ["", "   ", "boom", "slow", "/quit"])

    assert agent.messages == ["boom", "slow"]
    assert "error: down" in printed
    assert "error: timeout" in printed


def test_repl_stops_on_eof_without_calling_the_agent():
    agent = ScriptedAgent([])

    printed = _run(agent, [])

    assert agent.messages == []
    assert printed[-1] == ""
