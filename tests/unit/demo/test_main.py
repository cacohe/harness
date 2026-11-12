from src.core.task import Task
from src.demo.main import repl
from src.schema.task import (
    TaskFailReason,
    TaskPauseReason,
    TaskResult,
    TaskRunningReason,
)
from src.schema.tool import ToolExecutionResult


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


class Approving:
    def __init__(self, requests):
        self.pending_task = Task(original_input="run")
        self.pending_task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
        self.pending_task.transition_to_paused(TaskPauseReason.TASK_PAUSED_WAITING_APPROVAL)
        self.requests = requests
        self.decisions = []
        self.resumed = False

    def approval_requests(self):
        decided = {tool_id for tool_id, _ in self.decisions}
        return [item for item in self.requests if item.tool_id not in decided]

    def decide_tool_call(self, tool_id, approved):
        self.decisions.append((tool_id, approved))

    def resume(self):
        self.resumed = True
        task = self.pending_task
        task.transition_to_running(TaskRunningReason.TASK_RUNNING_RESUME)
        task.set_result(TaskResult("ran"))
        task.transition_to_completed()
        self.pending_task = None
        return task


def _request(tool_id, name, arguments):
    return ToolExecutionResult(
        tool_id=tool_id, tool_name=name, arguments=arguments, needs_approval=True
    )


def test_repl_asks_for_each_tool_call_then_resumes():
    agent = Approving(
        [
            _request("call_1", "shell", '{"command": "git status"}'),
            _request("call_2", "write_file", '{"path": "x.txt", "content": "hi"}'),
        ]
    )

    printed = _run(agent, ["y", "n", "/exit"])

    assert agent.decisions == [("call_1", True), ("call_2", False)]
    assert agent.resumed
    text = "\n".join(printed)
    assert "shell" in text
    assert "command: git status" in text
    assert "write_file" in text
    assert "path: x.txt" in text
    assert "call_1" not in text


def test_repl_asks_again_until_the_answer_is_yes_or_no():
    agent = Approving([_request("call_1", "shell", '{"command": "git status"}')])

    printed = _run(agent, ["maybe", "yes", "/exit"])

    assert agent.decisions == [("call_1", True)]
    assert "Enter y or n." in printed


def test_repl_stops_on_eof_during_approval_without_resuming():
    agent = Approving([_request("call_1", "shell", '{"command": "git status"}')])

    _run(agent, [])

    assert agent.decisions == []
    assert not agent.resumed


def test_repl_stops_on_eof_without_calling_the_agent():
    agent = ScriptedAgent([])

    printed = _run(agent, [])

    assert agent.messages == []
    assert printed[-1] == ""
