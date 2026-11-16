import json
import sys

from src.application.agent import Agent
from src.config import ConfigError
from src.core.task import Task
from src.logger import setup_logging

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}
HELP_COMMAND = "/help"
YES_ANSWERS = {"y", "yes"}
NO_ANSWERS = {"n", "no"}


def main() -> None:
    setup_logging(level="INFO")
    try:
        agent = Agent()
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    repl(agent)


def repl(agent, input_fn=input, output_fn=print) -> None:
    """
    Read lines from the terminal and print errors or pauses. Completed answers are streamed.
    A paused task asks for a decision on each tool call before it resumes.
    Args:
        agent: Object with chat(message) -> Task. Agents that pause also have pending_task,
            approval_requests(), decide_tool_call(tool_id, approved), and resume() -> Task.
        input_fn: Reads one line. Receives the prompt and raises EOFError at end of input.
        output_fn: Prints one line.
    """
    output_fn("agent-harness. Enter a message, or /exit to quit.")
    while True:
        if getattr(agent, "pending_task", None) is not None:
            if not review_approvals(agent, input_fn, output_fn):
                return
            try:
                task = agent.resume()
            except KeyboardInterrupt:
                output_fn("interrupted")
                continue
            except Exception as error:
                output_fn(f"error: {error}")
                continue
            rendered = format_turn(task)
            if rendered:
                output_fn(rendered)
            continue

        try:
            line = input_fn("> ")
        except EOFError:
            output_fn("")
            return
        except KeyboardInterrupt:
            output_fn("")
            return

        command = line.strip()
        if not command:
            continue
        if command in EXIT_COMMANDS:
            return
        if command == HELP_COMMAND:
            output_fn("Commands: /help, /exit, /quit")
            continue

        try:
            task = agent.chat(command)
        except KeyboardInterrupt:
            output_fn("interrupted")
            continue
        except Exception as error:
            output_fn(f"error: {error}")
            continue
        rendered = format_turn(task)
        if rendered:
            output_fn(rendered)


def review_approvals(agent, input_fn, output_fn) -> bool:
    """
    Ask the user to approve or reject each tool call waiting on the paused task.
    Args:
        agent: Object with approval_requests() and decide_tool_call(tool_id, approved).
        input_fn: Reads one line. Receives the prompt and raises EOFError at end of input.
        output_fn: Prints one line.
    Returns:
        True when every call is decided. False when input ended or was interrupted.
    """
    requests = agent.approval_requests()
    for number, item in enumerate(requests, start=1):
        output_fn(f"Approval {number}/{len(requests)}: {item.tool_name}")
        for line in describe_arguments(item.arguments):
            output_fn(f"  {line}")
        approved = ask_yes_no(input_fn, output_fn)
        if approved is None:
            return False
        agent.decide_tool_call(item.tool_id, approved)
    return True


def describe_arguments(arguments: str | None) -> list[str]:
    """
    Turn the model's tool arguments into lines a person can read.
    Args:
        arguments: The raw JSON arguments from the model.
    Returns:
        One "name: value" line per argument, or the raw text when it is not a JSON object.
    """
    if not arguments:
        return []
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return [arguments]
    if not isinstance(parsed, dict):
        return [arguments]
    return [f"{name}: {value}" for name, value in parsed.items()]


def ask_yes_no(input_fn, output_fn) -> bool | None:
    """
    Ask until the user answers yes or no.
    Args:
        input_fn: Reads one line. Receives the prompt and raises EOFError at end of input.
        output_fn: Prints one line.
    Returns:
        True for yes, False for no, or None when input ended or was interrupted.
    """
    while True:
        try:
            answer = input_fn("Approve? [y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return None
        if answer in YES_ANSWERS:
            return True
        if answer in NO_ANSWERS:
            return False
        output_fn("Enter y or n.")


def format_turn(task: Task) -> str | None:
    """
    Render a turn that the event stream did not already show.
    Args:
        task: The task returned by chat.
    Returns:
        A short error or pause line. A completed answer is empty because it was streamed.
    """
    if task.is_completed():
        return ""
    if task.is_failed():
        reason = task.state_reason.value if task.state_reason else "failed"
        return f"error: {reason}"
    if task.is_paused():
        reason = task.state_reason.value if task.state_reason else "paused"
        return f"paused: {reason}"
    if task.result is not None:
        return task.result.output
    return ""


if __name__ == "__main__":
    main()
