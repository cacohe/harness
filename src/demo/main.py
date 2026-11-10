import sys

from src.config import ConfigError
from src.core.task import Task
from src.demo.agent import Agent
from src.logger import setup_logging

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}
HELP_COMMAND = "/help"


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
    Args:
        agent: Object with chat(message) -> Task.
        input_fn: Reads one line. Receives the prompt and raises EOFError at end of input.
        output_fn: Prints one line.
    """
    output_fn("agent-harness. Enter a message, or /exit to quit.")
    while True:
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


def format_turn(task: Task) -> str:
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
