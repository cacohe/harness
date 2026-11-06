import contextvars
import json
import tempfile
from pathlib import Path

import pytest
from src.logger import get_logger, log_context, setup_logging


def test_get_logger_uses_the_project_namespace():
    assert get_logger().name == "harness"
    assert get_logger("core.loop").name == "harness.core.loop"
    assert get_logger("harness.core.loop").name == "harness.core.loop"


def test_text_log_includes_the_active_context(capsys):
    setup_logging(level="INFO")
    with log_context(task_id="t1"):
        with log_context(step=2):
            get_logger("core.loop").info("step finished")
        get_logger("core.loop").info("task still running")

    lines = capsys.readouterr().err.splitlines()
    assert "INFO" in lines[0]
    assert "harness.core.loop" in lines[0]
    assert "step finished" in lines[0]
    assert "task_id=t1" in lines[0]
    assert "step=2" in lines[0]
    assert "task_id=t1" in lines[1]
    assert "step=" not in lines[1]


def test_log_context_restores_fields_when_the_body_raises(capsys):
    setup_logging(level="INFO")
    with pytest.raises(RuntimeError, match="boom"):
        with log_context(task_id="t1"):
            raise RuntimeError("boom")

    get_logger("core.loop").info("after")
    assert "task_id" not in capsys.readouterr().err


def test_json_log_is_one_object_per_line(capsys):
    setup_logging(level="DEBUG", json_logs=True)
    with log_context(task_id="t1"):
        get_logger("core.llm").warning("retrying")

    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "harness.core.llm"
    assert payload["message"] == "retrying"
    assert payload["context"] == {"task_id": "t1"}
    assert "time" in payload


def test_debug_lines_are_hidden_when_the_level_is_info(capsys):
    setup_logging(level="INFO")
    get_logger("core.loop").debug("hidden")
    get_logger("core.loop").info("shown")

    err = capsys.readouterr().err
    assert "hidden" not in err
    assert "shown" in err


def test_json_log_includes_the_traceback(capsys):
    setup_logging(level="ERROR", json_logs=True)
    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("core.loop").exception("task failed")

    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["message"] == "task failed"
    assert "ValueError: boom" in payload["exception"]


def test_file_log_is_json_even_when_the_console_is_text(capsys):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "app.log"
        setup_logging(level="INFO", log_file=path)
        with log_context(task_id="t1"):
            get_logger("core.guard").error("rejected")

        console = capsys.readouterr().err
        assert "rejected" in console
        assert not console.strip().startswith("{")

        payload = json.loads(path.read_text(encoding="utf-8").strip())
        assert payload["level"] == "ERROR"
        assert payload["message"] == "rejected"
        assert payload["context"] == {"task_id": "t1"}
        setup_logging()


def test_setup_logging_replaces_previous_handlers(capsys):
    setup_logging(level="INFO")
    setup_logging(level="INFO")
    get_logger("core.loop").info("once")

    assert len(capsys.readouterr().err.splitlines()) == 1


def test_log_context_does_not_leak_into_another_context(capsys):
    setup_logging(level="INFO", json_logs=True)

    def inside():
        with log_context(task_id="child"):
            get_logger("core.loop").info("from child")

    with log_context(task_id="parent"):
        contextvars.copy_context().run(inside)
        get_logger("core.loop").info("from parent")

    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert lines[0]["context"]["task_id"] == "child"
    assert lines[1]["context"]["task_id"] == "parent"
