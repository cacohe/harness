from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

LOGGER_NAME = "agent_harness"
_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("agent_harness_log_context", default={})


class _TextFormatter(logging.Formatter):
    """One human-readable line, with context fields appended."""

    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        context = getattr(record, "context", None) or {}
        if not context:
            return line
        fields = " ".join(f"{key}={value}" for key, value in context.items())
        return f"{line} {fields}"


class _JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def __init__(self) -> None:
        super().__init__(datefmt="%Y-%m-%dT%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "context": getattr(record, "context", None) or {},
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _ContextFilter(logging.Filter):
    """Attach the active log context to each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = dict(_LOG_CONTEXT.get())
        explicit = getattr(record, "context", None) or {}
        context.update(explicit)
        record.context = context
        return True


def setup_logging(
    level: str | int = "INFO",
    *,
    json_logs: bool = False,
    log_file: str | Path | None = None,
) -> None:
    """
    Configure the project logger.

    Console output is plain text unless json_logs is set. A log file, when
    given, is always JSON so a run can be replayed. Calling this again closes
    the previous handlers.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_JsonFormatter() if json_logs else _TextFormatter())
    console.addFilter(_ContextFilter())
    logger.addHandler(console)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(_JsonFormatter())
        file_handler.addFilter(_ContextFilter())
        logger.addHandler(file_handler)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the agent_harness namespace."""
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """
    Attach fields to every log emitted inside the block.
    Args:
        **fields: Values merged over the fields from the enclosing block.
    """
    token = _LOG_CONTEXT.set({**_LOG_CONTEXT.get(), **fields})
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)
