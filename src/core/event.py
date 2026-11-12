import sys
from dataclasses import asdict
from datetime import datetime
from typing import Callable

from src.logger import get_logger
from src.schema.event import Event, EventChannel, EventType

logger = get_logger("core.event")


def _client_line(event: Event) -> str:
    parts = [f"client {event.type.value}"]
    name = getattr(event.payload, "name", None)
    arguments = getattr(event.payload, "arguments", None)
    content = getattr(event.payload, "content", None)
    reason = getattr(event.payload, "reason", None)
    goal = getattr(event.payload, "goal", None)
    if name:
        parts.append(str(name))
    if arguments:
        parts.append(str(arguments))
    if goal:
        parts.append(str(goal))
    if reason:
        parts.append(str(reason))
    if content and event.type is not EventType.TASK_COMPLETED and not name:
        parts.append(str(content))
    return " ".join(parts)


def _tracing_line(event: Event) -> str:
    timestamp = event.timestamp
    if isinstance(timestamp, datetime):
        stamp = timestamp.isoformat(timespec="seconds")
    else:
        stamp = str(timestamp)
    fields = [f"id={event.id}", f"type={event.type.value}"]
    if event.task_id:
        fields.append(f"task_id={event.task_id}")
    if event.step_id:
        fields.append(f"step_id={event.step_id}")
    for key, value in asdict(event.payload).items():
        if value is None or key == "task_id":
            continue
        fields.append(f"{key}={value}")
    return f"tracing {stamp} " + " ".join(fields)


def client_emitter(event: Event) -> None:
    """Print one user-visible event to standard output."""
    if event.type is EventType.LLM_DELTA:
        print(event.payload.content or "", end="", flush=True, file=sys.stdout)
        return
    if event.type is EventType.TASK_COMPLETED:
        print(file=sys.stdout)
    print(_client_line(event), file=sys.stdout)


def tracing_emitter(event: Event) -> None:
    """Print one tracing event to standard output."""
    print(_tracing_line(event), file=sys.stdout)


class EventHandler:
    """Event handler."""

    def __init__(
        self,
        client_emitter: Callable[[Event], None] | None = client_emitter,
        tracing_emitter: Callable[[Event], None] | None = tracing_emitter,
    ):
        self.client_emitter = client_emitter
        self.tracing_emitter = tracing_emitter

    def emit_client(self, event: Event) -> None:
        """Emit the event to the client."""
        if self.client_emitter is None:
            return
        try:
            self.client_emitter(event)
        except Exception:
            logger.exception("event emitter failed", extra={"context": {"channel": "client"}})
            return

    def emit_tracing(self, event: Event) -> None:
        """Emit the event to the tracing."""
        if self.tracing_emitter is None:
            return
        try:
            self.tracing_emitter(event)
        except Exception:
            logger.exception("event emitter failed", extra={"context": {"channel": "tracing"}})
            return

    def emit(self, event: Event, channel: EventChannel = EventChannel.CLIENT) -> None:
        """Emit the event to the specified channel."""
        if channel == EventChannel.CLIENT:
            self.emit_client(event)
        elif channel == EventChannel.TRACING:
            self.emit_tracing(event)
        elif channel == EventChannel.ALL:
            self.emit_client(event)
            self.emit_tracing(event)
        else:
            raise ValueError(f"Invalid event channel: {channel}")
