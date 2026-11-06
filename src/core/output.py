from src.schema.output import OutputType


class OutputHandler:
    """Output handler."""

    def emit_event(self) -> None:
        """Emit the event."""
        ...

    def emit_log(self) -> None:
        """Emit the log."""
        ...

    def emit(self, type: OutputType) -> None:
        """Emit the output."""
        if type == OutputType.ALL_TYPES:
            self.emit_event()
            self.emit_log()
        elif type == OutputType.EVENT_TYPE:
            self.emit_event()
        elif type == OutputType.LOG_TYPE:
            self.emit_log()

    def persist(self) -> None: ...

    def to_opentelemetry(self) -> dict: ...
