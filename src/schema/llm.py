from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_TRANSIENT_ERRORS = {
    "Timeout",
    "TimeoutError",
    "APITimeoutError",
    "APIConnectionError",
    "APIConnectionTimeoutError",
    "ConnectError",
    "ReadError",
    "RemoteProtocolError",
    "ConnectionResetError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
    "OverloadedError",
}
_PERMANENT_STATUS = {400, 401, 402, 403, 404, 413, 422}
_PERMANENT_ERRORS = {
    "AuthenticationError",
    "PermissionDeniedError",
    "BadRequestError",
    "NotFoundError",
    "UnprocessableEntityError",
    "ContentPolicyViolationError",
    "ContextWindowExceededError",
    "RequestTooLargeError",
    "BudgetExceededError",
    "UnsupportedParamsError",
}
_QUOTA_ERROR_CODES = {"insufficient_quota"}
_UNUSABLE_OUTPUT_ERRORS = {
    "ValidationError",
    "JSONDecodeError",
}


class LlmErrorKind(str, Enum):
    """The kind of error raised by the model provider."""

    PERMANENT_ERROR = "permanent_error"
    TRANSIENT_ERROR = "transient_error"
    UNUSABLE_OUTPUT = "unusable_output"


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int):
        return status
    return None


def _error_codes(error: Exception) -> set[str]:
    """
    Collect the code and type strings carried by a provider exception.
    Args:
        error: The exception raised by the model provider.
    Returns:
        The code and type values found on the exception and in its response body.
    """
    codes: set[str] = set()
    for attr in ("code", "type"):
        value = getattr(error, attr, None)
        if isinstance(value, str) and value:
            codes.add(value)
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict):
            for key in ("code", "type"):
                nested = inner.get(key)
                if isinstance(nested, str) and nested:
                    codes.add(nested)
    return codes


def _is_insufficient_quota(error: Exception) -> bool:
    if _error_codes(error) & _QUOTA_ERROR_CODES:
        return True
    return "insufficient_quota" in str(error)


def _llm_error_kind(error: Exception) -> str | None:
    name = type(error).__name__
    if name in _PERMANENT_ERRORS or _is_insufficient_quota(error):
        return LlmErrorKind.PERMANENT_ERROR
    if name in _UNUSABLE_OUTPUT_ERRORS:
        return LlmErrorKind.UNUSABLE_OUTPUT
    status = _status_code(error)
    if isinstance(status, int) and status in _PERMANENT_STATUS:
        return LlmErrorKind.PERMANENT_ERROR
    if isinstance(status, int) and (status in _TRANSIENT_STATUS or status >= 500):
        return LlmErrorKind.TRANSIENT_ERROR
    if name in _TRANSIENT_ERRORS:
        return LlmErrorKind.TRANSIENT_ERROR
    return None


def is_transient_llm_error(error: Exception) -> bool:
    """
    Decide whether sending the same model request again can succeed.
    Args:
        error: The exception raised by the model provider.
    Returns:
        True for timeouts, connection failures, rate limits, and server errors.
    """
    return _llm_error_kind(error) == LlmErrorKind.TRANSIENT_ERROR


def is_permanent_llm_error(error: Exception) -> bool:
    """
    Decide whether the same model request will fail again.
    Args:
        error: The exception raised by the model provider.
    Returns:
        True for authentication, billing, invalid requests, and policy rejections.
    """
    return _llm_error_kind(error) == LlmErrorKind.PERMANENT_ERROR


def is_unusable_llm_output(error: Exception) -> bool:
    """
    Decide whether the model responded with output that cannot be used.
    Args:
        error: The exception raised while reading the model output.
    Returns:
        True for invalid JSON and schema validation failures.
    """
    return _llm_error_kind(error) == LlmErrorKind.UNUSABLE_OUTPUT


class FinishReason(str, Enum):
    """Why the model stopped generating."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    FUNCTION_CALL = "function_call"

    def requests_tools(self) -> bool:
        """
        Report whether the model stopped to call a tool.
        Returns:
            True for tool_calls and function_call.
        """
        return self in {FinishReason.TOOL_CALLS, FinishReason.FUNCTION_CALL}


@dataclass
class LlmToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: str


@dataclass
class LlmUsage:
    """Token counts for one model call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMOutput:
    """Model output used by the agent loop."""

    finish_reason: FinishReason | None
    content: str | None
    tool_calls: list[LlmToolCall]
    reasoning_content: str | None
    thinking_blocks: list[dict[str, Any]] | None
    usage: LlmUsage | None = None
    structured: Any | None = None

    def requests_tool_calls(self) -> bool:
        """
        Report whether this output asks the harness to run tool calls.
        Returns:
            True when the finish reason asks for tools and at least one call is present.
        """
        if self.finish_reason is None or not self.tool_calls:
            return False
        return self.finish_reason.requests_tools()


def assemble_llm_stream(pieces: Iterable[LLMOutput]) -> tuple[LLMOutput, list[str]]:
    """
    Fold streamed model pieces into one output.
    Args:
        pieces: LLMOutput fragments in arrival order.
    Returns:
        The assembled output and the non-empty text fragments, in order.
    Raises:
        TypeError: A fragment is not an LLMOutput.
    """
    text: list[str] = []
    calls: list[LlmToolCall] = []
    finish_reason = None
    usage = None
    reasoning = None
    thinking = None
    structured = None
    for piece in pieces:
        if not isinstance(piece, LLMOutput):
            raise TypeError("infer did not return a model output")
        if piece.content:
            text.append(piece.content)
        _merge_tool_calls(calls, piece.tool_calls)
        if piece.finish_reason is not None:
            finish_reason = piece.finish_reason
        if piece.usage is not None:
            usage = piece.usage
        if piece.reasoning_content:
            reasoning = piece.reasoning_content
        if piece.thinking_blocks:
            thinking = piece.thinking_blocks
        if piece.structured is not None:
            structured = piece.structured
    output = LLMOutput(
        finish_reason,
        "".join(text) or None,
        calls,
        reasoning,
        thinking,
        usage,
        structured,
    )
    return output, text


def _merge_tool_calls(calls: list[LlmToolCall], deltas: list[LlmToolCall]) -> None:
    for delta in deltas:
        if delta.id or not calls:
            calls.append(LlmToolCall(delta.id, delta.name, delta.arguments))
            continue
        current = calls[-1]
        if delta.name and not current.name:
            current.name = delta.name
        current.arguments += delta.arguments
