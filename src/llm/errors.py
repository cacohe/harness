_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_TRANSIENT_ERRORS = {
    "Timeout",
    "TimeoutError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
}


def is_transient_llm_error(error: Exception) -> bool:
    """
    Decide whether a model call can be retried.
    Args:
        error: The exception raised by the model provider.
    Returns:
        True for timeouts, rate limits, and server errors.
    """
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int) and (status in _TRANSIENT_STATUS or status >= 500):
        return True
    return type(error).__name__ in _TRANSIENT_ERRORS
