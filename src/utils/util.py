def decode_output(data: bytes | str | None) -> str:
    """
    Decode the data to a string.
    Args:
        data: The data to decode.
    Returns:
        The decoded data.
    """
    if not data:
        return ""
    if isinstance(data, str):
        return data
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("gbk", errors="replace")
