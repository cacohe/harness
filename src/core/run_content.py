from src.core.artifacts import ArtifactExternalizer, ArtifactRef


async def externalize_final_text(
    text: str, max_inline_bytes: int, externalizer: ArtifactExternalizer
) -> tuple[str, tuple[ArtifactRef, ...]]:
    if len(text.encode("utf-8")) <= max_inline_bytes:
        return text, ()
    content = await externalizer.externalize(text, "text/plain")
    inline = content.inline_content
    if inline is not None:
        value = inline.value
        if type(value) is not str or len(value.encode("utf-8")) > max_inline_bytes:
            raise ValueError("Final inline result exceeds the Run limit")
        return value, ()
    summary = content.summary
    if summary is None or len(summary.encode("utf-8")) > max_inline_bytes:
        raise ValueError("Final Artifact summary exceeds the Run limit")
    return summary, content.artifacts
