from src.core.artifacts import ArtifactExternalizer, ArtifactRef, ExternalizedContent
from src.core.tools import ToolResult


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


async def externalize_tool_result(
    result: ToolResult, max_inline_bytes: int, externalizer: ArtifactExternalizer
) -> ExternalizedContent:
    content = ExternalizedContent(result.inline_content, result.artifacts, result.summary)
    inline = content.inline_content
    if inline is not None and inline.size_bytes > max_inline_bytes:
        content = await externalizer.externalize(inline.value, inline.media_type)
    inline = content.inline_content
    if inline is not None and inline.size_bytes > max_inline_bytes:
        raise ValueError
    if content.summary is not None and len(content.summary.encode("utf-8")) > max_inline_bytes:
        raise ValueError
    return content
