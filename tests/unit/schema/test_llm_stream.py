import asyncio

import pytest
from src.schema.llm import (
    FinishReason,
    LLMOutput,
    LlmToolCall,
    LlmUsage,
    assemble_llm_async_stream,
    assemble_llm_stream,
)


def _piece(
    content=None,
    finish_reason=None,
    tool_calls=None,
    reasoning_content=None,
    thinking_blocks=None,
    usage=None,
    structured=None,
):
    return LLMOutput(
        finish_reason,
        content,
        tool_calls or [],
        reasoning_content,
        thinking_blocks,
        usage,
        structured,
    )


def test_assemble_llm_stream_joins_text_and_keeps_the_fragments():
    output, parts = assemble_llm_stream(
        [_piece("hel"), _piece("lo", finish_reason=FinishReason.STOP)]
    )

    assert output.content == "hello"
    assert output.finish_reason is FinishReason.STOP
    assert parts == ["hel", "lo"]


def test_assemble_llm_stream_continues_a_tool_call_without_an_id():
    output, parts = assemble_llm_stream(
        [
            _piece(tool_calls=[LlmToolCall("call_1", "", "")]),
            _piece(tool_calls=[LlmToolCall("", "shell", '{"x":')]),
            _piece(
                tool_calls=[LlmToolCall("", "", "1}")],
                finish_reason=FinishReason.TOOL_CALLS,
            ),
        ]
    )

    assert parts == []
    assert [(call.id, call.name, call.arguments) for call in output.tool_calls] == [
        ("call_1", "shell", '{"x":1}')
    ]


def test_assemble_llm_stream_starts_a_new_tool_call_when_the_delta_has_an_id():
    output, _parts = assemble_llm_stream(
        [
            _piece(tool_calls=[LlmToolCall("call_1", "shell", "{}")]),
            _piece(tool_calls=[LlmToolCall("call_2", "read", '{"path": "a"}')]),
        ]
    )

    assert [(call.id, call.name) for call in output.tool_calls] == [
        ("call_1", "shell"),
        ("call_2", "read"),
    ]


def test_assemble_llm_stream_keeps_the_latest_present_fields():
    usage = LlmUsage(1, 2, 3)
    output, _parts = assemble_llm_stream(
        [
            _piece(reasoning_content="first", thinking_blocks=[{"type": "thinking"}]),
            _piece(
                finish_reason=FinishReason.STOP,
                reasoning_content="second",
                thinking_blocks=[{"type": "thinking", "thinking": "done"}],
                usage=usage,
                structured={"ok": True},
            ),
        ]
    )

    assert output.finish_reason is FinishReason.STOP
    assert output.reasoning_content == "second"
    assert output.thinking_blocks == [{"type": "thinking", "thinking": "done"}]
    assert output.usage is usage
    assert output.structured == {"ok": True}


def test_assemble_llm_stream_returns_an_empty_output_for_no_pieces():
    output, parts = assemble_llm_stream([])

    assert parts == []
    assert output.content is None
    assert output.tool_calls == []
    assert output.finish_reason is None
    assert output.structured is None


def test_assemble_llm_stream_rejects_a_non_output_piece():
    with pytest.raises(TypeError, match="infer did not return a model output"):
        assemble_llm_stream(["hel"])


def test_async_stream_assembly_matches_sync_stream_for_text_tools_and_usage():
    usage = LlmUsage(2, 3, 5)
    pieces = [
        _piece("hel", tool_calls=[LlmToolCall("call_1", "", '{"x":')]),
        _piece("lo", tool_calls=[LlmToolCall("", "shell", "1}")]),
        _piece(finish_reason=FinishReason.TOOL_CALLS, usage=usage),
    ]

    async def async_pieces():
        for piece in pieces:
            yield piece

    async_output, async_parts = asyncio.run(assemble_llm_async_stream(async_pieces()))
    sync_output, sync_parts = assemble_llm_stream(pieces)

    assert async_output == sync_output
    assert async_parts == sync_parts == ["hel", "lo"]


def test_async_stream_assembly_accepts_a_sync_iterator_without_blocking_contract_changes():
    pieces = iter([_piece("done", finish_reason=FinishReason.STOP)])

    output, parts = asyncio.run(assemble_llm_async_stream(pieces))

    assert output.content == "done"
    assert parts == ["done"]


def test_async_stream_assembly_folds_a_reused_mutable_chunk_on_arrival():
    def reused_chunks():
        piece = _piece("hello ")
        yield piece
        piece.content = "world"
        piece.finish_reason = FinishReason.STOP
        yield piece

    output, parts = asyncio.run(assemble_llm_async_stream(reused_chunks()))

    assert output.content == "hello world"
    assert parts == ["hello ", "world"]


def test_async_stream_stops_consuming_when_byte_budget_is_exceeded():
    consumed = []
    closed = []

    async def pieces():
        try:
            for index in range(100):
                consumed.append(index)
                yield _piece("x" * 50)
        finally:
            closed.append(True)

    with pytest.raises(OverflowError, match="max_bytes"):
        asyncio.run(assemble_llm_async_stream(pieces(), max_bytes=120))

    assert len(consumed) < 100
    assert closed == [True]
