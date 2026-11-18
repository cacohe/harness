import asyncio
from types import SimpleNamespace

import pytest
from src.adapters.model.litellm import LiteLLMModelGateway, ModelProviderError
from src.core.model import ModelMessage, ModelRequest, ModelRole, ModelToolCall
from src.core.tools import ToolSpec


def test_complete_preserves_wire_and_returns_core_types() -> None:
    captured = {}

    async def completion(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content="ok", tool_calls=[])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
        )

    gateway = LiteLLMModelGateway("provider/model", max_output_tokens=128, completion=completion)
    response = asyncio.run(gateway.complete(ModelRequest((ModelMessage(ModelRole.USER, "hello"),))))
    assert response.text == "ok"
    assert response.usage is not None and response.usage.input_tokens == 2
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["max_tokens"] == 128


def test_provider_error_is_sanitized_and_secret_not_in_repr() -> None:
    async def completion(**_kwargs):
        raise RuntimeError("secret provider detail")

    gateway = LiteLLMModelGateway(
        "provider/model", max_output_tokens=128, api_key="top-secret", completion=completion
    )
    assert "top-secret" not in repr(gateway)
    try:
        asyncio.run(gateway.complete(ModelRequest((ModelMessage(ModelRole.USER, "x"),))))
    except ModelProviderError as error:
        assert str(error) == "Model request failed"
        assert error.__cause__ is None
    else:
        raise AssertionError("provider failure was not mapped")


def test_all_roles_tool_batch_and_schema_preserve_wire() -> None:
    captured = {}

    async def completion(**kwargs):
        captured.update(kwargs)
        call = {"id": "next", "function": {"name": "lookup", "arguments": '{"q":1}'}}
        return {
            "choices": [{"message": {"content": None, "tool_calls": [call]}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }

    request = ModelRequest(
        (
            ModelMessage(ModelRole.SYSTEM, "system"),
            ModelMessage(ModelRole.USER, "question"),
            ModelMessage(ModelRole.ASSISTANT, tool_calls=(ModelToolCall("old", "lookup", "{}"),)),
            ModelMessage(ModelRole.TOOL, "result", tool_call_id="old"),
        ),
        (ToolSpec("lookup", "lookup", '{"type":"object"}'),),
    )
    response = asyncio.run(
        LiteLLMModelGateway("p/m", max_output_tokens=128, completion=completion).complete(request)
    )
    assert captured["messages"][2]["tool_calls"][0]["id"] == "old"
    assert captured["messages"][3]["tool_call_id"] == "old"
    assert captured["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert response.tool_calls[0].arguments_json == '{"q":1}'


class _Stream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iterator = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration


def test_stream_assembles_fragments_and_usage() -> None:
    async def completion(**_kwargs):
        return _Stream(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "content": "a",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c",
                                        "function": {"name": "t", "arguments": "{"},
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "content": "b",
                                "tool_calls": [{"index": 0, "function": {"arguments": "}"}}],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
            ]
        )

    response = asyncio.run(
        LiteLLMModelGateway(
            "p/m", max_output_tokens=128, stream=True, completion=completion
        ).complete(ModelRequest((ModelMessage(ModelRole.USER, "x"),)))
    )
    assert response.text == "ab"
    assert response.tool_calls == (ModelToolCall("c", "t", "{}"),)
    assert response.usage is not None and response.usage.output_tokens == 2


def test_stream_rejects_duplicate_index_and_byte_limit() -> None:
    async def out_of_order(**_kwargs):
        return _Stream(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "a",
                                        "function": {"name": "t", "arguments": "{}"},
                                    },
                                    {"index": 0, "function": {"arguments": " "}},
                                ]
                            }
                        }
                    ]
                },
            ]
        )

    request = ModelRequest((ModelMessage(ModelRole.USER, "x"),))
    with pytest.raises(ModelProviderError):
        asyncio.run(
            LiteLLMModelGateway(
                "p/m", max_output_tokens=128, stream=True, completion=out_of_order
            ).complete(request)
        )

    async def oversized(**_kwargs):
        return _Stream([{"choices": [{"delta": {"content": "12345"}}]}])

    with pytest.raises(ModelProviderError):
        asyncio.run(
            LiteLLMModelGateway(
                "p/m",
                max_output_tokens=128,
                stream=True,
                max_response_bytes=4,
                completion=oversized,
            ).complete(request)
        )


def test_timeout_and_caller_cancellation_are_distinct() -> None:
    async def slow(**_kwargs):
        await asyncio.sleep(10)

    request = ModelRequest((ModelMessage(ModelRole.USER, "x"),))
    with pytest.raises(TimeoutError, match="Model request timed out"):
        asyncio.run(
            LiteLLMModelGateway(
                "p/m", max_output_tokens=128, timeout_seconds=0.01, completion=slow
            ).complete(request)
        )

    async def cancel():
        task = asyncio.create_task(
            LiteLLMModelGateway("p/m", max_output_tokens=128, completion=slow).complete(request)
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())


def test_parallel_tool_fragments_continue_by_index_across_chunks() -> None:
    async def completion(**_kwargs):
        return _Stream(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "a",
                                        "function": {"name": "one", "arguments": "{"},
                                    },
                                    {
                                        "index": 1,
                                        "id": "b",
                                        "function": {"name": "two", "arguments": "{"},
                                    },
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": "}"}},
                                    {"index": 1, "function": {"arguments": "}"}},
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ]
        )

    request = ModelRequest((ModelMessage(ModelRole.USER, "x"),))
    response = asyncio.run(
        LiteLLMModelGateway(
            "p/m", max_output_tokens=128, stream=True, completion=completion
        ).complete(request)
    )
    assert [call.arguments_json for call in response.tool_calls] == ["{}", "{}"]


def test_stream_requires_finish_and_closes_provider_iterator() -> None:
    closed = False

    class Stream(_Stream):
        async def aclose(self):
            nonlocal closed
            closed = True

    async def completion(**_kwargs):
        return Stream([{"choices": [{"delta": {"content": "unfinished"}}]}])

    request = ModelRequest((ModelMessage(ModelRole.USER, "x"),))
    with pytest.raises(ModelProviderError):
        asyncio.run(
            LiteLLMModelGateway(
                "p/m", max_output_tokens=128, stream=True, completion=completion
            ).complete(request)
        )
    assert closed


def test_stream_close_has_a_finite_cleanup_deadline() -> None:
    close_started = asyncio.Event()

    class Stream(_Stream):
        async def aclose(self):
            close_started.set()
            await asyncio.Event().wait()

    async def completion(**_kwargs):
        return Stream([{"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}])

    async def scenario():
        gateway = LiteLLMModelGateway(
            "p/m",
            max_output_tokens=128,
            stream=True,
            timeout_seconds=1,
            cleanup_timeout_seconds=0.01,
            completion=completion,
        )
        request = ModelRequest((ModelMessage(ModelRole.USER, "x"),))
        with pytest.raises(ModelProviderError, match="Model request failed"):
            await asyncio.wait_for(gateway.complete(request), 0.2)
        assert close_started.is_set()

    asyncio.run(scenario())


def test_stream_cancellation_preserves_cancelled_error_after_bounded_close() -> None:
    iteration_started = asyncio.Event()
    close_started = asyncio.Event()

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            iteration_started.set()
            await asyncio.Event().wait()

        async def aclose(self):
            close_started.set()
            await asyncio.Event().wait()

    async def completion(**_kwargs):
        return Stream()

    async def scenario():
        gateway = LiteLLMModelGateway(
            "p/m",
            max_output_tokens=128,
            stream=True,
            timeout_seconds=1,
            cleanup_timeout_seconds=0.01,
            completion=completion,
        )
        request = ModelRequest((ModelMessage(ModelRole.USER, "x"),))
        task = asyncio.create_task(gateway.complete(request))
        await iteration_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 0.2)
        assert close_started.is_set()

    asyncio.run(scenario())
