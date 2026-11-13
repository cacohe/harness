from pydantic import BaseModel
from src.llm.litellm import LiteLLMGateway
from src.schema.context import Context, Message
from src.schema.llm import FinishReason, LlmToolCall


def _context() -> Context:
    return Context(
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Weather in Beijing?"),
        ],
        tools=[],
    )


def test_infer_returns_text_tool_calls_reasoning_and_usage(monkeypatch):
    class Function:
        name = "get_weather"
        arguments = '{"city": "Beijing"}'

    class Call:
        id = "call_1"
        function = Function()

    class Block:
        def model_dump(self):
            return {"type": "thinking", "thinking": "look up the weather", "signature": "sig"}

    class Message:
        content = "Checking the weather."
        tool_calls = [Call()]
        reasoning_content = "look up the weather"
        thinking_blocks = [Block()]

    class Choice:
        message = Message()
        finish_reason = "tool_calls"

    class Usage:
        prompt_tokens = 11
        completion_tokens = 7
        total_tokens = 18

    class Response:
        choices = [Choice()]
        usage = Usage()

    def fake_completion(**kwargs):
        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["stream"] is False
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][1]["content"] == "Weather in Beijing?"
        return Response()

    monkeypatch.setattr("src.llm.litellm.litellm.completion", fake_completion)

    output = LiteLLMGateway("openai/gpt-4o").infer(_context(), stream=False)

    assert output.finish_reason is FinishReason.TOOL_CALLS
    assert output.content == "Checking the weather."
    assert [(call.id, call.name, call.arguments) for call in output.tool_calls] == [
        ("call_1", "get_weather", '{"city": "Beijing"}')
    ]
    assert output.reasoning_content == "look up the weather"
    assert output.thinking_blocks == [
        {"type": "thinking", "thinking": "look up the weather", "signature": "sig"}
    ]
    assert output.usage.prompt_tokens == 11
    assert output.usage.completion_tokens == 7
    assert output.usage.total_tokens == 18


def test_infer_stream_yields_text_and_usage(monkeypatch):
    class Delta:
        content = "Hi"
        tool_calls = None
        reasoning_content = None
        thinking_blocks = None

    class Choice:
        delta = Delta()
        finish_reason = "stop"

    class Usage:
        prompt_tokens = 1
        completion_tokens = 1
        total_tokens = 2

    class Chunk:
        def __init__(self, choices, usage):
            self.choices = choices
            self.usage = usage

    def fake_completion(**kwargs):
        assert kwargs["stream"] is True
        assert kwargs["stream_options"]["include_usage"] is True
        return [Chunk([Choice()], None), Chunk([], Usage())]

    monkeypatch.setattr("src.llm.litellm.litellm.completion", fake_completion)

    pieces = list(LiteLLMGateway("m").infer(_context(), stream=True))

    assert pieces[0].content == "Hi"
    assert pieces[0].finish_reason is FinishReason.STOP
    assert pieces[0].structured is None
    assert pieces[1].usage.total_tokens == 2


def test_infer_returns_a_requested_structure(monkeypatch):
    class Weather(BaseModel):
        city: str
        celsius: float

    class Message:
        content = '{"city": "Beijing", "celsius": 18}'
        tool_calls = None
        reasoning_content = None
        thinking_blocks = None

    class Choice:
        message = Message()
        finish_reason = "stop"

    class Usage:
        prompt_tokens = 2
        completion_tokens = 3
        total_tokens = 5

    class Response:
        choices = [Choice()]
        usage = Usage()

    def fake_completion(**kwargs):
        assert kwargs["response_format"] is Weather
        assert kwargs["stream"] is False
        return Response()

    monkeypatch.setattr("src.llm.litellm.litellm.completion", fake_completion)

    output = LiteLLMGateway("m").infer(_context(), stream=False, response_format=Weather)

    assert output.content == '{"city": "Beijing", "celsius": 18}'
    assert output.structured == Weather(city="Beijing", celsius=18)
    assert output.usage.total_tokens == 5


def test_deepseek_structured_output_uses_json_object(monkeypatch):
    class Weather(BaseModel):
        city: str

    class Message:
        content = '{"city": "Beijing"}'
        tool_calls = None
        reasoning_content = None
        thinking_blocks = None

    class Choice:
        message = Message()
        finish_reason = "stop"

    class Response:
        choices = [Choice()]
        usage = None

    def fake_completion(**kwargs):
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert "json" in kwargs["messages"][0]["content"]
        assert '"city": "example"' in kwargs["messages"][0]["content"]
        return Response()

    monkeypatch.setattr("src.llm.litellm.litellm.completion", fake_completion)

    output = LiteLLMGateway("deepseek/deepseek-flash", api_base="https://api.deepseek.com").infer(
        _context(), stream=False, response_format=Weather
    )

    assert output.structured == Weather(city="Beijing")


def test_infer_replays_tool_calls_and_reasoning(monkeypatch):
    seen = {}

    def fake_completion(**kwargs):
        seen["messages"] = kwargs["messages"]

        class MessageOut:
            content = "done"
            tool_calls = None
            reasoning_content = None
            thinking_blocks = None

        class Choice:
            message = MessageOut()
            finish_reason = "stop"

        class Response:
            choices = [Choice()]
            usage = None

        return Response()

    monkeypatch.setattr("src.llm.litellm.litellm.completion", fake_completion)
    context = Context(
        messages=[
            Message(role="user", content="Weather in Beijing?"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[LlmToolCall("call_1", "get_weather", '{"city": "Beijing"}')],
                reasoning_content="look up the weather",
                thinking_blocks=[{"type": "thinking", "thinking": "look up", "signature": "sig"}],
            ),
            Message(role="tool", content="sunny", tool_call_id="call_1", name="get_weather"),
        ],
        tools=[],
    )

    LiteLLMGateway("m").infer(context, stream=False)

    assistant = seen["messages"][1]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"city": "Beijing"}'
    assert assistant["reasoning_content"] == "look up the weather"
    assert assistant["thinking_blocks"][0]["signature"] == "sig"
    assert seen["messages"][2]["tool_call_id"] == "call_1"
    assert seen["messages"][2]["content"] == "sunny"


def test_infer_preserves_raw_provider_message_fields(monkeypatch):
    seen = {}

    def fake_completion(**kwargs):
        seen["messages"] = kwargs["messages"]

        class MessageOut:
            content = "done"
            tool_calls = None
            reasoning_content = None
            thinking_blocks = None

        class Choice:
            message = MessageOut()
            finish_reason = "stop"

        class Response:
            choices = [Choice()]
            usage = None

        return Response()

    monkeypatch.setattr("src.llm.litellm.litellm.completion", fake_completion)
    raw_message = {
        "role": "user",
        "content": [{"type": "text", "text": "Inspect this"}],
        "cache_control": {"type": "ephemeral"},
        "provider_extension": {"trace": "keep-me"},
    }

    LiteLLMGateway("m").infer(Context(messages=[raw_message]), stream=False)

    assert seen["messages"] == [raw_message]
