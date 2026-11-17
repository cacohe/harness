import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import litellm
from pydantic import BaseModel

from src.schema.context import Context
from src.schema.llm import ROLE_TOOL, FinishReason, LLMOutput, LlmToolCall, LlmUsage
from src.schema.tool import Tool
from src.utils.async_utils import iterate_maybe_async


def _is_deepseek(model: str, api_base: str | None) -> bool:
    name = model.lower()
    provider, _, model_id = name.partition("/")
    if (
        provider == "deepseek"
        or provider.startswith("deepseek-")
        or model_id.startswith("deepseek")
    ):
        return True
    return "api.deepseek.com" in (api_base or "").lower()


def _schema_of(response_format: type[BaseModel] | dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return response_format.model_json_schema()
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        schema = response_format.get("json_schema", {}).get("schema")
        if isinstance(schema, dict):
            return schema
    return None


def _completion_request(
    model: str,
    api_base: str | None,
    messages: list[dict[str, Any]],
    response_format: type[BaseModel] | dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], type[BaseModel] | dict[str, Any] | None]:
    schema = None if response_format is None else _schema_of(response_format)
    if schema is None or not _is_deepseek(model, api_base):
        return messages, response_format
    return _with_json_example(messages, schema), {"type": "json_object"}


def _with_json_example(
    messages: list[dict[str, Any]], schema: dict[str, Any]
) -> list[dict[str, Any]]:
    instruction = (
        "Respond in json. Use a json object with the same shape as this example:\n"
        + json.dumps(_json_example(schema), ensure_ascii=False, indent=2)
    )
    prepared = [dict(message) for message in messages]
    for message in prepared:
        if message.get("role") != "system":
            continue
        content = message.get("content") or ""
        message["content"] = f"{content}\n\n{instruction}" if content else instruction
        return prepared
    prepared.insert(0, {"role": "system", "content": instruction})
    return prepared


def _json_example(schema: dict[str, Any], defs: dict[str, Any] | None = None) -> Any:
    definitions = schema.get("$defs") if isinstance(schema.get("$defs"), dict) else defs
    definitions = definitions or {}
    if "$ref" in schema:
        name = str(schema["$ref"]).rsplit("/", 1)[-1]
        return _json_example(definitions.get(name, {}), definitions)
    if "anyOf" in schema:
        choices = schema["anyOf"]
        selected = next((item for item in choices if item.get("type") != "null"), choices[0])
        return _json_example(selected, definitions)
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    kind = schema.get("type", "string")
    if kind == "object":
        properties = schema.get("properties") or {}
        return {key: _json_example(value, definitions) for key, value in properties.items()}
    if kind == "array":
        return [_json_example(schema.get("items") or {"type": "string"}, definitions)]
    if kind == "integer":
        return 0
    if kind == "number":
        return 0
    if kind == "boolean":
        return False
    return "example"


class LiteLLMGateway:
    """Call an LLM through LiteLLM and return the fields an agent loop consumes."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        **options,
    ):
        """
        Create a gateway for one model.
        Args:
            model: LiteLLM model name, such as "openai/gpt-4o" or "anthropic/claude-sonnet-4-5".
            api_key: Provider API key. Omit it to use the provider environment variable.
            api_base: Provider base URL. Omit it to use the provider default.
            **options: Extra arguments forwarded to every LiteLLM completion call.
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.options = options

    async def ainfer(
        self,
        context: Context,
        stream: bool = True,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
    ) -> LLMOutput | AsyncIterator[LLMOutput]:
        """Run one LLM call through LiteLLM's native asynchronous API."""
        messages, options = self._request(context, stream, response_format)
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            stream=stream,
            **options,
        )
        if stream:
            return self._astream(response, response_format)
        return self._complete_output(response, response_format)

    def infer(
        self,
        context: Context,
        stream: bool = True,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
    ) -> LLMOutput | Iterator[LLMOutput]:
        """
        Run one LLM call.
        Args:
            context: System text, user text, prior assistant text, and available tools.
            stream: True yields each streamed piece. False returns the finished output.
            response_format: Pydantic model or JSON schema dict the assistant text must follow.
                LiteLLM sends a Pydantic model as json_schema. DeepSeek Chat Completions
                receives json_object instead, the prompt includes a json example, and
                thinking is disabled so the JSON is written to content.
                The parsed value is output.structured.
        Returns:
            The finish reason, assistant text, tool calls, reasoning, token usage, and,
            when response_format is set, the parsed structure.
            A stream yields the same fields as each piece arrives. The parsed structure
            arrives on the piece that carries the finish reason.
        """
        messages, options = self._request(context, stream, response_format)

        response = litellm.completion(
            model=self.model,
            messages=messages,
            stream=stream,
            **options,
        )
        if stream:
            return self._stream(response, response_format)
        return self._complete_output(response, response_format)

    def _request(
        self,
        context: Context,
        stream: bool,
        response_format: type[BaseModel] | dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        messages, wire_format = _completion_request(
            self.model, self.api_base, self._messages(context), response_format
        )
        options = self._call_options()
        tools = self._tools(getattr(context, "tools", None))
        if tools:
            options["tools"] = tools
            options.setdefault("tool_choice", "auto")
        if wire_format is not None:
            options["response_format"] = wire_format
            if _is_deepseek(self.model, self.api_base) and wire_format == {"type": "json_object"}:
                extra_body = dict(options.get("extra_body") or {})
                extra_body["thinking"] = {"type": "disabled"}
                options["extra_body"] = extra_body
        if stream:
            stream_options = dict(options.get("stream_options") or {})
            stream_options.setdefault("include_usage", True)
            options["stream_options"] = stream_options
        return messages, options

    def _complete_output(
        self,
        response: Any,
        response_format: type[BaseModel] | dict[str, Any] | None,
    ) -> LLMOutput:
        choice = response.choices[0]
        output = self._output(
            choice.message,
            self._finish_reason(choice.finish_reason),
            self._usage(getattr(response, "usage", None)),
        )
        output.structured = self._structured(output.content, response_format)
        return output

    def _messages(self, context: Context) -> list[dict[str, Any]]:
        """
        Build the chat messages for one call.
        Args:
            context: Context for this call.
        Returns:
            The messages in call order, including tool calls and reasoning.
        """
        return [self._message(item) for item in context.messages]

    @staticmethod
    def _message(message) -> dict[str, Any]:
        """
        Convert one transcript item into a LiteLLM message.
        Args:
            message: A message dict, or an object with role and content.
        Returns:
            A chat message. Assistant items keep tool calls, reasoning, and thinking blocks.
            Tool items keep the tool call id.
        """
        if isinstance(message, dict):
            return message
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            payload["tool_calls"] = [LiteLLMGateway._tool_call_payload(call) for call in tool_calls]
        for field in ("reasoning_content", "thinking_blocks", "tool_call_id"):
            value = getattr(message, field, None)
            if value:
                payload[field] = value
        name = getattr(message, "name", None)
        if name and message.role == ROLE_TOOL:
            payload["name"] = name
        return payload

    @staticmethod
    def _tool_call_payload(call) -> dict[str, Any]:
        arguments = (
            call.arguments if isinstance(call.arguments, str) else json.dumps(call.arguments)
        )
        return {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": arguments},
        }

    def _tools(self, tools) -> list[dict[str, Any]] | None:
        """
        Convert harness tools into the LiteLLM tools parameter.
        Args:
            tools: Tool objects, or tool definitions already in LiteLLM form.
        Returns:
            Tool definitions for completion, or None when no tools are available.
        """
        if not tools or isinstance(tools, str):
            return None

        converted = []
        for tool in tools:
            if isinstance(tool, dict):
                converted.append(tool)
                continue
            if not isinstance(tool, Tool):
                continue
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema or {"type": "object", "properties": {}},
                    },
                }
            )
        return converted or None

    def _call_options(self) -> dict[str, Any]:
        """
        Build the provider options for one completion call.
        Returns:
            Options passed through to LiteLLM, including api_key and api_base when set.
        """
        options = dict(self.options)
        if self.api_key is not None:
            options["api_key"] = self.api_key
        if self.api_base is not None:
            options["api_base"] = self.api_base
        return options

    def _stream(
        self,
        response,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
    ) -> Iterator[LLMOutput]:
        """
        Yield each streamed piece as an LlmInferOutput.
        Args:
            response: Streaming completion returned by LiteLLM.
            response_format: Structure used to parse the assembled assistant text.
        Returns:
            An iterator of output pieces. finish_reason is set on the piece that carries it.
            That piece also carries the parsed structure when response_format is set.
        """
        text_parts: list[str] = []
        for chunk in response:
            usage = self._usage(getattr(chunk, "usage", None))
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                if usage is not None:
                    yield LLMOutput(None, None, [], None, None, usage=usage)
                continue
            choice = choices[0]
            output = self._output(
                getattr(choice, "delta", None),
                self._finish_reason(getattr(choice, "finish_reason", None)),
                usage,
            )
            if output.content:
                text_parts.append(output.content)
            if response_format is not None and output.finish_reason is not None:
                output.structured = self._structured("".join(text_parts), response_format)
            if self._has_payload(output):
                yield output

    async def _astream(
        self,
        response: Any,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
    ) -> AsyncIterator[LLMOutput]:
        """Yield each streamed piece from a native async LiteLLM response."""
        text_parts: list[str] = []
        async for chunk in iterate_maybe_async(response):
            usage = self._usage(getattr(chunk, "usage", None))
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                if usage is not None:
                    yield LLMOutput(None, None, [], None, None, usage=usage)
                continue
            choice = choices[0]
            output = self._output(
                getattr(choice, "delta", None),
                self._finish_reason(getattr(choice, "finish_reason", None)),
                usage,
            )
            if output.content:
                text_parts.append(output.content)
            if response_format is not None and output.finish_reason is not None:
                output.structured = self._structured("".join(text_parts), response_format)
            if self._has_payload(output):
                yield output

    @staticmethod
    def _structured(content: str | None, response_format: type[BaseModel] | dict[str, Any] | None):
        """
        Parse assistant text into the caller-supplied structure.
        Args:
            content: Assistant text, expected to be JSON when a format is set.
            response_format: Pydantic model or JSON schema dict passed to LiteLLM.
        Returns:
            A model instance or a dict, or None when no format or text is present.
        """
        if response_format is None or not content:
            return None
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return response_format.model_validate_json(content)
        return json.loads(content)

    def _output(
        self,
        message,
        finish_reason: FinishReason | None,
        usage: LlmUsage | None = None,
    ) -> LLMOutput:
        """
        Read the agent fields from one LiteLLM message or stream delta.
        Args:
            message: Completion message or streaming delta.
            finish_reason: Why the model stopped, when this piece includes it.
            usage: Token counts for this call, when the response includes them.
        Returns:
            The finish reason, text, tool calls, reasoning, and token usage from that message.
        """
        if message is None:
            return LLMOutput(finish_reason, None, [], None, None, usage)

        content = getattr(message, "content", None)
        if not isinstance(content, str):
            content = None
        return LLMOutput(
            finish_reason=finish_reason,
            content=content,
            tool_calls=self._tool_calls(message),
            reasoning_content=getattr(message, "reasoning_content", None),
            thinking_blocks=self._thinking_blocks(message),
            usage=usage,
        )

    @staticmethod
    def _finish_reason(value: str | None) -> FinishReason | None:
        """
        Convert a LiteLLM finish reason into FinishReason.
        Args:
            value: Finish reason returned by LiteLLM.
        Returns:
            The matching FinishReason, or None when this piece has no finish reason.
        Raises:
            ValueError: If LiteLLM returns a finish reason outside FinishReason.
        """
        if not value:
            return None
        return FinishReason(value)

    @staticmethod
    def _usage(usage) -> LlmUsage | None:
        """
        Read token counts from a LiteLLM usage object.
        Args:
            usage: Usage on a completion response or stream chunk.
        Returns:
            Prompt, completion, and total token counts, or None when usage is absent.
        """
        if usage is None:
            return None
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
        if prompt_tokens is None and completion_tokens is None and total_tokens is None:
            return None
        return LlmUsage(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=total_tokens or 0,
        )

    @staticmethod
    def _tool_calls(message) -> list[LlmToolCall]:
        """
        Read tool calls from a LiteLLM message or stream delta.
        Args:
            message: Completion message or streaming delta.
        Returns:
            Tool calls present on this message. Arguments may be a partial JSON string while streaming.
        """
        calls = getattr(message, "tool_calls", None) or []
        parsed = []
        for call in calls:
            function = getattr(call, "function", None)
            if isinstance(call, dict):
                function = call.get("function") or {}
                name = function.get("name") if isinstance(function, dict) else None
                arguments = function.get("arguments") if isinstance(function, dict) else None
                call_id = call.get("id")
            else:
                name = getattr(function, "name", None)
                arguments = getattr(function, "arguments", None)
                call_id = getattr(call, "id", None)
            if not call_id and not name and not arguments:
                continue
            parsed.append(LlmToolCall(id=call_id or "", name=name or "", arguments=arguments or ""))
        return parsed

    @staticmethod
    def _thinking_blocks(message) -> list[dict[str, Any]] | None:
        """
        Read Anthropic thinking blocks from a LiteLLM message.
        Args:
            message: Completion message or streaming delta.
        Returns:
            Thinking blocks, or None when this response has none.
        """
        blocks = getattr(message, "thinking_blocks", None)
        if not blocks:
            return None
        converted = []
        for block in blocks:
            if isinstance(block, dict):
                converted.append(block)
            elif hasattr(block, "model_dump"):
                converted.append(block.model_dump())
            else:
                converted.append(
                    {
                        "type": getattr(block, "type", None),
                        "thinking": getattr(block, "thinking", None),
                        "signature": getattr(block, "signature", None),
                    }
                )
        return converted or None

    @staticmethod
    def _has_payload(output: LLMOutput) -> bool:
        """
        Decide whether a streamed piece contains anything the caller needs.
        Args:
            output: One streamed piece.
        Returns:
            True when the piece has text, a tool call, reasoning, a finish reason, or token usage.
        """
        return any(
            (
                output.finish_reason,
                output.content,
                output.tool_calls,
                output.reasoning_content,
                output.thinking_blocks,
                output.usage,
                output.structured,
            )
        )
