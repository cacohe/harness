from __future__ import annotations

import json

from src.core.context import ConversationInteraction
from src.core.journal import TranscriptEntry, TranscriptEntryType
from src.core.model import ModelMessage, ModelRole, ModelToolCall

MAX_TRANSCRIPT_ROWS_PER_INTERACTION = 65


def transcript_interactions(
    entries: tuple[TranscriptEntry, ...],
) -> tuple[ConversationInteraction, ...]:
    """Project chronological transcript rows into complete model interactions."""

    if type(entries) is not tuple or any(type(entry) is not TranscriptEntry for entry in entries):
        raise ValueError("Transcript projection requires durable transcript rows")
    if any(
        type(entry.entry_type) is not TranscriptEntryType or entry.format_version != 1
        for entry in entries
    ):
        raise ValueError("Transcript projection requires a supported structured format")
    if any(older.sequence >= newer.sequence for older, newer in zip(entries, entries[1:])):
        raise ValueError("Transcript rows must be strictly ordered")
    interactions: list[ConversationInteraction] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        if entry.entry_type is TranscriptEntryType.TOOL_RESULT:
            raise ValueError("Transcript tool batch is missing its assistant call")
        if entry.entry_type is TranscriptEntryType.ASSISTANT_TOOL_BATCH:
            if entry.role != "assistant":
                raise ValueError("Transcript assistant tool batch has the wrong role")
            assistant = _assistant_message(entry.content)
            count = len(assistant.tool_calls)
            batch = entries[index + 1 : index + count + 1]
            if len(batch) != count or any(
                item.role != "tool" or item.entry_type is not TranscriptEntryType.TOOL_RESULT
                for item in batch
            ):
                raise ValueError("Transcript tool batch is incomplete")
            tools = tuple(_tool_message(item.content) for item in batch)
            expected = {call.tool_call_id for call in assistant.tool_calls}
            actual = {message.tool_call_id for message in tools}
            if len(actual) != len(tools) or actual != expected:
                raise ValueError("Transcript tool batch does not match its assistant call")
            interactions.append(
                ConversationInteraction(
                    entry.sequence,
                    batch[-1].sequence,
                    (assistant, *tools),
                )
            )
            index += count + 1
            continue
        if entry.entry_type is not TranscriptEntryType.MESSAGE:
            raise ValueError("Transcript row has an unsupported entry type")
        if entry.role == "assistant":
            message = ModelMessage(ModelRole.ASSISTANT, entry.content)
        elif entry.role == "user":
            message = ModelMessage(ModelRole.USER, entry.content)
        elif entry.role == "system":
            message = ModelMessage(ModelRole.SYSTEM, entry.content)
        else:
            raise ValueError("Transcript row has an unsupported model role")
        interactions.append(ConversationInteraction(entry.sequence, entry.sequence, (message,)))
        index += 1
    return tuple(interactions)


def newest_transcript_interactions(
    entries: tuple[TranscriptEntry, ...], limit: int
) -> tuple[ConversationInteraction, ...]:
    if type(limit) is not int or limit <= 0:
        raise ValueError("Transcript page limit is invalid")
    return tuple(reversed(transcript_interactions(entries)))[:limit]


def _assistant_message(content: str) -> ModelMessage:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, RecursionError):
        raise ValueError("Transcript assistant tool batch is invalid") from None
    if type(payload) is not dict or set(payload) != {"text", "tool_calls"}:
        raise ValueError("Transcript assistant tool batch is invalid")
    text = payload.get("text")
    calls = payload.get("tool_calls")
    if type(text) is not str or type(calls) is not list or not calls:
        raise ValueError("Transcript assistant tool batch is invalid")
    projected: list[ModelToolCall] = []
    for call in calls:
        if type(call) is not dict or set(call) != {"id", "name", "arguments"}:
            raise ValueError("Transcript assistant tool batch is invalid")
        identifier = call["id"]
        name = call["name"]
        arguments = call["arguments"]
        if type(identifier) is not str or type(name) is not str or type(arguments) is not str:
            raise ValueError("Transcript assistant tool batch is invalid")
        projected.append(ModelToolCall(identifier, name, arguments))
    return ModelMessage(ModelRole.ASSISTANT, text or None, tuple(projected))


def _tool_message(content: str) -> ModelMessage:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, RecursionError):
        raise ValueError("Transcript tool batch result is invalid") from None
    if type(payload) is not dict or type(payload.get("tool_call_id")) is not str:
        raise ValueError("Transcript tool batch result is invalid")
    return ModelMessage(ModelRole.TOOL, content, tool_call_id=payload["tool_call_id"])
