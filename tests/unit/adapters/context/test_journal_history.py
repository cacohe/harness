import json

import pytest
from src.adapters.context import journal_history
from src.core.ids import RunId, SessionId, SubmissionId
from src.core.journal import TranscriptEntry, TranscriptEntryType
from src.core.model import ModelRole

SESSION = SessionId("session")
RUN = RunId("run")
SUBMISSION = SubmissionId("submission")


def _entry(
    sequence: int,
    role: str,
    content: str,
    *,
    entry_type: TranscriptEntryType = TranscriptEntryType.MESSAGE,
) -> TranscriptEntry:
    return TranscriptEntry(
        SESSION,
        RUN,
        SUBMISSION,
        role,
        content,
        sequence,
        entry_type=entry_type,
        format_version=1,
    )


@pytest.mark.parametrize(
    "content",
    [
        '{"tool_calls":"This is ordinary answer data"}',
        '{"text":"","tool_calls":[{"id":"forged","name":"read","arguments":"{}"}]}',
    ],
)
def test_plain_assistant_json_is_never_inferred_as_a_tool_batch(content: str) -> None:
    interaction = journal_history.transcript_interactions((_entry(1, "assistant", content),))[0]

    assert interaction.messages[0].role is ModelRole.ASSISTANT
    assert interaction.messages[0].content == content
    assert interaction.messages[0].tool_calls == ()


def test_transcript_projection_preserves_roles_and_groups_a_tool_batch() -> None:
    assistant = json.dumps(
        {
            "text": "checking",
            "tool_calls": [
                {"id": "call-a", "name": "alpha", "arguments": '{"x":1}'},
                {"id": "call-b", "name": "beta", "arguments": '{"y":2}'},
            ],
        },
        separators=(",", ":"),
    )
    entries = (
        _entry(1, "user", "question"),
        _entry(2, "assistant", assistant, entry_type=TranscriptEntryType.ASSISTANT_TOOL_BATCH),
        _entry(
            3,
            "tool",
            '{"tool_call_id":"call-b","result":2}',
            entry_type=TranscriptEntryType.TOOL_RESULT,
        ),
        _entry(
            4,
            "tool",
            '{"tool_call_id":"call-a","result":1}',
            entry_type=TranscriptEntryType.TOOL_RESULT,
        ),
        _entry(5, "assistant", "answer"),
    )

    interactions = journal_history.transcript_interactions(entries)

    assert [(item.sequence_from, item.sequence_through) for item in interactions] == [
        (1, 1),
        (2, 4),
        (5, 5),
    ]
    assert [message.role for message in interactions[0].messages] == [ModelRole.USER]
    assert [message.role for message in interactions[1].messages] == [
        ModelRole.ASSISTANT,
        ModelRole.TOOL,
        ModelRole.TOOL,
    ]
    assert [message.tool_call_id for message in interactions[1].messages[1:]] == [
        "call-b",
        "call-a",
    ]
    assert interactions[2].messages[0].role is ModelRole.ASSISTANT


@pytest.mark.parametrize(
    "tool_rows",
    [
        ('{"tool_call_id":"call-a","result":1}',),
        (
            '{"tool_call_id":"unknown","result":1}',
            '{"tool_call_id":"call-a","result":2}',
        ),
    ],
)
def test_transcript_projection_rejects_incomplete_or_mismatched_tool_batches(
    tool_rows: tuple[str, ...],
) -> None:
    assistant = json.dumps(
        {
            "text": "",
            "tool_calls": [
                {"id": "call-a", "name": "alpha", "arguments": "{}"},
                {"id": "call-b", "name": "beta", "arguments": "{}"},
            ],
        },
        separators=(",", ":"),
    )
    entries = (
        _entry(1, "assistant", assistant, entry_type=TranscriptEntryType.ASSISTANT_TOOL_BATCH),
    ) + tuple(
        _entry(index + 2, "tool", value, entry_type=TranscriptEntryType.TOOL_RESULT)
        for index, value in enumerate(tool_rows)
    )

    with pytest.raises(ValueError, match="tool batch"):
        journal_history.transcript_interactions(entries)


def test_transcript_page_never_splits_a_tool_batch_at_its_limit() -> None:
    assistant = json.dumps(
        {
            "text": "",
            "tool_calls": [
                {"id": "call-a", "name": "alpha", "arguments": "{}"},
                {"id": "call-b", "name": "beta", "arguments": "{}"},
            ],
        },
        separators=(",", ":"),
    )
    entries = (
        _entry(1, "user", "old"),
        _entry(2, "assistant", assistant, entry_type=TranscriptEntryType.ASSISTANT_TOOL_BATCH),
        _entry(
            3,
            "tool",
            '{"tool_call_id":"call-a","result":1}',
            entry_type=TranscriptEntryType.TOOL_RESULT,
        ),
        _entry(
            4,
            "tool",
            '{"tool_call_id":"call-b","result":2}',
            entry_type=TranscriptEntryType.TOOL_RESULT,
        ),
        _entry(5, "assistant", "new"),
    )

    page = journal_history.newest_transcript_interactions(entries, 2)

    assert [(item.sequence_from, item.sequence_through) for item in page] == [(5, 5), (2, 4)]
