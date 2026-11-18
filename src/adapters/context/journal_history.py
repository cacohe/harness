from src.core.context import ConversationInteraction
from src.core.journal import TranscriptEntry
from src.core.model import ModelMessage, ModelRole


def transcript_interaction(entry: TranscriptEntry) -> ConversationInteraction:
    """Project one durable journal row into an indivisible, model-safe history item."""

    return ConversationInteraction(
        entry.sequence,
        entry.sequence,
        (ModelMessage(ModelRole.SYSTEM, f"Historical {entry.role}: {entry.content}"),),
    )
