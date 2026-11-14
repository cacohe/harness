import re

import pytest
from src.utils.tool_names import provider_tool_name


def test_provider_tool_name_normalizes_each_segment_to_safe_ascii():
    name = provider_tool_name("skill", "weather api", "read.file/v2")

    assert name == "skill__weather_api__read_file_v2"
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)


@pytest.mark.parametrize(
    "segments",
    [
        ("天气", "forecast"),
        ("skill", "!!!"),
        ("skill", 42),
    ],
)
def test_provider_tool_name_rejects_segments_without_safe_ascii(segments):
    with pytest.raises(ValueError, match="provider-safe ASCII"):
        provider_tool_name(*segments)


def test_provider_tool_name_truncation_is_stable_and_collision_resistant():
    first = provider_tool_name("skill", "research", "read_" + "x" * 100)
    repeated = provider_tool_name("skill", "research", "read_" + "x" * 100)
    different = provider_tool_name("skill", "research", "read_" + "y" * 100)

    assert first == repeated
    assert len(first) == 64
    assert first != different
    assert first[-10:] != different[-10:]
