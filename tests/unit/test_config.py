import math

import pytest
from src.config import Config, ConfigError, Settings


def test_resource_governance_defaults():
    config = Config(llm="model")

    assert config.model_timeout_seconds == 90
    assert config.tool_timeout_seconds == 30
    assert config.max_model_output_bytes == 1_048_576
    assert config.max_tool_result_bytes == 1_048_576
    assert config.max_event_bytes == 1_048_576
    assert config.max_task_snapshot_bytes == 8_388_608
    assert config.max_pending_submissions == 0
    assert config.close_timeout_seconds == 30


def test_resource_fields_load_from_environment_aliases():
    settings = Settings(
        {},
        {
            "LLM_MODEL": "model",
            "MODEL_TIMEOUT_SECONDS": "1.5",
            "TOOL_TIMEOUT_SECONDS": "2",
            "MAX_MODEL_OUTPUT_BYTES": "101",
            "MAX_TOOL_RESULT_BYTES": "102",
            "MAX_EVENT_BYTES": "103",
            "MAX_TASK_SNAPSHOT_BYTES": "104",
            "MAX_PENDING_SUBMISSIONS": "0",
            "CLOSE_TIMEOUT_SECONDS": "3.5",
        },
    )

    config = Config.from_settings(settings)

    assert config.model_timeout_seconds == 1.5
    assert config.tool_timeout_seconds == 2
    assert config.max_model_output_bytes == 101
    assert config.max_tool_result_bytes == 102
    assert config.max_event_bytes == 103
    assert config.max_task_snapshot_bytes == 104
    assert config.max_pending_submissions == 0
    assert config.close_timeout_seconds == 3.5


def test_resource_fields_load_from_file_names():
    config = Config.from_settings(
        Settings(
            {
                "llm": "model",
                "model_timeout_seconds": 4.5,
                "tool_timeout_seconds": 5,
                "max_model_output_bytes": 201,
                "max_tool_result_bytes": 202,
                "max_event_bytes": 203,
                "max_task_snapshot_bytes": 204,
                "max_pending_submissions": 2,
                "close_timeout_seconds": 6,
            },
            {},
        )
    )

    assert config.model_timeout_seconds == 4.5
    assert config.tool_timeout_seconds == 5
    assert config.max_pending_submissions == 2
    assert config.close_timeout_seconds == 6


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf, True])
@pytest.mark.parametrize(
    "field", ["model_timeout_seconds", "tool_timeout_seconds", "close_timeout_seconds"]
)
def test_seconds_require_positive_finite_numbers(field, value):
    with pytest.raises(ValueError, match=field):
        Config(llm="model", **{field: value})


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
@pytest.mark.parametrize(
    "field",
    [
        "max_model_output_bytes",
        "max_tool_result_bytes",
        "max_event_bytes",
        "max_task_snapshot_bytes",
    ],
)
def test_byte_limits_require_positive_integers(field, value):
    with pytest.raises(ValueError, match=field):
        Config(llm="model", **{field: value})


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_pending_submission_limit_is_a_nonnegative_integer(value):
    with pytest.raises(ValueError, match="max_pending_submissions"):
        Config(llm="model", max_pending_submissions=value)


@pytest.mark.parametrize(
    "key,value",
    [
        ("MODEL_TIMEOUT_SECONDS", "nan"),
        ("TOOL_TIMEOUT_SECONDS", "0"),
        ("MAX_EVENT_BYTES", "false"),
        ("MAX_PENDING_SUBMISSIONS", "-1"),
    ],
)
def test_invalid_loaded_resource_values_raise_config_error(key, value):
    with pytest.raises(ConfigError, match=key):
        Config.from_settings(Settings({}, {"LLM_MODEL": "model", key: value}))
