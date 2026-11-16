"""Load runtime settings and expose the values the harness reads."""

import copy
import json
import math
import os
import tomllib
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from pydantic import BaseModel, field_validator

_ENV_FILE_NAME = ".env"
_CONFIG_FILE_NAMES = ("config.toml", "config.json")
_CONFIG_FILE_VARIABLE = "HARNESS_CONFIG_FILE"
_ENV_FILE_VARIABLE = "HARNESS_ENV_FILE"
_DEFAULT_MAX_STEPS = 10
_DEFAULT_MAX_SECONDS = 60 * 30
_DEFAULT_MODEL_TIMEOUT_SECONDS = 90
_DEFAULT_TOOL_TIMEOUT_SECONDS = 30
_DEFAULT_MAX_MODEL_OUTPUT_BYTES = 1_048_576
_DEFAULT_MAX_TOOL_RESULT_BYTES = 1_048_576
_DEFAULT_MAX_EVENT_BYTES = 1_048_576
_DEFAULT_MAX_TASK_SNAPSHOT_BYTES = 8_388_608
_DEFAULT_MAX_PENDING_SUBMISSIONS = 0
_DEFAULT_CLOSE_TIMEOUT_SECONDS = 30


class ConfigError(Exception):
    """A configuration file or value cannot be used."""


class Settings:
    """Values loaded from the process environment, an env file, and a config file."""

    def __init__(self, files: dict[str, Any], environ: Mapping[str, str]):
        self._files = files
        self._environ = environ

    def get(self, key: str, default: Any = None) -> Any:
        """
        Read one value.
        Args:
            key: A top-level name, or a dotted path such as "llm.model".
            default: Value returned when the key is absent.
        Returns:
            A copy of the stored value. The process environment wins over the env file,
            and the env file wins over the config file.
        """
        found, value = self._find(key)
        if not found:
            return default
        return copy.deepcopy(value)

    def require(self, key: str) -> Any:
        """
        Read one value that must be present.
        Args:
            key: A top-level name, or a dotted path such as "llm.model".
        Returns:
            A copy of the stored value. An empty string is present.
        Raises:
            ConfigError: The key is absent.
        """
        found, value = self._find(key)
        if not found:
            raise ConfigError(f"missing required config: {key}")
        return copy.deepcopy(value)

    def as_dict(self) -> dict[str, Any]:
        """
        Return the values that came from files.
        Returns:
            A copy of the config file overlaid with the env file.
            Process environment values are not included.
        """
        return copy.deepcopy(self._files)

    def __contains__(self, key: str) -> bool:
        return self._find(key)[0]

    def _find(self, key: str) -> tuple[bool, Any]:
        if key in self._environ:
            return True, self._environ[key]
        return _lookup(self._files, key)


def load(
    env_file: Path | str | None = None,
    config_file: Path | str | None = None,
    environ: MutableMapping[str, str] | None = None,
    override: bool = False,
    export: bool = True,
) -> Settings:
    """
    Load settings from the process environment, an env file, and a config file.
    Args:
        env_file: Env file to read. When omitted, HARNESS_ENV_FILE or .env is used.
        config_file: JSON or TOML file to read. When omitted, HARNESS_CONFIG_FILE,
            config.toml, or config.json is used. config.toml wins when both defaults exist.
        environ: Environment mapping. When omitted, the process environment is used.
        override: When True, env-file values replace keys already present in environ.
        export: When True, env-file values are written into environ.
    Returns:
        Settings whose lookup order is process environment, env file, then config file.
    Raises:
        ConfigError: A selected file is missing, unsupported, or invalid.
    """
    if environ is None:
        environ = os.environ
    env_path = _resolve_env_file(env_file, environ)
    config_path = _resolve_config_file(config_file, environ)
    file_values: dict[str, Any] = {}
    if config_path is not None:
        file_values.update(_read_config(config_path))
    if env_path is not None:
        env_values = _read_env(env_path)
        file_values.update(env_values)
        if export:
            for key, value in env_values.items():
                if override or key not in environ:
                    environ[key] = value
    return Settings(file_values, environ)


class Config(BaseModel):
    """Runtime values injected into the harness."""

    llm: str
    llm_api_key: str | None = None
    llm_api_base: str | None = None
    max_steps: int = _DEFAULT_MAX_STEPS
    max_seconds: int = _DEFAULT_MAX_SECONDS
    model_timeout_seconds: float = _DEFAULT_MODEL_TIMEOUT_SECONDS
    tool_timeout_seconds: float = _DEFAULT_TOOL_TIMEOUT_SECONDS
    max_model_output_bytes: int = _DEFAULT_MAX_MODEL_OUTPUT_BYTES
    max_tool_result_bytes: int = _DEFAULT_MAX_TOOL_RESULT_BYTES
    max_event_bytes: int = _DEFAULT_MAX_EVENT_BYTES
    max_task_snapshot_bytes: int = _DEFAULT_MAX_TASK_SNAPSHOT_BYTES
    max_pending_submissions: int = _DEFAULT_MAX_PENDING_SUBMISSIONS
    close_timeout_seconds: float = _DEFAULT_CLOSE_TIMEOUT_SECONDS

    @field_validator(
        "model_timeout_seconds", "tool_timeout_seconds", "close_timeout_seconds", mode="before"
    )
    @classmethod
    def validate_positive_seconds(cls, value: Any, info) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be a positive finite number")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{info.field_name} must be a positive finite number") from None
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{info.field_name} must be a positive finite number")
        return number

    @field_validator(
        "max_model_output_bytes",
        "max_tool_result_bytes",
        "max_event_bytes",
        "max_task_snapshot_bytes",
        mode="before",
    )
    @classmethod
    def validate_positive_bytes(cls, value: Any, info) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return value

    @field_validator("max_pending_submissions", mode="before")
    @classmethod
    def validate_pending_limit(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("max_pending_submissions must be a non-negative integer")
        return value

    @classmethod
    def load(cls, **kwargs) -> "Config":
        """
        Load the harness config from the environment and config files.
        Args:
            **kwargs: Arguments forwarded to load.
        Returns:
            The config used to build a gateway and a harness.
        Raises:
            ConfigError: The model name is missing, or a limit is not a positive integer.
        """
        return cls.from_settings(load(**kwargs))

    @classmethod
    def from_settings(cls, settings: Settings) -> "Config":
        """
        Read harness fields from loaded settings.
        Args:
            settings: Settings returned by load.
        Returns:
            The config used to build a gateway and a harness.
        Raises:
            ConfigError: The model name is missing, or a limit is not a positive integer.
        """
        model = _pick_text(settings, ("LLM_MODEL", "LLM", "llm", "llm.model"))
        if model is None:
            raise ConfigError("missing required config: llm")
        return cls(
            llm=model,
            llm_api_key=_pick_text(settings, ("LLM_API_KEY", "llm_api_key", "llm.api_key")),
            llm_api_base=_pick_text(settings, ("LLM_API_BASE", "llm_api_base", "llm.api_base")),
            max_steps=_pick_positive_int(settings, ("MAX_STEPS", "max_steps"), _DEFAULT_MAX_STEPS),
            max_seconds=_pick_positive_int(
                settings, ("MAX_SECONDS", "max_seconds"), _DEFAULT_MAX_SECONDS
            ),
            model_timeout_seconds=_pick_positive_float(
                settings,
                ("MODEL_TIMEOUT_SECONDS", "model_timeout_seconds"),
                _DEFAULT_MODEL_TIMEOUT_SECONDS,
            ),
            tool_timeout_seconds=_pick_positive_float(
                settings,
                ("TOOL_TIMEOUT_SECONDS", "tool_timeout_seconds"),
                _DEFAULT_TOOL_TIMEOUT_SECONDS,
            ),
            max_model_output_bytes=_pick_positive_int(
                settings,
                ("MAX_MODEL_OUTPUT_BYTES", "max_model_output_bytes"),
                _DEFAULT_MAX_MODEL_OUTPUT_BYTES,
            ),
            max_tool_result_bytes=_pick_positive_int(
                settings,
                ("MAX_TOOL_RESULT_BYTES", "max_tool_result_bytes"),
                _DEFAULT_MAX_TOOL_RESULT_BYTES,
            ),
            max_event_bytes=_pick_positive_int(
                settings,
                ("MAX_EVENT_BYTES", "max_event_bytes"),
                _DEFAULT_MAX_EVENT_BYTES,
            ),
            max_task_snapshot_bytes=_pick_positive_int(
                settings,
                ("MAX_TASK_SNAPSHOT_BYTES", "max_task_snapshot_bytes"),
                _DEFAULT_MAX_TASK_SNAPSHOT_BYTES,
            ),
            max_pending_submissions=_pick_nonnegative_int(
                settings,
                ("MAX_PENDING_SUBMISSIONS", "max_pending_submissions"),
                _DEFAULT_MAX_PENDING_SUBMISSIONS,
            ),
            close_timeout_seconds=_pick_positive_float(
                settings,
                ("CLOSE_TIMEOUT_SECONDS", "close_timeout_seconds"),
                _DEFAULT_CLOSE_TIMEOUT_SECONDS,
            ),
        )


def _pick_text(settings: Settings, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key not in settings:
            continue
        value = settings.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _pick_positive_int(settings: Settings, keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        if key not in settings:
            continue
        value = settings.get(key)
        if isinstance(value, str):
            value = value.strip()
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ConfigError(f"invalid integer config: {key}")
        try:
            number = int(value)
        except ValueError:
            raise ConfigError(f"invalid integer config: {key}") from None
        if number <= 0:
            raise ConfigError(f"invalid integer config: {key}")
        return number
    return default


def _pick_nonnegative_int(settings: Settings, keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        if key not in settings:
            continue
        value = settings.get(key)
        if isinstance(value, str):
            value = value.strip()
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ConfigError(f"invalid integer config: {key}")
        try:
            number = int(value)
        except ValueError:
            raise ConfigError(f"invalid integer config: {key}") from None
        if number < 0:
            raise ConfigError(f"invalid integer config: {key}")
        return number
    return default


def _pick_positive_float(
    settings: Settings, keys: tuple[str, ...], default: float
) -> float:
    for key in keys:
        if key not in settings:
            continue
        value = settings.get(key)
        if isinstance(value, str):
            value = value.strip()
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ConfigError(f"invalid number config: {key}")
        try:
            number = float(value)
        except ValueError:
            raise ConfigError(f"invalid number config: {key}") from None
        if not math.isfinite(number) or number <= 0:
            raise ConfigError(f"invalid number config: {key}")
        return number
    return float(default)


def _resolve_env_file(env_file: Path | str | None, environ: Mapping[str, str]) -> Path | None:
    if env_file is not None:
        return _existing_file(env_file)
    selected = environ.get(_ENV_FILE_VARIABLE)
    if selected:
        return _existing_file(selected)
    default = Path.cwd() / _ENV_FILE_NAME
    if default.is_file():
        return default
    return None


def _resolve_config_file(config_file: Path | str | None, environ: Mapping[str, str]) -> Path | None:
    if config_file is not None:
        return _existing_file(config_file)
    selected = environ.get(_CONFIG_FILE_VARIABLE)
    if selected:
        return _existing_file(selected)
    for name in _CONFIG_FILE_NAMES:
        default = Path.cwd() / name
        if default.is_file():
            return default
    return None


def _existing_file(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_file():
        raise ConfigError(f"config file not found: {candidate}")
    return candidate


def _read_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8-sig")
        if suffix == ".json":
            value = json.loads(text)
        elif suffix == ".toml":
            value = tomllib.loads(text)
        else:
            raise ConfigError(f"unsupported config file: {path}")
    except ConfigError:
        raise
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"invalid config file: {path}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"invalid config file: {path}")
    return value


def _read_env(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise ConfigError(f"invalid config file: {path}") from error
    return _parse_env(text)


def _parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") or line.startswith("export\t"):
            line = line.split(None, 1)[1].strip()
        if "=" not in line:
            raise ConfigError(f"invalid env line {line_number}")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"invalid env line {line_number}")
        values[key] = _parse_env_value(raw_value)
    return values


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] not in {'"', "'"}:
        comment = value.find(" #")
        if comment != -1:
            value = value[:comment]
        return value.strip()
    return _parse_quoted(value)


def _parse_quoted(value: str) -> str:
    quote = value[0]
    chars: list[str] = []
    index = 1
    while index < len(value):
        char = value[index]
        if char == "\\" and quote == '"':
            if index + 1 >= len(value):
                raise ConfigError("unterminated quote")
            escaped = value[index + 1]
            chars.append(
                {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}.get(escaped, escaped)
            )
            index += 2
            continue
        if char == quote:
            return "".join(chars)
        chars.append(char)
        index += 1
    raise ConfigError("unterminated quote")


def _lookup(values: Mapping[str, Any], key: str) -> tuple[bool, Any]:
    if key in values:
        return True, values[key]
    current: Any = values
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current
