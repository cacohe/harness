from dataclasses import dataclass

MAX_ACTIVE_SKILLS = 16
MAX_SKILL_ID_BYTES = 128


def bounded_skill_text(value: object, limit: int, label: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be non-blank canonical text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{label} must be valid UTF-8 text") from None
    if len(encoded) > limit:
        raise ValueError(f"{label} exceeds its UTF-8 byte limit")
    return value


@dataclass(frozen=True, slots=True)
class SkillBinding:
    name: str
    version: str
    content_sha256: str

    def __post_init__(self) -> None:
        bounded_skill_text(self.name, MAX_SKILL_ID_BYTES, "Skill name")
        bounded_skill_text(self.version, MAX_SKILL_ID_BYTES, "Skill version")
        if (
            type(self.content_sha256) is not str
            or len(self.content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.content_sha256)
        ):
            raise ValueError("Skill content digest must be lowercase SHA-256 hex")


def snapshot_skill_binding(value: object) -> SkillBinding:
    if type(value) is not SkillBinding:
        raise ValueError("Skill binding has an invalid type")
    try:
        return SkillBinding(
            object.__getattribute__(value, "name"),
            object.__getattribute__(value, "version"),
            object.__getattribute__(value, "content_sha256"),
        )
    except AttributeError:
        raise ValueError("Skill binding is invalid") from None


def snapshot_skill_bindings(value: object) -> tuple[SkillBinding, ...]:
    if type(value) is not tuple or len(value) > MAX_ACTIVE_SKILLS:
        raise ValueError("Active Skills must be a bounded tuple")
    bindings = tuple(snapshot_skill_binding(binding) for binding in value)
    if len({binding.name for binding in bindings}) != len(bindings):
        raise ValueError("Active Skill names must be unique")
    return bindings
