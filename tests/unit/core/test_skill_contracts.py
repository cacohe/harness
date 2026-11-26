from dataclasses import FrozenInstanceError

import pytest
from src.core.context.model import ModelMessage, ModelRole
from src.core.context.skills import (
    ResolvedSkill,
    SkillBinding,
    SkillCatalogEntry,
    skill_content_sha256,
    snapshot_skill_catalog,
)


def test_skill_values_are_bounded_immutable_and_copy_isolated():
    entry = SkillCatalogEntry("review", "Review a change", "v1")
    message = ModelMessage(ModelRole.SYSTEM, "Follow the review workflow")
    binding = SkillBinding("review", "v1", skill_content_sha256((message,)))
    resolved = ResolvedSkill(binding, (message,))

    assert snapshot_skill_catalog((entry,)) == (entry,)
    assert resolved.binding == binding
    assert resolved.binding is not binding
    assert resolved.messages == (message,)
    assert resolved.messages[0] is not message
    with pytest.raises(FrozenInstanceError):
        resolved.binding.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SkillCatalogEntry("", "description", "v1"),
        lambda: SkillCatalogEntry("n" * 129, "description", "v1"),
        lambda: SkillCatalogEntry("name", "", "v1"),
        lambda: SkillCatalogEntry("name", "é" * 2049, "v1"),
        lambda: SkillCatalogEntry("name", "description", "v" * 129),
        lambda: SkillBinding("name", "v1", "A" * 64),
        lambda: SkillBinding("name", "v1", "a" * 63),
        lambda: ResolvedSkill(
            SkillBinding("name", "v1", "a" * 64),
            (ModelMessage(ModelRole.USER, "not system"),),
        ),
        lambda: ResolvedSkill(SkillBinding("name", "v1", "a" * 64), ()),
        lambda: ResolvedSkill(
            SkillBinding("name", "v1", "a" * 64),
            (ModelMessage(ModelRole.SYSTEM, "digest mismatch"),),
        ),
    ],
)
def test_skill_values_reject_invalid_or_unbounded_data(factory):
    with pytest.raises(ValueError):
        factory()


def test_skill_catalog_is_bounded_and_has_unique_names():
    with pytest.raises(ValueError):
        snapshot_skill_catalog(
            tuple(SkillCatalogEntry(f"skill-{index}", "description", "v1") for index in range(129))
        )
    with pytest.raises(ValueError):
        snapshot_skill_catalog(
            (
                SkillCatalogEntry("same", "first", "v1"),
                SkillCatalogEntry("same", "second", "v2"),
            )
        )
