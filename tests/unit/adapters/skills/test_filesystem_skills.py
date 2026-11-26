import asyncio
from pathlib import Path
from threading import Event

import pytest
import src.adapters.skills.filesystem as filesystem_module
from src.adapters.skills.filesystem import FilesystemSkillProvider, SkillProviderError
from src.core.context.skills import SkillBinding
from src.core.ids import ProfileId

PROFILE = ProfileId("profile")


def write_skill(
    root: Path,
    name: str,
    *,
    version: str = "2",
    description: str = "Testing practices for Python projects",
    body: str = "Use focused tests before implementation.",
) -> Path:
    directory = root / name
    directory.mkdir()
    path = directory / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\nversion: {version}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_catalog_activation_and_pinned_resolution_are_deterministic(tmp_path: Path):
    write_skill(tmp_path, "python-testing")
    write_skill(tmp_path, "architecture", version="1", description="Architecture review")
    provider = FilesystemSkillProvider(tmp_path.resolve())

    catalog = asyncio.run(provider.catalog(PROFILE))
    assert [(item.name, item.version) for item in catalog] == [
        ("architecture", "1"),
        ("python-testing", "2"),
    ]
    resolved = asyncio.run(provider.activate("python-testing", profile_id=PROFILE))
    assert resolved.binding.name == "python-testing"
    assert resolved.binding.version == "2"
    assert len(resolved.binding.content_sha256) == 64
    assert tuple(message.content for message in resolved.messages) == (
        "Skill: python-testing\nVersion: 2\n\nUse focused tests before implementation.",
    )
    again = asyncio.run(provider.resolve(resolved.binding, profile_id=PROFILE))
    assert again == resolved
    assert again is not resolved
    assert again.binding is not resolved.binding
    assert again.messages[0] is not resolved.messages[0]


@pytest.mark.parametrize(
    "document",
    [
        "name: skill\nversion: 1\ndescription: missing delimiters\nbody",
        "---\nname: skill\nname: skill\nversion: 1\ndescription: duplicate\n---\nbody",
        "---\nname: skill\nversion: 1\ndescription: ok\nunknown: no\n---\nbody",
        "---\nname: other\nversion: 1\ndescription: mismatch\n---\nbody",
        "---\nname: skill\nversion: 1\ndescription: ok\n---\n   \n",
        "---\nname: skill\nversion: 1\ndescription: ok\n---\nbody\x00bad",
    ],
)
def test_malformed_skill_documents_fail_closed(tmp_path: Path, document: str):
    directory = tmp_path / "skill"
    directory.mkdir()
    (directory / "SKILL.md").write_text(document, encoding="utf-8")
    provider = FilesystemSkillProvider(tmp_path.resolve())
    with pytest.raises(SkillProviderError):
        asyncio.run(provider.catalog(PROFILE))


def test_invalid_utf8_and_oversized_files_fail_closed(tmp_path: Path):
    path = write_skill(tmp_path, "skill")
    provider = FilesystemSkillProvider(tmp_path.resolve(), max_skill_bytes=64)
    with pytest.raises(SkillProviderError):
        asyncio.run(provider.catalog(PROFILE))

    path.write_bytes(b"\xff")
    provider = FilesystemSkillProvider(tmp_path.resolve())
    with pytest.raises(SkillProviderError):
        asyncio.run(provider.activate("skill", profile_id=PROFILE))


def test_catalog_bounds_and_duplicate_logical_names_are_rejected(tmp_path: Path):
    write_skill(tmp_path, "one", description="x" * 40)
    write_skill(tmp_path, "two", description="x" * 40)
    with pytest.raises(SkillProviderError):
        asyncio.run(
            FilesystemSkillProvider(tmp_path.resolve(), max_catalog_entries=1).catalog(PROFILE)
        )
    with pytest.raises(SkillProviderError):
        asyncio.run(
            FilesystemSkillProvider(tmp_path.resolve(), max_catalog_bytes=32).catalog(PROFILE)
        )


def test_catalog_byte_limit_uses_canonical_context_encoding(tmp_path: Path):
    for index in range(16):
        write_skill(
            tmp_path,
            f"skill-{index:02d}",
            version="v" * 128,
            description="d" * 3_950,
        )

    provider = FilesystemSkillProvider(tmp_path.resolve())

    with pytest.raises(SkillProviderError, match="byte limit"):
        asyncio.run(provider.catalog(PROFILE))


@pytest.mark.parametrize("name", ["../skill", "a/b", r"a\b", ".", "..", ""])
def test_activate_accepts_only_a_logical_name(tmp_path: Path, name: str):
    write_skill(tmp_path, "skill")
    provider = FilesystemSkillProvider(tmp_path.resolve())
    with pytest.raises(SkillProviderError):
        asyncio.run(provider.activate(name, profile_id=PROFILE))


def test_changed_version_or_content_cannot_replace_a_pinned_binding(tmp_path: Path):
    path = write_skill(tmp_path, "skill", body="original")
    provider = FilesystemSkillProvider(tmp_path.resolve())
    binding = asyncio.run(provider.activate("skill", profile_id=PROFILE)).binding

    path.write_text(
        "---\nname: skill\nversion: 2\ndescription: changed\n---\nreplacement\n",
        encoding="utf-8",
    )
    with pytest.raises(SkillProviderError):
        asyncio.run(provider.resolve(binding, profile_id=PROFILE))
    with pytest.raises(SkillProviderError):
        asyncio.run(
            provider.resolve(SkillBinding("skill", "3", binding.content_sha256), profile_id=PROFILE)
        )


def test_symlinked_directory_or_file_is_rejected(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    write_skill(target, "real")
    linked_directory = tmp_path / "linked"
    try:
        linked_directory.symlink_to(target / "real", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(SkillProviderError):
        asyncio.run(FilesystemSkillProvider(tmp_path.resolve()).catalog(PROFILE))


def test_root_must_be_an_existing_absolute_directory(tmp_path: Path):
    for root in (Path("relative"), (tmp_path / "missing").resolve()):
        with pytest.raises(ValueError):
            FilesystemSkillProvider(root)


def test_filesystem_reads_do_not_block_the_event_loop(tmp_path: Path, monkeypatch):
    write_skill(tmp_path, "skill")
    provider = FilesystemSkillProvider(tmp_path.resolve())
    entered = Event()
    release = Event()
    original = filesystem_module._read_document

    def blocking(*args, **kwargs):
        entered.set()
        release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(filesystem_module, "_read_document", blocking)

    async def scenario():
        pending = asyncio.create_task(provider.catalog(PROFILE))
        assert await asyncio.to_thread(entered.wait, 2)
        advanced = False

        async def marker():
            nonlocal advanced
            advanced = True

        await marker()
        assert advanced
        release.set()
        await pending

    asyncio.run(scenario())


def test_cancelled_read_does_not_publish_a_result(tmp_path: Path, monkeypatch):
    write_skill(tmp_path, "skill")
    provider = FilesystemSkillProvider(tmp_path.resolve())
    entered = Event()
    release = Event()
    original = filesystem_module._read_document

    def blocking(*args, **kwargs):
        entered.set()
        release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(filesystem_module, "_read_document", blocking)

    async def scenario():
        pending = asyncio.create_task(provider.activate("skill", profile_id=PROFILE))
        assert await asyncio.to_thread(entered.wait, 2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()

    asyncio.run(scenario())
