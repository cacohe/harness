from dataclasses import FrozenInstanceError

import pytest
from src.core.artifacts import ArtifactRef, ArtifactWrite, ExternalizedContent, InlineContent

DIGEST = "a" * 64
ARTIFACT_ID = "sha256:" + DIGEST


async def _empty_stream():
    if False:
        yield b""


def test_ref_metadata_is_copy_isolated_and_ref_is_frozen():
    source = {"labels": ["a"]}
    ref = ArtifactRef(ARTIFACT_ID, "text/plain", 0, DIGEST, "结果.txt", source)

    source["labels"].append("b")
    ref.metadata["labels"].append("c")

    assert ref.metadata == {"labels": ["a"]}
    with pytest.raises(FrozenInstanceError):
        ref.name = "changed"


def test_metadata_at_the_exact_encoded_size_limit_is_accepted():
    metadata = {"x": "a" * 65_528}
    ref = ArtifactRef(ARTIFACT_ID, "text/plain", 0, DIGEST, metadata=metadata)
    assert ref.metadata == metadata

    metadata["x"] += "a"
    with pytest.raises(ValueError):
        ArtifactRef(ARTIFACT_ID, "text/plain", 0, DIGEST, metadata=metadata)


@pytest.mark.parametrize(
    "location",
    [
        "source,/private/secret",
        "source|/private/secret",
        "source;/private/secret",
        "source,C:\\private\\secret",
        "source|file:secret",
        "src]/private/secret",
        "src]~/private/secret",
        "src]../private/secret",
        "src]C:\\private\\secret",
        "src]\\\\server\\share",
        "/",
        " / folder/file",
        "src]/",
    ],
)
def test_metadata_rejects_host_locations_at_token_boundaries(location):
    with pytest.raises(ValueError):
        ArtifactRef(ARTIFACT_ID, "text/plain", 0, DIGEST, metadata={"source": location})


@pytest.mark.parametrize(
    "key",
    [
        "bucket_access_key",
        "bucket_secret_key",
        "aws_access_key_id",
        "AWS_SECRET_ACCESS_KEY",
        "secret_access_key",
        "presigned_url",
        "signed-url",
        "credentials",
        "aws_session_token",
        "awsAccessKeyId",
        "AWSAccessKeyID",
        "access_key",
        "secret_key",
        "sessionToken",
        "presignedDownloadURL",
        "storageCredentialsVersion",
    ],
)
def test_metadata_rejects_credential_fields_at_any_depth(key):
    with pytest.raises(ValueError):
        ArtifactRef(
            ARTIFACT_ID,
            "text/plain",
            0,
            DIGEST,
            metadata={"nested": [{key: "opaque"}]},
        )


def test_metadata_preserves_ordinary_statistics_fields():
    metadata = {"token_count": 3, "key_count": 2, "tokenCount": 3, "keyCount": 2}
    ref = ArtifactRef(
        ARTIFACT_ID,
        "text/plain",
        0,
        DIGEST,
        metadata=metadata,
    )
    assert ref.metadata == metadata


@pytest.mark.parametrize("value", ["foo/bar", "1/2", "folder/subfolder/item", "比例1/2"])
def test_metadata_accepts_slashes_within_non_host_tokens(value):
    ref = ArtifactRef(ARTIFACT_ID, "text/plain", 0, DIGEST, metadata={"source": value})
    assert ref.metadata["source"] == value


@pytest.mark.parametrize(
    "forged",
    [
        '{"nested":[NaN]}',
        '{"result":Infinity}',
        '{"result":-Infinity}',
        '{"result":9e999}',
        '{"result":"source|/private/secret"}',
        '{"nested":{"bucket_secret_key":"opaque"}}',
        '{"result":' + "[" * 16 + "0" + "]" * 16 + "}",
        "[]",
    ],
)
def test_metadata_read_validates_untrusted_serialized_fields(forged):
    ref = ArtifactRef(ARTIFACT_ID, "text/plain", 0, DIGEST)
    object.__setattr__(ref, "_metadata_json", forged)
    with pytest.raises(ValueError):
        _ = ref.metadata


@pytest.mark.parametrize(
    "metadata",
    [
        "{}",
        [],
        {1: "non-string key"},
        {"binary": b"abc"},
        {"tuple": (1, 2)},
        {"nan": float("nan")},
        {"nested": [[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]},
        {"too_large": "界" * 24_000},
        {"unpaired": "\ud800"},
        {"location": "C:\\private\\test"},
        {"location": "https://example.com/file"},
    ],
)
def test_ref_and_write_require_bounded_strict_json_metadata(metadata):
    with pytest.raises(ValueError):
        ArtifactRef(ARTIFACT_ID, "text/plain", 0, DIGEST, metadata=metadata)
    with pytest.raises(ValueError):
        ArtifactWrite(_empty_stream(), "text/plain", metadata=metadata)


@pytest.mark.parametrize("size", [-1, True, 2**63])
def test_ref_rejects_invalid_size(size):
    with pytest.raises(ValueError):
        ArtifactRef(ARTIFACT_ID, "text/plain", size, DIGEST)


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64, DIGEST + "\n"])
def test_ref_requires_a_full_lowercase_sha256(digest):
    with pytest.raises(ValueError):
        ArtifactRef("sha256:" + digest, "text/plain", 0, digest)


def test_ref_requires_id_bound_to_the_sha256():
    with pytest.raises(ValueError):
        ArtifactRef("sha256:" + "b" * 64, "text/plain", 0, DIGEST)


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "./result.txt",
        "nested/result.txt",
        "nested\\result.txt",
        "C:result.txt",
        "https://example.com/a",
        "report\n.txt",
        " report.txt",
        "R" * 257,
        "\ud800",
    ],
)
def test_ref_name_is_a_bounded_display_name(name):
    with pytest.raises(ValueError):
        ArtifactRef(ARTIFACT_ID, "text/plain", 0, DIGEST, name)


@pytest.mark.parametrize(
    "media_type",
    [
        "Text/plain",
        " text/plain",
        "text/plain; charset=utf-8",
        "text/plain\r\n",
        "text/*",
        "text",
        "text/plain/json",
        "text/" + "a" * 128,
    ],
)
def test_ref_requires_a_canonical_exact_media_type(media_type):
    with pytest.raises(ValueError):
        ArtifactRef(ARTIFACT_ID, media_type, 0, DIGEST)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_size_bytes": -1},
        {"expected_size_bytes": True},
        {"expected_sha256": "A" * 64},
        {"max_bytes": True},
        {"max_bytes": 0},
        {"max_bytes": 2**63},
        {"expected_size_bytes": 3, "max_bytes": 2},
    ],
)
def test_write_validates_expected_hash_size_and_stream_limit(kwargs):
    with pytest.raises(ValueError):
        ArtifactWrite(_empty_stream(), "text/plain", **kwargs)


def test_write_rejects_synchronous_input():
    with pytest.raises(ValueError):
        ArtifactWrite(iter([b"abc"]), "text/plain")


def test_inline_values_are_isolated_and_distinguish_json_null_from_absence():
    content = InlineContent({"value": ["ok"]}, "application/json")
    content.value["value"].append("mutation")
    assert content.value == {"value": ["ok"]}
    assert content.size_bytes == 16
    null = ExternalizedContent(inline_content=InlineContent(None, "application/json"))
    assert null.inline_content.value is None
    assert null.inline_content.size_bytes == 4
    assert null.artifacts == () and null.summary is None
    assert InlineContent(b"", "application/octet-stream").value == b""
    assert InlineContent("界", "text/plain").size_bytes == 3


def test_externalized_content_requires_one_complete_variant():
    ref = ArtifactRef(ARTIFACT_ID, "text/plain", 0, DIGEST)
    external = ExternalizedContent(artifacts=(ref,), summary="empty file")
    assert external.inline_content is None
    assert external.artifacts == (ref,)

    for kwargs in (
        {},
        {"summary": "missing artifact"},
        {"artifacts": (ref,)},
        {"artifacts": (ref,), "summary": ""},
        {"artifacts": (ref,), "summary": "\ud800"},
        {"artifacts": (ref,), "summary": "界" * 400_000},
        {
            "artifacts": (ref,),
            "summary": "mixed",
            "inline_content": InlineContent("", "text/plain"),
        },
        {"artifacts": [ref], "summary": "mutable refs"},
        {"artifacts": (object(),), "summary": "invalid ref"},
        {"artifacts": (ref,) * 17, "summary": "too many refs"},
        {"inline_content": "untyped"},
    ):
        with pytest.raises(ValueError):
            ExternalizedContent(**kwargs)


def test_externalized_content_isolates_inputs_and_readers_from_nested_mutations():
    ref = ArtifactRef(ARTIFACT_ID, "text/plain", 0, DIGEST, name="original.txt")
    inline = InlineContent({"value": 1}, "application/json")
    externalized = ExternalizedContent(artifacts=(ref,), summary="summary")
    inlined = ExternalizedContent(inline_content=inline)

    object.__setattr__(ref, "artifact_id", "corrupted")
    object.__setattr__(inline, "_encoded", b'{"value":2}')
    object.__setattr__(externalized.artifacts[0], "name", "mutated.txt")
    object.__setattr__(inlined.inline_content, "size_bytes", 0)

    assert externalized.artifacts[0].artifact_id == ARTIFACT_ID
    assert externalized.artifacts[0].name == "original.txt"
    assert inlined.inline_content.value == {"value": 1}
    assert inlined.inline_content.size_bytes == 11


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_id", "sha256:" + "b" * 64),
        ("size_bytes", -1),
        ("_metadata_json", '{"x":NaN}'),
    ],
)
def test_externalized_content_rejects_forged_refs(field, value):
    ref = ArtifactRef(ARTIFACT_ID, "text/plain", 0, DIGEST)
    object.__setattr__(ref, field, value)
    with pytest.raises(ValueError):
        ExternalizedContent(artifacts=(ref,), summary="valid summary")


@pytest.mark.parametrize(
    "field,value",
    [
        ("size_bytes", 0),
        ("size_bytes", True),
        ("_is_binary", "yes"),
        ("_is_binary", True),
        ("_encoded", b"NaN!"),
        ("_encoded", b"01.0"),
        ("media_type", "text/plain;invalid"),
    ],
)
def test_externalized_content_rejects_forged_inline_values(field, value):
    inline = InlineContent(None, "application/json")
    object.__setattr__(inline, field, value)
    with pytest.raises(ValueError):
        ExternalizedContent(inline_content=inline)


@pytest.mark.parametrize(
    "value,media_type",
    [
        (b"x" * 1_048_577, "application/octet-stream"),
        (float("nan"), "application/json"),
        ({"x": 1}, "text/plain"),
    ],
)
def test_inline_content_rejects_invalid_or_over_limit_values(value, media_type):
    with pytest.raises(ValueError):
        InlineContent(value, media_type)
