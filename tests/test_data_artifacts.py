"""Fail-closed contracts for versioned packed-data artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hagi.data.artifacts import (
    MANIFEST_SCHEMA_VERSION,
    atomic_publish_directory,
    deduplicate_normalized_lines,
    file_entry,
    load_published_artifact,
    normalize_mix,
    quarantine_jsonl,
    read_utf8_records,
    split_packed_tokens,
    validate_manifest,
    validate_relative_path,
    validate_uint32_stream_bytes,
)


def manifest(**overrides):
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "dataset",
        "sources": [
            {
                "name": "a",
                "ratio": 2,
                "dataset": "ds",
                "revision": "r1",
                "license": "lic",
                "source_url": "local",
                "retrieval_timestamp": "2026",
                "tokenizer_version": "tok",
                "filter_policy_version": "f1",
                "dedup_policy_version": "d1",
                "byte_count": 10,
                "token_count": 5,
                "input_sha256": "a" * 64,
                "output_sha256": "b" * 64,
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_manifest_validation():
    validate_manifest(manifest())

    bad = manifest()
    del bad["sources"][0]["license"]
    with pytest.raises(ValueError, match="missing fields"):
        validate_manifest(bad)


def test_manifest_rejects_bad_ratio_and_duplicate_names():
    bad = manifest()
    bad["sources"][0]["ratio"] = float("nan")
    with pytest.raises(ValueError, match="finite positive"):
        validate_manifest(bad)

    duplicate = manifest()
    duplicate["sources"].append(dict(duplicate["sources"][0]))
    with pytest.raises(ValueError, match="unique"):
        validate_manifest(duplicate)


@pytest.mark.parametrize("value", ["../x.bin", "a/../x.bin", "/x.bin", "a//x.bin", "a\\x.bin"])
def test_traversal_rejected(value):
    with pytest.raises(ValueError):
        validate_relative_path(value)


def test_dedup_and_empty_records():
    accepted, quarantine = deduplicate_normalized_lines(["x", " x ", "", "y"])
    assert accepted == ["x", "y"]
    assert [record["reason"] for record in quarantine] == ["duplicate", "empty"]
    assert all("line_number" in record and "content_sha256" in record for record in quarantine)
    assert quarantine_jsonl(quarantine).endswith(b"\n")


def test_malformed_duplicate_and_empty_utf8_records(tmp_path):
    source = tmp_path / "records.txt"
    source.write_bytes(b"one\n  one  \n\n\xffbad\ntwo\n")
    accepted, rejected = read_utf8_records(source)
    assert accepted == ["one", "two"]
    assert [record["reason"] for record in rejected] == ["duplicate", "empty", "malformed_utf8"]
    ledger = [json.loads(line) for line in quarantine_jsonl(rejected).splitlines()]
    assert [record["reason"] for record in ledger] == ["duplicate", "empty", "malformed_utf8"]


def test_uint32_truncated_valid_and_range():
    with pytest.raises(ValueError, match="truncated"):
        validate_uint32_stream_bytes(b"\x01")
    assert validate_uint32_stream_bytes((1).to_bytes(4, "little")) == [1]
    with pytest.raises(ValueError, match="outside vocabulary"):
        validate_uint32_stream_bytes((5).to_bytes(4, "little"), vocab_size=5)
    with pytest.raises(ValueError, match="truncated uint32 token payload"):
        file_entry("tokens.bin", b"\x01")
    with pytest.raises(ValueError, match="token_count"):
        file_entry("tokens.bin", (1).to_bytes(4, "little"), token_count=2)


def test_mix_deterministic():
    result1 = normalize_mix([{"name": "b", "ratio": 1}, {"name": "a", "ratio": 3}])
    result2 = normalize_mix([{"name": "b", "ratio": 1}, {"name": "a", "ratio": 3}])
    assert result1 == result2
    assert [source["name"] for source in result1] == ["a", "b"]
    assert sum(source["ratio"] for source in result1) == pytest.approx(1.0)


def test_manifest_last_and_complete(tmp_path):
    target = tmp_path / "artifact"
    payload = np.asarray([7, 8], dtype=np.uint32).tobytes()
    entry = file_entry("sources/data.bin", payload)
    artifact_manifest = {**manifest(), "files": [entry]}

    atomic_publish_directory(target, {"sources/data.bin": payload}, artifact_manifest)
    assert (target / "manifest.json").exists()
    assert (target / "sources" / "data.bin").exists()
    assert load_published_artifact(target)["files"] == [entry]

    (target / "sources" / "data.bin").write_bytes(payload[:-1])
    with pytest.raises(ValueError, match="byte count mismatch"):
        load_published_artifact(target)


def test_manifest_rejects_unlisted_and_wrong_hash(tmp_path):
    payload = np.asarray([7, 8], dtype=np.uint32).tobytes()
    entry = file_entry("sources/data.bin", payload)
    with pytest.raises(ValueError, match="manifest/file map mismatch"):
        atomic_publish_directory(
            tmp_path / "missing",
            {"other.bin": payload},
            {**manifest(), "files": [entry]},
        )

    target = tmp_path / "artifact"
    atomic_publish_directory(target, {"sources/data.bin": payload}, {**manifest(), "files": [entry]})
    bad = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    bad["files"][0]["sha256"] = "f" * 64
    (target / "manifest.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_published_artifact(target)


def test_incomplete_directory_not_complete(tmp_path):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="manifest is missing"):
        load_published_artifact(incomplete)


def test_publish_rejects_truncated_token_payload(tmp_path):
    bad_entry = {
        "path": "tokens.bin",
        "kind": "tokens",
        "sha256": "a" * 64,
        "byte_count": 1,
        "token_count": 0,
    }
    with pytest.raises(ValueError, match="truncated"):
        atomic_publish_directory(
            tmp_path / "truncated-artifact",
            {"tokens.bin": b"\x01"},
            {**manifest(), "files": [bad_entry]},
        )
    assert not (tmp_path / "truncated-artifact").exists()


def test_manifest_rejects_mismatched_total_counts(tmp_path):
    payload = np.asarray([7, 8], dtype=np.uint32).tobytes()
    entry = file_entry("sources/data.bin", payload)
    bad_manifest = {
        **manifest(),
        "files": [entry],
        "total_token_count": 99,
        "total_byte_count": len(payload),
    }
    with pytest.raises(ValueError, match="total_token_count"):
        atomic_publish_directory(tmp_path / "bad-count", {"sources/data.bin": payload}, bad_manifest)


def test_split_is_deterministic_disjoint_and_fail_closed(tmp_path):
    source = tmp_path / "packed"
    source.mkdir()
    tokens = np.arange(1, 21, dtype=np.uint32)
    payload = tokens.tobytes()
    (source / "a.bin").write_bytes(payload[:40])
    (source / "b.bin").write_bytes(payload[40:])

    first = split_packed_tokens(source, tmp_path / "split-1", holdout_tokens=4)
    second = split_packed_tokens(source, tmp_path / "split-2", holdout_tokens=4)
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert (first / "train/data.bin").read_bytes() == (second / "train/data.bin").read_bytes()
    assert (first / "holdout/data.bin").read_bytes() == (second / "holdout/data.bin").read_bytes()

    published = load_published_artifact(first)
    assert published["split_policy"] == "contiguous_suffix"
    assert published["holdout_tokens"] == 4
    assert published["train_tokens"] + published["holdout_tokens"] == 20

    train = set(validate_uint32_stream_bytes((first / "train/data.bin").read_bytes()))
    holdout = set(validate_uint32_stream_bytes((first / "holdout/data.bin").read_bytes()))
    assert train & holdout == set()
    assert train | holdout == set(int(value) for value in tokens)

    with pytest.raises(FileExistsError):
        split_packed_tokens(source, first, holdout_tokens=4)
    with pytest.raises(ValueError, match="holdout_tokens"):
        split_packed_tokens(source, tmp_path / "too-large", holdout_tokens=20)
    with pytest.raises(ValueError, match="holdout_tokens"):
        split_packed_tokens(source, tmp_path / "zero", holdout_tokens=0)

    (source / "b.bin").write_bytes(payload[40:-1])
    with pytest.raises(ValueError, match="truncated"):
        split_packed_tokens(source, tmp_path / "tampered", holdout_tokens=4)


def test_repeat_publication_is_deterministic(tmp_path):
    payload = np.asarray([1, 2, 3], dtype=np.uint32).tobytes()
    entry = file_entry("shard.bin", payload)
    source_manifest = {**manifest(), "files": [entry]}
    first = tmp_path / "first"
    second = tmp_path / "second"
    atomic_publish_directory(first, {"shard.bin": payload}, source_manifest)
    atomic_publish_directory(second, {"shard.bin": payload}, source_manifest)
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert (first / "shard.bin").read_bytes() == (second / "shard.bin").read_bytes()


def test_artifact_validation_rejects_symlink_escape(tmp_path):
    payload = np.asarray([7, 8], dtype=np.uint32).tobytes()
    entry = file_entry("sources/data.bin", payload)
    artifact = tmp_path / "artifact"
    atomic_publish_directory(artifact, {"sources/data.bin": payload}, {**manifest(), "files": [entry]})
    outside = tmp_path / "outside.bin"
    outside.write_bytes(payload)
    linked = artifact / "sources" / "data.bin"
    linked.unlink()
    try:
        linked.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink or junction"):
        load_published_artifact(artifact)


def test_artifact_validation_rejects_intermediate_directory_symlink(tmp_path):
    payload = np.asarray([7, 8], dtype=np.uint32).tobytes()
    entry = file_entry("sources/data.bin", payload)
    artifact = tmp_path / "artifact"
    atomic_publish_directory(artifact, {"sources/data.bin": payload}, {**manifest(), "files": [entry]})
    real_sources = artifact / "sources"
    linked_sources = tmp_path / "linked-sources"
    try:
        linked_sources.symlink_to(real_sources, target_is_directory=True)
        real_sources.rename(tmp_path / "detached-sources")
        linked_sources.rename(real_sources)
    except OSError as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink or junction"):
        load_published_artifact(artifact)


def test_publish_rejects_symlinked_destination_parent(tmp_path):
    payload = np.asarray([7, 8], dtype=np.uint32).tobytes()
    entry = file_entry("sources/data.bin", payload)
    artifact_manifest = {**manifest(), "files": [entry]}
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink or junction"):
        atomic_publish_directory(
            linked_parent / "artifact",
            {"sources/data.bin": payload},
            artifact_manifest,
        )


def test_artifact_validation_rejects_symlink_root(tmp_path):
    payload = np.asarray([7, 8], dtype=np.uint32).tobytes()
    entry = file_entry("sources/data.bin", payload)
    real_root = tmp_path / "real"
    atomic_publish_directory(real_root, {"sources/data.bin": payload}, {**manifest(), "files": [entry]})
    linked_root = tmp_path / "linked"
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink or junction"):
        load_published_artifact(linked_root)


def test_artifact_validation_rejects_symlinked_ancestor(tmp_path):
    payload = np.asarray([7, 8], dtype=np.uint32).tobytes()
    entry = file_entry("sources/data.bin", payload)
    real_parent = tmp_path / "real-parent"
    real_root = real_parent / "real"
    atomic_publish_directory(real_root, {"sources/data.bin": payload}, {**manifest(), "files": [entry]})
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink or junction"):
        load_published_artifact(linked_parent / "real")


def test_publish_rejects_absolute_manifest_path(tmp_path):
    payload = np.asarray([7, 8], dtype=np.uint32).tobytes()
    valid_entry = file_entry("outside.bin", payload)
    bad_entry = {**valid_entry, "path": "/outside.bin"}
    with pytest.raises(ValueError, match="path"):
        atomic_publish_directory(
            tmp_path / "absolute-artifact",
            {"outside.bin": payload},
            {**manifest(), "files": [bad_entry]},
        )
