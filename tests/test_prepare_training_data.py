"""Offline acceptance tests for bounded data acquisition and preparation."""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest

import scripts.prepare_training_data as pipeline
from scripts.prepare_training_data import acquire_source, main, prepare_data, validate_output


def tokenizer_for(row_tokens):
    def tokenize(rows):
        return [list(row_tokens) for _ in rows]

    return tokenize


def test_acquire_local_bounded_and_hash_checked(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    output = tmp_path / "nested" / "copy.bin"

    acquire_source(str(source), output, 7, digest, 1)
    assert output.read_bytes() == b"payload"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        acquire_source(str(source), tmp_path / "bad.bin", 7, "0" * 64, 1)
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        acquire_source(str(source), tmp_path / "too-big.bin", 6, digest, 1)


def test_remote_requires_hash_and_rejects_url_credentials(tmp_path):
    with pytest.raises(ValueError, match="requires expected_sha256"):
        acquire_source("https://example.test/data", tmp_path / "data", 10, None, 1)
    with pytest.raises(ValueError, match="credentials"):
        acquire_source("https://user:pass@example.test/data", tmp_path / "data", 10, "0" * 64, 1)


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, content_length: str | None = None):
        super().__init__(payload)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_remote_acquire_bounds_hash_and_temp_cleanup(monkeypatch, tmp_path):
    payload = b"remote"
    digest = hashlib.sha256(payload).hexdigest()

    def fake_open(request, timeout):
        assert request.full_url == "https://example.test/data"
        assert timeout == 3
        return FakeResponse(payload, str(len(payload)))

    monkeypatch.setattr(pipeline, "_open_no_redirect", fake_open)
    output = tmp_path / "download.bin"
    acquire_source("https://example.test/data", output, 6, digest, 3)
    assert output.read_bytes() == payload
    assert not list(tmp_path.glob(".download.bin.*"))

    monkeypatch.setattr(
        pipeline,
        "_open_no_redirect",
        lambda request, timeout: FakeResponse(payload, "99"),
    )
    with pytest.raises(ValueError, match="content-length"):
        acquire_source("https://example.test/data", tmp_path / "large.bin", 6, digest, 3)
    assert not (tmp_path / "large.bin").exists()
    assert not list(tmp_path.glob(".large.bin.*"))

    monkeypatch.setattr(
        pipeline,
        "_open_no_redirect",
        lambda request, timeout: FakeResponse(payload, "not-an-integer"),
    )
    with pytest.raises(ValueError, match="invalid Content-Length"):
        acquire_source("https://example.test/data", tmp_path / "bad-header.bin", 6, digest, 3)
    assert not (tmp_path / "bad-header.bin").exists()
    assert not list(tmp_path.glob(".bad-header.bin.*"))

    monkeypatch.setattr(
        pipeline,
        "_open_no_redirect",
        lambda request, timeout: FakeResponse(payload, "-1"),
    )
    with pytest.raises(ValueError, match="invalid Content-Length"):
        acquire_source("https://example.test/data", tmp_path / "negative-header.bin", 6, digest, 3)
    assert not (tmp_path / "negative-header.bin").exists()
    assert not list(tmp_path.glob(".negative-header.bin.*"))

    monkeypatch.setattr(
        pipeline,
        "_open_no_redirect",
        lambda request, timeout: FakeResponse(payload, None),
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        acquire_source("https://example.test/data", tmp_path / "bad.bin", 6, "0" * 64, 3)
    assert not (tmp_path / "bad.bin").exists()
    assert not list(tmp_path.glob(".bad.bin.*"))


def test_redirect_handler_fails_closed():
    handler = pipeline._NoRedirect()
    with pytest.raises(pipeline.HTTPError, match="redirects are not allowed"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://other.test/data")


def test_prepare_quarantine_shards_manifest_and_validation(tmp_path):
    source = tmp_path / "records.txt"
    source.write_bytes(b" alpha \nalpha\n\n\xffbroken\nbeta\n")
    output = tmp_path / "artifact"

    prepare_data(
        source,
        output,
        "fixture-tokenizer-v1",
        tokenizer_callable=tokenizer_for([3, 4]),
        source_name="fixture",
        dataset="unit",
        revision="r1",
        license_name="CC0",
        retrieval_timestamp="2026-09-24T00:00:00Z",
        eos_token_id=1,
        vocab_size=8,
        shard_tokens=4,
        batch_size=1,
    )

    manifest = validate_output(output)
    assert manifest["quarantine_count"] == 3
    assert manifest["total_token_count"] == 6  # two docs × (two ids + EOS)
    token_files = [entry for entry in manifest["files"] if entry["kind"] == "tokens"]
    assert manifest["sources"][0]["output_sha256"] == hashlib.sha256(
        b"".join(
            (output / entry["path"]).read_bytes()
            for entry in token_files
        )
    ).hexdigest()
    quarantine_entries = [entry for entry in manifest["files"] if entry["kind"] == "jsonl"]
    assert [entry["token_count"] for entry in quarantine_entries] == [0]
    assert [entry["token_count"] for entry in token_files] == [3, 3]
    packed_shards = [np.fromfile(output / entry["path"], dtype=np.uint32).tolist() for entry in token_files]
    assert packed_shards == [[3, 4, 1], [3, 4, 1]]
    quarantine = [
        json.loads(line)
        for line in (output / "quarantine" / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["reason"] for record in quarantine] == ["duplicate", "empty", "malformed_utf8"]
    assert all("content_sha256" in record and "line_number" in record for record in quarantine)


def test_prepare_rejects_empty_invalid_range_and_oversized_document(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"\n\n")
    with pytest.raises(ValueError, match="no valid"):
        prepare_data(empty, tmp_path / "empty-artifact", "x", tokenizer_callable=tokenizer_for([1]))

    source = tmp_path / "records.txt"
    source.write_text("one\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside declared vocabulary"):
        prepare_data(
            source,
            tmp_path / "range-artifact",
            "x",
            tokenizer_callable=tokenizer_for([9]),
            vocab_size=8,
        )

    with pytest.raises(ValueError, match="exceeds shard_tokens"):
        prepare_data(
            source,
            tmp_path / "large-doc-artifact",
            "x",
            tokenizer_callable=tokenizer_for([1, 2, 3]),
            shard_tokens=3,
        )

    for kwargs, match in (
        ({"batch_size": 0}, "batch_size"),
        ({"eos_token_id": "1"}, "eos_token_id"),
        ({"vocab_size": "8"}, "vocab_size"),
        ({"shard_tokens": 2.0}, "shard_tokens"),
        ({"max_input_bytes": 0}, "max_input_bytes"),
    ):
        with pytest.raises(ValueError, match=match):
            prepare_data(
                source,
                tmp_path / f"invalid-{match}",
                "x",
                tokenizer_callable=tokenizer_for([1]),
                **kwargs,
            )

    with pytest.raises(ValueError, match="source_url"):
        prepare_data(
            source,
            tmp_path / "credential-url-artifact",
            "x",
            tokenizer_callable=tokenizer_for([1]),
            source_url="https://user:secret@example.test/dataset",
        )


def test_repeat_prepare_is_byte_identical(tmp_path):
    source = tmp_path / "records.txt"
    source.write_text("alpha\nbeta\n", encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"
    for output in (first, second):
        prepare_data(
            source,
            output,
            "fixture-tokenizer-v1",
            tokenizer_callable=tokenizer_for([2, 5]),
            source_name="fixture",
            dataset="unit",
            revision="r1",
            license_name="CC0",
            retrieval_timestamp="fixed",
            vocab_size=8,
            shard_tokens=10,
        )
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    for path in sorted(first.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            relative = path.relative_to(first)
            assert path.read_bytes() == (second / relative).read_bytes()


def test_validate_rejects_corrupt_manifest_and_stream(tmp_path):
    source = tmp_path / "records.txt"
    source.write_text("alpha\n", encoding="utf-8")
    output = tmp_path / "artifact"
    prepare_data(
        source,
        output,
        "fixture",
        tokenizer_callable=tokenizer_for([2]),
        source_name="fixture",
        dataset="unit",
        revision="r1",
        license_name="CC0",
        retrieval_timestamp="fixed",
        vocab_size=8,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = "f" * 64
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_output(output)


def test_official_gigatoken_adapter_uses_native_batch_list_api(monkeypatch):
    class NativeTokenizer:
        def __init__(self, name):
            assert name == "fixture-tokenizer"
            self.as_hf_called = False

        def as_hf(self):
            self.as_hf_called = True
            raise AssertionError("as_hf().encode_batch is not the installed native API")

        def encode_batch_list(self, rows):
            assert rows == ["alpha", "beta"]
            return [[3, 4], [5, 6]]

    fake_module = type("FakeGigatoken", (), {"Tokenizer": NativeTokenizer})
    monkeypatch.setitem(__import__("sys").modules, "gigatoken", fake_module)

    encoder = pipeline._official_gigatoken_batch("fixture-tokenizer")
    assert encoder(["alpha", "beta"]) == [[3, 4], [5, 6]]


def test_official_gigatoken_adapter_rejects_missing_batch_list_api(monkeypatch):
    class BrokenTokenizer:
        def __init__(self, name):
            pass

    fake_module = type("FakeGigatoken", (), {"Tokenizer": BrokenTokenizer})
    monkeypatch.setitem(__import__("sys").modules, "gigatoken", fake_module)
    with pytest.raises(RuntimeError, match="encode_batch_list"):
        pipeline._official_gigatoken_batch("fixture-tokenizer")


def test_cli_prepare_and_validate(tmp_path):
    source = tmp_path / "records.txt"
    source.write_text("alpha\n", encoding="utf-8")
    # Use a tiny injected-equivalent tokenizer through the library boundary;
    # the CLI's real tokenizer is tested separately when gigatoken is present.
    assert main(["acquire", str(source), str(tmp_path / "copy.txt"), "--max-bytes", "100"]) == 0
    assert (tmp_path / "copy.txt").read_text(encoding="utf-8") == "alpha\n"
    assert callable(prepare_data)
    assert callable(validate_output)
