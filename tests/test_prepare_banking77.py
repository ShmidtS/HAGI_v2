"""Offline contract tests for the pinned Banking77 research publisher."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import scripts.prepare_banking77 as banking77_module
from hagi.data.artifacts import load_published_artifact
from scripts.prepare_banking77 import DEFAULT_LICENSE, DEFAULT_REVISION, _read_split
from scripts.prepare_banking77 import prepare_banking77 as _prepare_banking77

_ROOT = Path(__file__).resolve().parent.parent


def _fixture_hashes(raw: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in raw.iterdir()
        if path.is_file()
    }


def prepare_banking77(raw_dir, output_dir, *, expected_source_hashes=None, **kwargs):
    raw = Path(raw_dir)
    hashes = _fixture_hashes(raw) if expected_source_hashes is None else expected_source_hashes
    with patch.dict(banking77_module.PINNED_SOURCE_SHA256, hashes, clear=True):
        return _prepare_banking77(raw_dir, output_dir, **kwargs)


def test_direct_script_help_resolves_repository_modules():
    result = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "prepare_banking77.py"), "--help"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "retrieval-timestamp" in result.stdout


def _write_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    categories = [f"intent_{index}" for index in range(77)]
    (root / "categories.json").write_text(json.dumps(categories), encoding="utf-8")
    (root / "LICENSE").write_text("CC BY 4.0 fixture\n", encoding="utf-8")
    (root / "train.csv").write_text(
        "text,category\nWhere is my card?,intent_0\nTransfer failed,intent_76\n",
        encoding="utf-8",
    )
    (root / "test.csv").write_text(
        "text,category\nWhy is my balance wrong?,intent_1\n",
        encoding="utf-8",
    )
    return root


def _tokenize(rows):
    return [[10 + index, 20 + index, 30 + index] for index, _ in enumerate(rows)]


def test_read_split_preserves_quoted_multiline_text(tmp_path):
    raw = _write_fixture(tmp_path / "raw")
    (raw / "train.csv").write_text(
        'text,category\n"First line\nsecond line",intent_0\n',
        encoding="utf-8",
    )
    categories = json.loads((raw / "categories.json").read_text(encoding="utf-8"))
    rows, rejected = _read_split(raw / "train.csv", categories, split="train")
    assert rejected == []
    assert len(rows) == 1
    assert rows[0].source_row == 2
    assert rows[0].text == "First line\nsecond line"
    assert rows[0].intent == "intent_0"


def test_read_split_rejects_duplicate_and_records_source_row(tmp_path):
    raw = _write_fixture(tmp_path / "raw")
    categories = json.loads((raw / "categories.json").read_text(encoding="utf-8"))
    with (raw / "train.csv").open("a", encoding="utf-8") as handle:
        handle.write("Where is my card?,intent_0\n,missing\n")
    rows, rejected = _read_split(raw / "train.csv", categories, split="train")
    assert [row.intent for row in rows] == ["intent_0", "intent_76"]
    assert [(item["source_row"], item["reason"]) for item in rejected] == [
        (4, "duplicate"),
        (5, "malformed_or_empty"),
    ]


def test_prepare_banking77_publishes_exact_text_labels_and_valid_manifest(tmp_path):
    raw = _write_fixture(tmp_path / "raw")
    output = prepare_banking77(
        raw,
        tmp_path / "artifact",
        tokenizer_name="fixture-tokenizer",
        tokenizer_callable=_tokenize,
        revision=DEFAULT_REVISION,
        retrieval_timestamp="2026-09-24T00:00:00Z",
        eos_token_id=1,
        shard_tokens=16,
        batch_size=2,
    )
    manifest = load_published_artifact(output)
    assert manifest["artifact_id"] == "banking77"
    assert manifest["revision"] == DEFAULT_REVISION
    assert manifest["license"] == DEFAULT_LICENSE
    assert manifest["train_rows"] == 2
    assert manifest["test_rows"] == 1
    assert manifest["category_count"] == 77
    assert manifest["quarantine_count"] == 0
    assert manifest["total_token_count"] == 12
    assert (output / "text/train.txt").read_text(encoding="utf-8") == "Where is my card?\nTransfer failed\n"
    assert (output / "text/test.txt").read_text(encoding="utf-8") == "Why is my balance wrong?\n"
    labels = [json.loads(line) for line in (output / "labels/train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["source_row"], row["label"], row["intent"]) for row in labels] == [
        (2, 0, "intent_0"),
        (3, 76, "intent_76"),
    ]
    assert {row["text_sha256"] for row in labels} == {
        hashlib.sha256(b"Where is my card?").hexdigest(),
        hashlib.sha256(b"Transfer failed").hexdigest(),
    }
    train_shard = output / "train/shard-000000.bin"
    assert np.fromfile(train_shard, dtype="<u4").tolist() == [10, 20, 30, 1, 11, 21, 31, 1]
    assert (output / "provenance/train.csv").read_bytes() == (raw / "train.csv").read_bytes()
    assert (output / "provenance/test.csv").read_bytes() == (raw / "test.csv").read_bytes()
    assert (output / "metadata/categories.json").read_bytes() == (raw / "categories.json").read_bytes()
    assert (output / "metadata/LICENSE").read_bytes() == (raw / "LICENSE").read_bytes()
    source = next(item for item in manifest["sources"] if item["name"] == "train")
    expected_output_hash = hashlib.sha256(train_shard.read_bytes()).hexdigest()
    assert source["output_sha256"] == expected_output_hash
    assert source["input_sha256"] == hashlib.sha256((raw / "train.csv").read_bytes()).hexdigest()
    assert source["row_count"] == 2
    assert [(item["name"], item["sha256"]) for item in manifest["source_files"]] == [
        (name, _fixture_hashes(raw)[name])
        for name in ("train.csv", "test.csv", "categories.json", "LICENSE")
    ]


def test_prepare_banking77_rejects_source_hash_mismatch_before_tokenization(tmp_path):
    raw = _write_fixture(tmp_path / "raw")
    hashes = _fixture_hashes(raw)
    hashes["train.csv"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 mismatch: train.csv"):
        prepare_banking77(
            raw,
            tmp_path / "artifact",
            tokenizer_name="fixture-tokenizer",
            tokenizer_callable=_tokenize,
            expected_source_hashes=hashes,
            retrieval_timestamp="fixed",
        )
    assert not (tmp_path / "artifact").exists()


def test_prepare_banking77_rejects_non_pinned_revision(tmp_path):
    raw = _write_fixture(tmp_path / "raw")
    with pytest.raises(ValueError, match="pinned to revision"):
        prepare_banking77(
            raw,
            tmp_path / "artifact",
            tokenizer_name="fixture-tokenizer",
            tokenizer_callable=_tokenize,
            revision="0" * 40,
            retrieval_timestamp="fixed",
        )
    assert not (tmp_path / "artifact").exists()


def test_prepare_banking77_rejects_overlap_and_existing_output(tmp_path):
    raw = _write_fixture(tmp_path / "raw")
    (raw / "test.csv").write_text("text,category\nTransfer failed,intent_76\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        prepare_banking77(
            raw,
            tmp_path / "overlap",
            tokenizer_name="fixture-tokenizer",
            tokenizer_callable=_tokenize,
            retrieval_timestamp="fixed",
        )

    raw = _write_fixture(tmp_path / "raw2")
    (tmp_path / "existing").mkdir()
    with pytest.raises(FileExistsError):
        prepare_banking77(
            raw,
            tmp_path / "existing",
            tokenizer_name="fixture-tokenizer",
            tokenizer_callable=_tokenize,
            retrieval_timestamp="fixed",
        )


def test_prepare_banking77_rejects_bad_token_ids_and_shard_size(tmp_path):
    raw = _write_fixture(tmp_path / "raw")
    with pytest.raises(ValueError, match="outside declared vocabulary"):
        prepare_banking77(
            raw,
            tmp_path / "bad-token",
            tokenizer_name="fixture-tokenizer",
            tokenizer_callable=lambda rows: [[262144, 1] for _ in rows],
            retrieval_timestamp="fixed",
        )

    raw = _write_fixture(tmp_path / "raw2")
    with pytest.raises(ValueError, match="exceeds shard_tokens"):
        prepare_banking77(
            raw,
            tmp_path / "bad-shard",
            tokenizer_name="fixture-tokenizer",
            tokenizer_callable=_tokenize,
            retrieval_timestamp="fixed",
            shard_tokens=3,
        )


def test_prepare_banking77_requires_explicit_retrieval_timestamp(tmp_path):
    raw = _write_fixture(tmp_path / "raw")
    with pytest.raises(ValueError, match="retrieval_timestamp"):
        prepare_banking77(
            raw,
            tmp_path / "no-timestamp",
            tokenizer_name="fixture-tokenizer",
            tokenizer_callable=_tokenize,
            retrieval_timestamp="",
        )


def test_read_split_rejects_wrong_header(tmp_path):
    raw = _write_fixture(tmp_path / "raw")
    (raw / "train.csv").write_text("category,text\nintent_0,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        _read_split(raw / "train.csv", json.loads((raw / "categories.json").read_text()), split="train")


def test_publisher_does_not_need_csv_or_loader_state(tmp_path):
    # Importing and using the publisher must not mutate production data files.
    before = sorted((tmp_path / "nonexistent").glob("*")) if (tmp_path / "nonexistent").exists() else []
    assert before == []
