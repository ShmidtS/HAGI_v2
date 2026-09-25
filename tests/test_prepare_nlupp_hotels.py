"""Offline contract tests for the pinned NLU++ HOTELS publisher."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hagi.data.artifacts import load_published_artifact
from scripts import prepare_nlupp_hotels as subject
from scripts.prepare_nlupp_hotels import _banking77_text_hashes, prepare_nlupp_hotels

_ROOT = Path(__file__).resolve().parent.parent


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _source_spec(root: Path, relative: str) -> dict[str, object]:
    payload = (root / Path(relative).name).read_bytes()
    return {
        "upstream_path": relative,
        "git_blob_sha1": _git_blob_sha1(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
    }


def _ontology() -> dict[str, object]:
    return {
        "intents": {
            "book": {"description": "book", "domain": ["hotels"]},
            "acknowledge": {"description": "acknowledge", "domain": ["general"]},
        },
        "slots": {
            "number": {"description": "number", "domain": ["hotels"]},
            "place": {"description": "place", "domain": ["general"]},
        },
    }


def _row(text: str, *, with_slot: bool = False) -> dict[str, object]:
    row: dict[str, object] = {"text": text, "intents": ["book"]}
    if with_slot:
        start = text.rfind("hotel")
        row["slots"] = {
            "place": {"text": "hotel", "span": [start, start + len("hotel")], "value": "hotel"}
        }
    return row


def _write_fixture(root: Path) -> Path:
    raw = root / "raw"
    raw.mkdir(parents=True)
    folds = {
        0: [_row("book hotel", with_slot=True)],
        1: [_row("book train")],
        2: [_row("book plane")],
        3: [
            _row("book hotel", with_slot=True),
            _row("book train"),
            _row("book plane"),
            _row("book banking only"),
        ],
    }
    for fold, rows in folds.items():
        (raw / f"fold{fold}.json").write_text(json.dumps(rows), encoding="utf-8")
    (raw / "ontology.json").write_text(json.dumps(_ontology()), encoding="utf-8")
    (raw / "LICENSE").write_text("CC BY 4.0 fixture\n", encoding="utf-8")
    (raw / "data_loader.py").write_text("# pinned loader fixture\n", encoding="utf-8")
    return raw


def _fixture_specs(raw: Path) -> dict[str, dict[str, object]]:
    return {
        "ontology.json": _source_spec(raw, "nlupp/data/ontology.json"),
        "LICENSE": _source_spec(raw, "LICENSE"),
        "data_loader.py": _source_spec(raw, "nlupp/data_loader.py"),
        **{
            f"fold{fold}.json": _source_spec(raw, f"nlupp/data/hotels/fold{fold}.json")
            for fold in range(4)
        },
    }


def _tokenize(rows: list[str]) -> list[list[int]]:
    return [[10 + index, 20 + index, 30 + index] for index, _ in enumerate(rows)]


def _prepare(
    raw: Path,
    output: Path,
    *,
    banking_hashes: set[str] | None = None,
    specs: dict[str, dict[str, object]] | None = None,
) -> Path:
    banking_hashes = banking_hashes or set()
    specs = specs or _fixture_specs(raw)
    with (
        patch.object(subject, "FOLD_COUNT", 4),
        patch.object(subject, "TRAIN_FOLDS", (2, 3)),
        patch.object(subject, "EXPECTED_RAW_ROWS", 7),
        patch.object(subject, "EXPECTED_TRAIN_ROWS", 1),
        patch.object(subject, "EXPECTED_TEST_ROWS", 2),
        patch.object(subject, "EXPECTED_TRAIN_TEST_OVERLAP", 2),
        patch.object(subject, "EXPECTED_DUPLICATES", 1),
        patch.object(subject, "EXPECTED_CROSS_SOURCE_OVERLAP", 1),
        patch.object(subject, "EXPECTED_QUARANTINE_COUNT", 4),
        patch.object(subject, "EXPECTED_INTENTS", 1),
        patch.object(subject, "EXPECTED_SLOTS", 1),
        patch.dict(subject.PINNED_SOURCE_FILES, specs, clear=True),
        patch.object(subject, "_banking77_text_hashes", return_value=banking_hashes),
    ):
        return prepare_nlupp_hotels(
            raw,
            output,
            output.parent / "banking77-placeholder",
            tokenizer_name="fixture-tokenizer",
            tokenizer_callable=_tokenize,
            retrieval_timestamp="2026-09-24T00:00:00Z",
            eos_token_id=1,
            shard_tokens=16,
            batch_size=2,
        )


def test_direct_script_help_resolves_repository_modules():
    result = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "prepare_nlupp_hotels.py"), "--help"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "retrieval-timestamp" in result.stdout
    assert "banking77-artifact" in result.stdout


def test_publisher_reproduces_large_split_and_quarantines_leakage(tmp_path: Path):
    raw = _write_fixture(tmp_path / "raw")
    banking_text = "book banking only"
    output = _prepare(raw, tmp_path / "artifact", banking_hashes={hashlib.sha256(banking_text.encode()).hexdigest()})

    manifest = load_published_artifact(output)
    assert manifest["artifact_id"] == "nlupp_hotels"
    assert manifest["split_policy"].startswith("NLU++ large regime fold pair 0,1")
    assert manifest["raw_rows"] == 7
    assert manifest["train_rows"] == 1
    assert manifest["test_rows"] == 2
    assert manifest["quarantine_count"] == 4
    assert manifest["cross_source_overlap_count"] == 1
    assert manifest["train_test_overlap_count"] == 2
    assert manifest["duplicate_count"] == 1
    assert manifest["total_token_count"] == 12
    assert (output / "text/train.txt").read_text(encoding="utf-8") == "book plane\n"
    assert (output / "text/test.txt").read_text(encoding="utf-8") == "book hotel\nbook train\n"

    train_labels = [
        json.loads(line) for line in (output / "labels/train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert train_labels == [
        {
            "intents": ["book"],
            "slots": {},
            "source_fold": 2,
            "source_row": 0,
            "text_sha256": hashlib.sha256(b"book plane").hexdigest(),
        }
    ]
    test_labels = [
        json.loads(line) for line in (output / "labels/test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["source_fold"], row["source_row"], row["slots"]) for row in test_labels] == [
        (0, 0, {"place": {"span": [5, 10], "text": "hotel", "value": "hotel"}}),
        (1, 0, {}),
    ]
    quarantine = [
        json.loads(line)
        for line in (output / "quarantine/records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["reason"] for row in quarantine] == [
        "train_test_overlap",
        "train_test_overlap",
        "duplicate_within_train",
        "banking77_cross_source_overlap",
    ]
    assert all(len(row["content_sha256"]) == 64 for row in quarantine)
    assert np.fromfile(output / "train/shard-000000.bin", dtype="<u4").tolist() == [10, 20, 30, 1]
    assert np.fromfile(output / "test/shard-000000.bin", dtype="<u4").tolist() == [
        10,
        20,
        30,
        1,
        11,
        21,
        31,
        1,
    ]
    assert {entry["path"] for entry in manifest["source_files"]} == set(_fixture_specs(raw))
    for entry in manifest["source_files"]:
        assert entry["sha256"] == _fixture_specs(raw)[str(entry["path"])]["sha256"]
    assert (output / "provenance/fold0.json").read_bytes() == (raw / "fold0.json").read_bytes()
    assert (output / "provenance/ontology.json").read_bytes() == (raw / "ontology.json").read_bytes()
    assert (output / "provenance/data_loader.py").read_bytes() == (raw / "data_loader.py").read_bytes()
    assert (output / "provenance/LICENSE").read_bytes() == (raw / "LICENSE").read_bytes()


def test_source_identity_mismatch_fails_before_tokenization(tmp_path: Path):
    raw = _write_fixture(tmp_path / "raw")
    specs = _fixture_specs(raw)
    specs["fold0.json"] = {**specs["fold0.json"], "sha256": "0" * 64}
    with pytest.raises(ValueError, match="sha256 mismatch: fold0.json"):
        _prepare(raw, tmp_path / "bad", specs=specs)
    assert not (tmp_path / "bad").exists()


def test_git_blob_mismatch_fails_before_tokenization(tmp_path: Path):
    raw = _write_fixture(tmp_path / "raw")
    specs = _fixture_specs(raw)
    specs["fold1.json"] = {**specs["fold1.json"], "git_blob_sha1": "0" * 40}
    with pytest.raises(ValueError, match="git blob mismatch: fold1.json"):
        _prepare(raw, tmp_path / "bad", specs=specs)
    assert not (tmp_path / "bad").exists()


def test_bad_slot_span_fails_closed(tmp_path: Path):
    raw = _write_fixture(tmp_path / "raw")
    payload = json.loads((raw / "fold0.json").read_text(encoding="utf-8"))
    payload[0]["slots"]["place"]["span"] = [0, 5]
    (raw / "fold0.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="slot span mismatch"):
        _prepare(raw, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_banking77_artifact_manifest_is_hash_pinned(tmp_path: Path):
    root = tmp_path / "banking"
    root.mkdir()
    (root / "manifest.json").write_bytes(b"not the pinned Banking77 manifest")
    with pytest.raises(ValueError, match="manifest sha256 mismatch"):
        _banking77_text_hashes(root)


def test_embedded_newline_fails_closed(tmp_path: Path):
    raw = _write_fixture(tmp_path / "raw")
    payload = json.loads((raw / "fold0.json").read_text(encoding="utf-8"))
    payload[0]["text"] = "book" + chr(10) + "hotel"
    (raw / "fold0.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="embedded newline"):
        _prepare(raw, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_wrong_schema_fails_closed(tmp_path: Path):
    raw = _write_fixture(tmp_path / "raw")
    payload = json.loads((raw / "fold0.json").read_text(encoding="utf-8"))
    payload[0]["unexpected"] = True
    (raw / "fold0.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="record keys"):
        _prepare(raw, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_ontology_rejects_non_hotel_label(tmp_path: Path):
    raw = _write_fixture(tmp_path / "raw")
    payload = json.loads((raw / "ontology.json").read_text(encoding="utf-8"))
    payload["intents"]["book"]["domain"] = ["banking"]
    (raw / "ontology.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="intent 'book'"):
        _prepare(raw, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_existing_output_is_never_overwritten(tmp_path: Path):
    raw = _write_fixture(tmp_path / "raw")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _prepare(raw, output)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
