#!/usr/bin/env python3
"""Publish a pinned Banking77 artifact for HAGI research experiments.

This is an opt-in, offline publication step. It does not change the production
packed-data loader and it never accesses the network. The caller must provide
already acquired CSV files and an explicit source-retrieval timestamp. The
output contains exact split text, finite-option labels, native Gigatoken
packs, and a fail-closed manifest with upstream SHA-256 provenance.

The official source is the PolyAI-LDN/task-specific-datasets repository:

    https://github.com/PolyAI-LDN/task-specific-datasets

This script is intentionally separate from the generic text preparation path:
Banking77 labels must stay attached to their source rows, while common-reference
LM evaluation must score the exact text without label strings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _path in (_SRC, _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from hagi.data.artifacts import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    atomic_publish_directory,
    file_entry,
    quarantine_jsonl,
    sha256_bytes,
)
from scripts.prepare_training_data import default_tokenizer  # noqa: E402

DEFAULT_EOS_ID = 1
DEFAULT_VOCAB_SIZE = 262_144
DEFAULT_TOKENIZER = "google/gemma-4-E2B-it"
DEFAULT_DATASET = "PolyAI-LDN/task-specific-datasets/banking_data"
DEFAULT_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
DEFAULT_LICENSE = "CC-BY-4.0"
DEFAULT_SOURCE_BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"
PINNED_SOURCE_SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    "LICENSE": "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
}


class SplitRow:
    """One validated source row before tokenization."""

    __slots__ = ("source_row", "text", "label", "intent")

    def __init__(self, source_row: int, text: str, label: int, intent: str) -> None:
        self.source_row = source_row
        self.text = text
        self.label = label
        self.intent = intent


def _sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _verify_source_hashes(raw: Path, expected: Mapping[str, str]) -> None:
    expected_names = set(PINNED_SOURCE_SHA256)
    if set(expected) != expected_names:
        raise ValueError(f"expected source hashes must cover exactly {sorted(expected_names)}")
    for name in sorted(expected_names):
        digest = expected[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise ValueError(f"expected source hash for {name!r} must be a SHA-256 digest")
        path = raw / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256_file(path)
        if actual != digest.lower():
            raise ValueError(f"sha256 mismatch: {name}")


def _source_file_metadata(raw: Path, revision: str, expected: Mapping[str, str]) -> list[dict[str, Any]]:
    metadata = []
    for name in ("train.csv", "test.csv", "categories.json", "LICENSE"):
        relative_path = f"banking_data/{name}" if name in {"train.csv", "test.csv", "categories.json"} else name
        metadata.append(
            {
                "name": name,
                "url": f"{DEFAULT_SOURCE_BASE}/{revision}/{relative_path}",
                "sha256": expected[name].lower(),
                "byte_count": (raw / name).stat().st_size,
            }
        )
    return metadata


def _read_categories(path: Path) -> list[str]:
    try:
        categories = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("categories.json is not valid UTF-8 JSON") from exc
    if not isinstance(categories, list) or not categories or any(
        not isinstance(value, str) or not value for value in categories
    ):
        raise ValueError("categories must be a non-empty list of non-empty strings")
    if len(set(categories)) != len(categories):
        raise ValueError("categories must be unique")
    return categories


def _read_split(
    path: Path,
    categories: list[str],
    *,
    split: str,
) -> tuple[list[SplitRow], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{split}.csv is not strict UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if reader.fieldnames != ["text", "category"]:
        raise ValueError(f"{split}.csv header must be exactly text,category")
    label_by_intent = {intent: index for index, intent in enumerate(categories)}
    rows: list[SplitRow] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_row, record in enumerate(reader, start=2):
        value = record.get("text")
        intent = record.get("category")
        if value is None or intent is None or not value.strip() or intent not in label_by_intent:
            rejected.append(
                {
                    "split": split,
                    "source_row": source_row,
                    "reason": "malformed_or_empty",
                    "content_sha256": sha256_bytes((value or "").encode("utf-8")),
                }
            )
            continue
        digest = sha256_bytes(value.encode("utf-8"))
        if digest in seen:
            rejected.append(
                {
                    "split": split,
                    "source_row": source_row,
                    "reason": "duplicate",
                    "content_sha256": digest,
                }
            )
            continue
        seen.add(digest)
        rows.append(SplitRow(source_row, value, label_by_intent[intent], intent))
    if not rows:
        raise ValueError(f"{split}.csv has no valid rows")
    return rows, rejected


def _encode_rows(
    rows: list[SplitRow],
    tokenizer_callable: Callable[[list[str]], list[list[int]]] | None,
    tokenizer_name: str,
    *,
    batch_size: int,
) -> list[list[int]]:
    encoder = tokenizer_callable or default_tokenizer(tokenizer_name)
    encoded: list[list[int]] = []
    texts = [row.text for row in rows]
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        result = encoder(batch)
        if len(result) != len(batch):
            raise ValueError("tokenizer returned a different number of rows")
        for row, values in zip(rows[start : start + batch_size], result, strict=True):
            ids = [int(value) for value in values]
            if not ids:
                raise ValueError(f"tokenizer returned an empty row at source row {row.source_row}")
            if min(ids) < 0 or max(ids) >= DEFAULT_VOCAB_SIZE:
                raise ValueError(f"token id outside declared vocabulary at source row {row.source_row}")
            encoded.append(ids)
    return encoded


def _pack_tokens(rows: list[SplitRow], encoded: list[list[int]], eos_token_id: int, shard_tokens: int) -> dict[str, bytes]:
    if type(eos_token_id) is not int or eos_token_id < 0 or eos_token_id >= DEFAULT_VOCAB_SIZE:
        raise ValueError("eos_token_id must be in [0, 262144)")
    if type(shard_tokens) is not int or shard_tokens < 2:
        raise ValueError("shard_tokens must be >= 2")
    shards: dict[str, bytes] = {}
    current = bytearray()
    current_count = 0
    shard_index = 0
    for ids in encoded:
        required = len(ids) + 1
        if required > shard_tokens:
            raise ValueError("one document exceeds shard_tokens")
        if current_count and current_count + required > shard_tokens:
            shards[f"tokens/shard-{shard_index:06d}.bin"] = bytes(current)
            shard_index += 1
            current = bytearray()
            current_count = 0
        current.extend(np.asarray([*ids, eos_token_id], dtype="<u4").tobytes())
        current_count += required
    if current:
        shards[f"tokens/shard-{shard_index:06d}.bin"] = bytes(current)
    return shards


def _label_jsonl(rows: list[SplitRow]) -> bytes:
    records = [
        {
            "source_row": row.source_row,
            "label": row.label,
            "intent": row.intent,
            "text_sha256": sha256_bytes(row.text.encode("utf-8")),
        }
        for row in rows
    ]
    return ("\n".join(json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records) + "\n").encode(
        "utf-8"
    )


def _text_payload(rows: list[SplitRow]) -> bytes:
    return ("\n".join(row.text for row in rows) + "\n").encode("utf-8")


def _source_metadata(
    *,
    name: str,
    rows: int,
    input_path: Path,
    token_shards: dict[str, bytes],
    tokenizer_name: str,
    revision: str,
    retrieval_timestamp: str,
    source_url: str,
    ratio: float,
) -> dict[str, Any]:
    output_hash = hashlib.sha256()
    for path in sorted(token_shards):
        output_hash.update(token_shards[path])
    return {
        "name": name,
        "ratio": ratio,
        "dataset": DEFAULT_DATASET,
        "revision": revision,
        "license": DEFAULT_LICENSE,
        "source_url": source_url,
        "retrieval_timestamp": retrieval_timestamp,
        "tokenizer_version": tokenizer_name,
        "filter_policy_version": "banking77-csv-strict-v1",
        "dedup_policy_version": "exact-text-sha256-v1",
        "byte_count": input_path.stat().st_size,
        "token_count": sum(len(payload) // 4 for payload in token_shards.values()),
        "input_sha256": _sha256_file(input_path),
        "output_sha256": output_hash.hexdigest(),
        "row_count": rows,
    }


def prepare_banking77(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    tokenizer_callable: Callable[[list[str]], list[list[int]]] | None = None,
    revision: str = DEFAULT_REVISION,
    retrieval_timestamp: str,
    eos_token_id: int = DEFAULT_EOS_ID,
    shard_tokens: int = 100_000,
    batch_size: int = 256,
) -> Path:
    """Publish train/test text, labels, native token packs and manifest."""
    raw = Path(raw_dir)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(target)
    if not retrieval_timestamp or not isinstance(retrieval_timestamp, str):
        raise ValueError("retrieval_timestamp must be a non-empty explicit string")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if revision != DEFAULT_REVISION:
        raise ValueError(f"Banking77 publisher is pinned to revision {DEFAULT_REVISION}")
    hashes = dict(PINNED_SOURCE_SHA256)
    _verify_source_hashes(raw, hashes)
    categories = _read_categories(raw / "categories.json")
    if len(categories) != 77:
        raise ValueError("Banking77 requires exactly 77 categories")
    train_rows, train_rejected = _read_split(raw / "train.csv", categories, split="train")
    test_rows, test_rejected = _read_split(raw / "test.csv", categories, split="test")
    if {row.text for row in train_rows} & {row.text for row in test_rows}:
        raise ValueError("Banking77 train/test text overlap")

    files: dict[str, bytes] = {}
    entries: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for name, rows, rejected in (("train", train_rows, train_rejected), ("test", test_rows, test_rejected)):
        encoded = _encode_rows(rows, tokenizer_callable, tokenizer_name, batch_size=batch_size)
        shards = _pack_tokens(rows, encoded, eos_token_id, shard_tokens)
        files.update({f"{name}/{path.split('/', 1)[1]}": payload for path, payload in shards.items()})
        for path, payload in shards.items():
            entries.append(file_entry(f"{name}/{path.split('/', 1)[1]}", payload))
        text_payload = _text_payload(rows)
        text_path = f"text/{name}.txt"
        files[text_path] = text_payload
        entries.append(file_entry(text_path, text_payload, kind="text"))
        labels_payload = _label_jsonl(rows)
        labels_path = f"labels/{name}.jsonl"
        files[labels_path] = labels_payload
        entries.append(file_entry(labels_path, labels_payload, kind="jsonl"))
        quarantine_payload = quarantine_jsonl(rejected)
        if quarantine_payload:
            quarantine_path = f"quarantine/{name}.jsonl"
            files[quarantine_path] = quarantine_payload
            entries.append(file_entry(quarantine_path, quarantine_payload, kind="jsonl"))
        source_url = f"{DEFAULT_SOURCE_BASE}/{revision}/banking_data/{name}.csv"
        sources.append(
            _source_metadata(
                name=name,
                rows=len(rows),
                input_path=raw / f"{name}.csv",
                token_shards={path: payload for path, payload in shards.items()},
                tokenizer_name=tokenizer_name,
                revision=revision,
                retrieval_timestamp=retrieval_timestamp,
                source_url=source_url,
                ratio=float(len(rows)),
            )
        )
    categories_payload = (raw / "categories.json").read_bytes()
    license_payload = (raw / "LICENSE").read_bytes()
    for name in ("train", "test"):
        provenance_path = f"provenance/{name}.csv"
        provenance_payload = (raw / f"{name}.csv").read_bytes()
        files[provenance_path] = provenance_payload
        entries.append(file_entry(provenance_path, provenance_payload, kind="text"))
    files["metadata/categories.json"] = categories_payload
    entries.append(file_entry("metadata/categories.json", categories_payload, kind="text"))
    files["metadata/LICENSE"] = license_payload
    entries.append(file_entry("metadata/LICENSE", license_payload, kind="text"))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "dataset",
        "artifact_id": "banking77",
        "revision": revision,
        "dataset": DEFAULT_DATASET,
        "license": DEFAULT_LICENSE,
        "tokenizer_name": tokenizer_name,
        "vocab_size": DEFAULT_VOCAB_SIZE,
        "eos_token_id": eos_token_id,
        "split_policy": "official train/test CSV rows; exact text retained",
        "source_files": _source_file_metadata(raw, revision, hashes),
        "files": entries,
        "sources": sources,
        "total_token_count": sum(int(entry["token_count"]) for entry in entries if entry["kind"] == "tokens"),
        "total_byte_count": sum(int(entry["byte_count"]) for entry in entries),
        "quarantine_count": len(train_rejected) + len(test_rejected),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "category_count": len(categories),
    }
    atomic_publish_directory(target, files, manifest)
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--retrieval-timestamp", required=True)
    parser.add_argument("--eos-token-id", type=int, default=DEFAULT_EOS_ID)
    parser.add_argument("--shard-tokens", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    if not args.retrieval_timestamp:
        parser.error("--retrieval-timestamp is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepare_banking77(
        args.raw_dir,
        args.output_dir,
        tokenizer_name=args.tokenizer,
        revision=args.revision,
        retrieval_timestamp=args.retrieval_timestamp,
        eos_token_id=args.eos_token_id,
        shard_tokens=args.shard_tokens,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
