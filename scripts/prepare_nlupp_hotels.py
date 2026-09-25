#!/usr/bin/env python3
"""Publish a pinned, decontaminated NLU++ HOTELS artifact.

This publisher follows the split produced by the upstream NLU++ ``large``
regime for fold pair ``0,1``: folds 0 and 1 form the test set and folds 2--19
form the training set.  The upstream files contain exact duplicate text and
train/test overlap, so the publication boundary records every removed row in a
quarantine JSONL file.  The upstream ``data_loader.py`` is preserved as
provenance and its pinned blob identity is checked before parsing.

The script is offline: raw files must already be acquired.  The default
Gigatoken path is a local pinned JSON asset; there is deliberately no model-id
fallback.  The tokenizer callable seam is injectable for offline tests.
"""

from __future__ import annotations

import argparse
import hashlib
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
    load_published_artifact,
    quarantine_jsonl,
    sha256_bytes,
)

Row = dict[str, Any]
DEFAULT_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
DEFAULT_DATASET = "PolyAI-LDN/task-specific-datasets/nlupp/data/hotels"
DEFAULT_LICENSE = "CC-BY-4.0"
DEFAULT_TOKENIZER = "google/gemma-4-E2B-it"
DEFAULT_VOCAB_SIZE = 262_144
DEFAULT_EOS_ID = 1
DEFAULT_TOKENIZER_JSON_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
DEFAULT_TOKENIZER_CONFIG_SHA256 = "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df"
PINNED_BANKING77_MANIFEST_SHA256 = "9927589c92c84d78fe13608fd1799dd959e973d6e759ac53ce3e4d43dedd875f"
FOLD_COUNT = 20
TEST_FOLDS = (0, 1)
TRAIN_FOLDS = tuple(range(2, FOLD_COUNT))
EXPECTED_RAW_ROWS = 1009
EXPECTED_TRAIN_ROWS = 902
EXPECTED_TEST_ROWS = 95
EXPECTED_QUARANTINE_COUNT = 12
EXPECTED_TRAIN_TEST_OVERLAP = 3
EXPECTED_DUPLICATES = 9
EXPECTED_CROSS_SOURCE_OVERLAP = 0
EXPECTED_INTENTS = 40
EXPECTED_SLOTS = 14

# SHA-256 and Git blob SHA-1 are independent integrity checks.  The blob IDs
# were obtained from the pinned GitHub tree, not inferred from local content.
PINNED_SOURCE_FILES: dict[str, dict[str, str]] = {
    "ontology.json": {
        "upstream_path": "nlupp/data/ontology.json",
        "git_blob_sha1": "d5223242a20ea0a2fe9e5d57a3eb2f32ea078875",
        "sha256": "58736607898d17f73e8afa4fa41ca455670756398f3fc91751c31b62327e38c5",
    },
    "LICENSE": {
        "upstream_path": "LICENSE",
        "git_blob_sha1": "2f244ac814036ecd9ba9f69782e89ce6b1dca9eb",
        "sha256": "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
    },
    "data_loader.py": {
        "upstream_path": "nlupp/data_loader.py",
        "git_blob_sha1": "e05194cabed9796d1b155004549cc796f726c153",
        "sha256": "ffa3631681f1a242a8674adb5be6ecc2c2511459e5e9dbb5b7eb3ca256f229f5",
    },
    **{
        f"fold{fold}.json": {
            "upstream_path": f"nlupp/data/hotels/fold{fold}.json",
            "git_blob_sha1": blob,
            "sha256": digest,
        }
        for fold, blob, digest in (
            (0, "9892b69cb0ae68fb5d0aae20cdfce634c94092db", "16724351a44b5c8b32c84a827dcd8530f4d248c6fac4d309d88fd21245fca448"),
            (1, "c015cf59da07bddf037aa4ae84c28f1a89f9a781", "123aeb21f3c73a47ce7df75ff28ba2cd318c9782f1c779e75e8e9322a36e68e7"),
            (2, "6ca1adbf11bacd7b4d2e1b83b367a6c07a1c3595", "517732b73e40ac30993ec9ca7e7531ca2e7abac829258391961836dd7ebdc2d6"),
            (3, "fc4d4698285020bc10b68d50b322066f7a0e205d", "555561c4484df605b2d180f762559d013bc864b304a6619a172d582eff15f859"),
            (4, "ecfed2d6dedacc7ef6d7cba96e0a4eee3e4efdea", "a6c7459c085a09646a1c77de4c70a288076d2426a7f2f15c8838dc4e3d4ee9a6"),
            (5, "0ba81a4df9b440c435d4eb043959a481faff832c", "489b91cb4132083fc975b85cce9774602e9aa7af46db2b7561884af17e281453"),
            (6, "43f10782aae55567106b88a38f07342a5db9ec74", "052e52bf41f87b24c5bbf246e871f6456b105230bd84a8f7f13ff65bedcc2cad"),
            (7, "b1817557ffef386f314aa7173812f6d40ff1d742", "aa7869f8031cd3caa94675cc92052afd8526cbaa81b400de665fb95ba4b62c6c"),
            (8, "5a3a311d87f540e3a766573d9566619aa170f4d1", "1d20d1b1b0ff349f33649d219b2359828977fd9663ba758e4e9e67672a0954a2"),
            (9, "138ee04d5d850f556ade6e555f80b3ee901fb88a", "0346a8658c7e47750d7a8589cd8b3b3e22f8beed3074b6488daa4213f47b5966"),
            (10, "24cea85cf08531e718e77e832f73ceff2b28e749", "b5ed876c74026c5be1aa6ae3956134a89894e01baa410c1ded1e52e72750839e"),
            (11, "547b0e7d85078e481f2ee9ea0b32f5901217d249", "582d1de5ac593226182b5e32c146750b6678a9d6198aef4c0cdfe9ad8b077161"),
            (12, "caedab8f3e147d3e761c1e9e23bed5126fbf5677", "f36349037466f2cc01430fef6b39ffadb0d4222396c7beae028cf07f333f897a"),
            (13, "95b9f9ec7f70f56a7e2940457a7abb3632c6ead9", "ee9773b91c2e37ea566b4f09687b6951a5c44d36e22292cead93eb221904e06a"),
            (14, "9c3b076d0c23009441f5bcb000fb3afd184cc6bd", "722562aa47b087cbd432930064da2e080df0ada23000783a9493786125a22dad"),
            (15, "3f411bee9e170aec1dac35a0a0dd52db37ff7118", "f2d34e8c75d1e49f29b905de61d6690ae6f85f4e040f6f72882e161a09e57c24"),
            (16, "11b323517ec8f0c48b860e57bb8a0c8910bd27ef", "dac95faf7ba946819714ec5e4e3014d0cf3e054634cc10f1ac9f7693a90280cb"),
            (17, "1414c35bb2c68014c90b003f486f49b7818c0fd0", "3e0a95d9c8097d40d8919b5405938c0088a80926d262f03c9234fc93a20a5d33"),
            (18, "f80e32407551cd68538bf31f0ccb6caf5475e331", "1034017ae94f7eb50ecf2cfb4ef850c53e362506b69b02ee7c99f346eec16437"),
            (19, "b13b00552eea082c2b0166c05e964b0a0c720bc0", "b0140eecf03dd9e6cb251c1a4b42edff2f4ea2c74079a9988777ee7181ebb3e7"),
        )
    },
}

Row = dict[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return sha256_bytes(payload)


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON: {path}") from exc


def _read_ontology(path: Path) -> dict[str, dict[str, set[str]]]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or set(payload) != {"intents", "slots"}:
        raise ValueError("ontology must contain exactly intents and slots")
    result: dict[str, dict[str, set[str]]] = {}
    for group in ("intents", "slots"):
        values = payload[group]
        if not isinstance(values, dict) or not values:
            raise ValueError(f"ontology {group} must be a non-empty mapping")
        for name, metadata in values.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(metadata, dict)
                or set(metadata) != {"description", "domain"}
                or not isinstance(metadata["description"], str)
                or not isinstance(metadata["domain"], list)
                or not metadata["domain"]
                or any(not isinstance(domain, str) or not domain for domain in metadata["domain"])
            ):
                raise ValueError(f"invalid ontology {group} entry {name!r}")
            result.setdefault(group, {})[name] = set(metadata["domain"])
    return result


def _validate_record(record: object, ontology: dict[str, dict[str, set[str]]], *, fold: int, row: int) -> Row:
    if not isinstance(record, dict) or not set(record).issubset({"text", "intents", "slots"}):
        raise ValueError(f"invalid record keys at fold {fold} row {row}")
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"empty record text at fold {fold} row {row}")
    if "\n" in text or "\r" in text:
        raise ValueError(f"embedded newline in record text at fold {fold} row {row}")
    intents = record.get("intents", [])
    if not isinstance(intents, list) or len(set(intents)) != len(intents):
        raise ValueError(f"invalid intents at fold {fold} row {row}")
    for intent in intents:
        if not isinstance(intent, str) or intent not in ontology.get("intents", {}):
            raise ValueError(f"unknown intent {intent!r} at fold {fold} row {row}")
        if not ontology["intents"][intent].intersection({"general", "hotels"}):
            raise ValueError(f"intent {intent!r} is not valid for hotels at fold {fold} row {row}")
    slots = record.get("slots", {})
    if not isinstance(slots, dict):
        raise ValueError(f"invalid slots at fold {fold} row {row}")
    for name, annotation in slots.items():
        if name not in ontology.get("slots", {}):
            raise ValueError(f"unknown slot {name!r} at fold {fold} row {row}")
        if not ontology["slots"][name].intersection({"general", "hotels"}):
            raise ValueError(f"slot {name!r} is not valid for hotels at fold {fold} row {row}")
        if not isinstance(annotation, dict) or not set(annotation).issubset({"text", "span", "value", "values"}):
            raise ValueError(f"invalid slot annotation {name!r} at fold {fold} row {row}")
        if ("value" in annotation) == ("values" in annotation):
            raise ValueError(f"slot {name!r} must contain exactly one of value/values")
        if not isinstance(annotation.get("text"), str) or not isinstance(annotation.get("span"), list):
            raise ValueError(f"invalid slot span at fold {fold} row {row}")
        span = annotation["span"]
        if len(span) != 2 or any(type(value) is not int for value in span):
            raise ValueError(f"invalid slot span at fold {fold} row {row}")
        start, end = span
        if not 0 <= start <= end <= len(text) or text[start:end] != annotation["text"]:
            raise ValueError(f"slot span mismatch at fold {fold} row {row}")
    return {"source_fold": fold, "source_row": row, "text": text, "intents": list(intents), "slots": dict(slots)}


def _load_rows(raw: Path, ontology: dict[str, dict[str, set[str]]]) -> list[Row]:
    rows: list[Row] = []
    for fold in range(FOLD_COUNT):
        payload = _read_json(raw / f"fold{fold}.json")
        if not isinstance(payload, list):
            raise ValueError(f"fold {fold} must contain a JSON list")
        rows.extend(_validate_record(record, ontology, fold=fold, row=row) for row, record in enumerate(payload))
    return rows


def _banking77_text_hashes(artifact: str | Path) -> set[str]:
    root = Path(artifact).resolve(strict=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Banking77 manifest is missing")
    actual_manifest_sha256 = _sha256_bytes(manifest_path.read_bytes())
    if actual_manifest_sha256 != PINNED_BANKING77_MANIFEST_SHA256:
        raise ValueError("Banking77 manifest sha256 mismatch")
    load_published_artifact(root)
    hashes: set[str] = set()
    for name in ("train.txt", "test.txt"):
        path = root / "text" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes.update(_sha256_bytes(line.encode("utf-8")) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return hashes


def _encode_rows(rows: list[Row], tokenizer_callable: Callable[[list[str]], list[list[int]]] | None, tokenizer_name: str, batch_size: int) -> list[list[int]]:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if tokenizer_callable is None:
        raise ValueError("tokenizer callable is required")
    result: list[list[int]] = []
    texts = [row["text"] for row in rows]
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer_callable(batch)
        if len(encoded) != len(batch):
            raise ValueError("tokenizer returned a different number of rows")
        for row, values in zip(rows[start : start + batch_size], encoded, strict=True):
            ids = [int(value) for value in values]
            if not ids or min(ids) < 0 or max(ids) >= DEFAULT_VOCAB_SIZE:
                raise ValueError(f"token id outside declared vocabulary at fold {row['source_fold']} row {row['source_row']}")
            result.append(ids)
    return result


def _pack_tokens(rows: list[Row], encoded: list[list[int]], eos_token_id: int, shard_tokens: int) -> dict[str, bytes]:
    if type(eos_token_id) is not int or not 0 <= eos_token_id < DEFAULT_VOCAB_SIZE:
        raise ValueError("eos_token_id must be in [0, 262144)")
    if type(shard_tokens) is not int or shard_tokens < 2:
        raise ValueError("shard_tokens must be >= 2")
    shards: dict[str, bytes] = {}
    current = bytearray()
    count = 0
    index = 0
    for row, ids in zip(rows, encoded, strict=True):
        required = len(ids) + 1
        if required > shard_tokens:
            raise ValueError(f"document exceeds shard_tokens at fold {row['source_fold']} row {row['source_row']}")
        if count and count + required > shard_tokens:
            shards[f"tokens/shard-{index:06d}.bin"] = bytes(current)
            index += 1
            current.clear()
            count = 0
        current.extend(np.asarray([*ids, eos_token_id], dtype="<u4").tobytes())
        count += required
    if current:
        shards[f"tokens/shard-{index:06d}.bin"] = bytes(current)
    return shards


def _combine_hash(payloads: list[bytes]) -> str:
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _text_payload(rows: list[Row]) -> bytes:
    return ("\n".join(row["text"] for row in rows) + "\n").encode("utf-8")


def _label_payload(rows: list[Row]) -> bytes:
    records = [
        {
            "intents": row["intents"],
            "slots": row["slots"],
            "source_fold": row["source_fold"],
            "source_row": row["source_row"],
            "text_sha256": _sha256_bytes(row["text"].encode("utf-8")),
        }
        for row in rows
    ]
    return ("\n".join(json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records) + "\n").encode("utf-8")


def _source_entry(name: str, spec: Mapping[str, str], raw: Path, revision: str) -> dict[str, Any]:
    path = raw / name
    payload = path.read_bytes()
    actual_blob = _git_blob_sha1(payload)
    actual_sha = _sha256_bytes(payload)
    if actual_blob != spec["git_blob_sha1"]:
        raise ValueError(f"git blob mismatch: {name}")
    if actual_sha != spec["sha256"]:
        raise ValueError(f"sha256 mismatch: {name}")
    return {
        "path": name,
        "upstream_path": spec["upstream_path"],
        "git_blob_sha1": actual_blob,
        "sha256": actual_sha,
        "byte_count": len(payload),
        "url": f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{revision}/{spec['upstream_path']}",
    }


def _tokenizer_asset(
    tokenizer_json: str | Path | None,
    tokenizer_config: str | Path | None,
) -> dict[str, Any] | None:
    if tokenizer_json is None and tokenizer_config is None:
        return None
    if tokenizer_json is None or tokenizer_config is None:
        raise ValueError("tokenizer_json and tokenizer_config must be supplied together")
    json_path = Path(tokenizer_json).resolve(strict=True)
    config_path = Path(tokenizer_config).resolve(strict=True)
    json_sha = _sha256_bytes(json_path.read_bytes())
    config_sha = _sha256_bytes(config_path.read_bytes())
    if json_sha != DEFAULT_TOKENIZER_JSON_SHA256 or config_sha != DEFAULT_TOKENIZER_CONFIG_SHA256:
        raise ValueError("pinned local tokenizer asset hash mismatch")
    return {
        "name": DEFAULT_TOKENIZER,
        "vocab_size": DEFAULT_VOCAB_SIZE,
        "tokenizer_json_sha256": json_sha,
        "tokenizer_config_sha256": config_sha,
        "tokenizer_json_path": str(json_path),
        "tokenizer_config_path": str(config_path),
    }


def _local_tokenizer_callable(tokenizer_json: str | Path | None) -> Callable[[list[str]], list[list[int]]] | None:
    if tokenizer_json is None:
        return None
    import gigatoken

    native = gigatoken.Tokenizer.from_json(Path(tokenizer_json).read_bytes())
    if int(native.vocab_size) != DEFAULT_VOCAB_SIZE:
        raise ValueError("local Gigatoken vocabulary size mismatch")
    return native.encode_batch_list


def prepare_nlupp_hotels(
    raw_dir: str | Path,
    output_dir: str | Path,
    banking77_artifact: str | Path,
    *,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    tokenizer_callable: Callable[[list[str]], list[list[int]]] | None = None,
    tokenizer_json: str | Path | None = None,
    tokenizer_config: str | Path | None = None,
    revision: str = DEFAULT_REVISION,
    retrieval_timestamp: str,
    eos_token_id: int = DEFAULT_EOS_ID,
    shard_tokens: int = 100_000,
    batch_size: int = 256,
) -> Path:
    """Publish exact text, labels, packed tokens and a fail-closed manifest."""
    raw = Path(raw_dir)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(target)
    if revision != DEFAULT_REVISION:
        raise ValueError(f"publisher is pinned to revision {DEFAULT_REVISION}")
    if not isinstance(retrieval_timestamp, str) or not retrieval_timestamp:
        raise ValueError("retrieval_timestamp must be a non-empty explicit string")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if tokenizer_callable is None and tokenizer_json is not None:
        tokenizer_callable = _local_tokenizer_callable(tokenizer_json)
    tokenizer_asset = _tokenizer_asset(tokenizer_json, tokenizer_config)
    if tokenizer_callable is None and tokenizer_asset is None:
        raise ValueError("a local tokenizer asset or tokenizer_callable is required")

    source_files = [
        _source_entry(name, PINNED_SOURCE_FILES[name], raw, revision)
        for name in sorted(PINNED_SOURCE_FILES)
    ]
    ontology = _read_ontology(raw / "ontology.json")
    rows = _load_rows(raw, ontology)
    if len(rows) != EXPECTED_RAW_ROWS:
        raise ValueError(f"unexpected raw row count: {len(rows)}")
    if not all(row["source_fold"] in set(TRAIN_FOLDS) | set(TEST_FOLDS) for row in rows):
        raise ValueError("source row is outside the pinned fold set")

    # Test rows are accepted first, then train rows are filtered test-wins.
    test_rows: list[Row] = []
    test_hashes = set()
    duplicate_count = 0
    train_test_overlap_count = 0
    banking_hashes = _banking77_text_hashes(banking77_artifact)
    cross_source_overlap_count = 0
    quarantine: list[dict[str, Any]] = []
    accepted_train: list[Row] = []
    train_candidates: list[Row] = [row for row in rows if row["source_fold"] not in TEST_FOLDS]
    seen_train: set[str] = set()

    def quarantine_row(row: Row, reason: str) -> None:
        quarantine.append(
            {
                "source_fold": row["source_fold"],
                "source_row": row["source_row"],
                "reason": reason,
                "content_sha256": _sha256_bytes(row["text"].encode("utf-8")),
            }
        )

    for row in rows:
        if row["source_fold"] not in TEST_FOLDS:
            continue
        digest = _sha256_bytes(row["text"].encode("utf-8"))
        if digest in test_hashes:
            duplicate_count += 1
            quarantine_row(row, "duplicate_within_test")
        else:
            test_hashes.add(digest)
            test_rows.append(row)

    for row in train_candidates:
        digest = _sha256_bytes(row["text"].encode("utf-8"))
        if digest in test_hashes:
            train_test_overlap_count += 1
            quarantine_row(row, "train_test_overlap")
        elif digest in seen_train:
            duplicate_count += 1
            quarantine_row(row, "duplicate_within_train")
        elif digest in banking_hashes:
            cross_source_overlap_count += 1
            quarantine_row(row, "banking77_cross_source_overlap")
        else:
            seen_train.add(digest)
            accepted_train.append(row)

    if (len(test_rows), len(accepted_train)) != (EXPECTED_TEST_ROWS, EXPECTED_TRAIN_ROWS):
        raise ValueError(f"decontaminated row count mismatch: test={len(test_rows)} train={len(accepted_train)}")
    if (duplicate_count, train_test_overlap_count, cross_source_overlap_count) != (
        EXPECTED_DUPLICATES,
        EXPECTED_TRAIN_TEST_OVERLAP,
        EXPECTED_CROSS_SOURCE_OVERLAP,
    ):
        raise ValueError("decontamination counts differ from the pinned audit")
    if len(quarantine) != EXPECTED_QUARANTINE_COUNT:
        raise ValueError("quarantine count differs from the pinned audit")

    files: dict[str, bytes] = {}
    entries: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    split_rows = {"train": accepted_train, "test": test_rows}
    for split, split_data in split_rows.items():
        encoded = _encode_rows(split_data, tokenizer_callable, tokenizer_name, batch_size)
        shards = _pack_tokens(split_data, encoded, eos_token_id, shard_tokens)
        for path, payload in shards.items():
            relative = f"{split}/{path.removeprefix('tokens/')}"
            files[relative] = payload
            entries.append(file_entry(relative, payload))
        text_payload = _text_payload(split_data)
        files[f"text/{split}.txt"] = text_payload
        entries.append(file_entry(f"text/{split}.txt", text_payload, kind="text"))
        labels_payload = _label_payload(split_data)
        files[f"labels/{split}.jsonl"] = labels_payload
        entries.append(file_entry(f"labels/{split}.jsonl", labels_payload, kind="jsonl"))
        output_hash = _combine_hash(list(shards.values()))
        input_payloads = [
            (raw / f"fold{fold}.json").read_bytes()
            for fold in (TEST_FOLDS if split == "test" else TRAIN_FOLDS)
        ]
        sources.append(
            {
                "name": split,
                "ratio": float(len(split_data)),
                "dataset": DEFAULT_DATASET,
                "revision": revision,
                "license": DEFAULT_LICENSE,
                "source_url": f"https://github.com/PolyAI-LDN/task-specific-datasets/tree/{revision}/nlupp/data/hotels",
                "retrieval_timestamp": retrieval_timestamp,
                "tokenizer_version": tokenizer_name,
                "filter_policy_version": "nlupp-strict-schema-ontology-spans-v1",
                "dedup_policy_version": "nlupp-test-wins-exact-sha256-v1",
                "byte_count": sum(len(payload) for payload in input_payloads),
                "token_count": sum(len(ids) + 1 for ids in encoded),
                "input_sha256": _combine_hash(input_payloads),
                "output_sha256": output_hash,
            }
        )

    quarantine_payload = quarantine_jsonl(quarantine)
    files["quarantine/records.jsonl"] = quarantine_payload
    entries.append(file_entry("quarantine/records.jsonl", quarantine_payload, kind="jsonl"))
    for name in sorted(PINNED_SOURCE_FILES):
        payload = (raw / name).read_bytes()
        relative = f"provenance/{name}"
        files[relative] = payload
        entries.append(file_entry(relative, payload, kind="text"))

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "dataset",
        "artifact_id": "nlupp_hotels",
        "revision": revision,
        "dataset": DEFAULT_DATASET,
        "license": DEFAULT_LICENSE,
        "tokenizer_name": tokenizer_name,
        "tokenizer_asset": tokenizer_asset,
        "vocab_size": DEFAULT_VOCAB_SIZE,
        "eos_token_id": eos_token_id,
        "split_policy": "NLU++ large regime fold pair 0,1: test=folds0,1; train=folds2-19",
        "train_folds": list(TRAIN_FOLDS),
        "test_folds": list(TEST_FOLDS),
        "raw_rows": len(rows),
        "train_rows": len(accepted_train),
        "test_rows": len(test_rows),
        "intent_count": len({intent for row in rows for intent in row["intents"]}),
        "slot_count": len({slot for row in rows for slot in row["slots"]}),
        "quarantine_count": len(quarantine),
        "duplicate_count": duplicate_count,
        "train_test_overlap_count": train_test_overlap_count,
        "cross_source_overlap_count": cross_source_overlap_count,
        "source_files": source_files,
        "files": entries,
        "sources": sources,
        "total_token_count": sum(int(entry["token_count"]) for entry in entries if entry["kind"] == "tokens"),
        "total_byte_count": sum(int(entry["byte_count"]) for entry in entries),
    }
    if manifest["intent_count"] != EXPECTED_INTENTS or manifest["slot_count"] != EXPECTED_SLOTS:
        raise ValueError("ontology coverage differs from the pinned NLU++ audit")
    atomic_publish_directory(target, files, manifest)
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--banking77-artifact", type=Path, required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
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
    prepare_nlupp_hotels(
        args.raw_dir,
        args.output_dir,
        args.banking77_artifact,
        tokenizer_name=args.tokenizer,
        tokenizer_json=args.tokenizer_json,
        tokenizer_config=args.tokenizer_config,
        revision=args.revision,
        retrieval_timestamp=args.retrieval_timestamp,
        eos_token_id=args.eos_token_id,
        shard_tokens=args.shard_tokens,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
