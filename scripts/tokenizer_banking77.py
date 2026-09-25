"""Pinned Banking77 tokenizer-frontier experiment.

This file is deliberately opt-in and isolated from ``hagi.train.loop.Trainer``.
It implements the protocol in ``.omc/plans/tokenizer_banking77.md`` without
changing production tokenizer, packed-data, model, or optimizer code.

The tokenizer calls are the installed APIs documented by the projects:
Gigatoken uses ``encode_batch_list``/``decode`` and Hugging Face Tokenizers
uses ``Tokenizer.from_file``/``encode_batch``/``decode``.  Imports are lazy so
offline helper tests do not require either package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from importlib import metadata
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Protocol

# CUDA/ROCm reads this at first BLAS initialization. Set it before importing
# torch so a fresh CLI process cannot silently use a nondeterministic workspace.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.config import Config, validate_config  # noqa: E402

# Reuse the artifact publisher's existing symlink/junction-chain guard.
from hagi.data.artifacts import _reject_link_chain, load_published_artifact  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.optim import _muon_parameters, learning_rate_at  # noqa: E402

EXPECTED_MANIFEST_SHA256 = "44d50edd994e7c32f30066a26052e15f902c74b79a7498ba60f475c76b453f3b"
EXPECTED_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
EXPECTED_DATASET = "PolyAI-LDN/task-specific-datasets/banking_data"
EXPECTED_LICENSE = "CC-BY-4.0"
EXPECTED_TRAIN_ROWS = 10_003
EXPECTED_TEST_ROWS = 3_080
EXPECTED_CATEGORIES = 77
EXPECTED_MODEL_VOCAB = 262_144
EXPECTED_BASELINE_VOCAB = 262_144
EXPECTED_BASELINE_EOS = 1
EXPECTED_BASELINE_NAME = "google/gemma-4-E2B-it"
EXPECTED_CANDIDATE_NAME = "tokenizer-0997f410"
EXPECTED_BASELINE_TOKENIZER_JSON_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
EXPECTED_BASELINE_TOKENIZER_CONFIG_SHA256 = "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df"
EXPECTED_GIGATOKEN_VERSION = "0.10.0"
EXPECTED_CANDIDATE_TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
EXPECTED_CANDIDATE_CONFIG_SHA256 = "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
EXPECTED_CANDIDATE_ASSET_SHA256 = {
    "tokenizer.json": EXPECTED_CANDIDATE_TOKENIZER_SHA256,
    "tokenizer_config.json": EXPECTED_CANDIDATE_CONFIG_SHA256,
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "config.json": "f8d190c5b89c1521220f935d2567a587d6e291ed69066a45a106560b05a2174c",
    "generation_config.json": "303aba891d66ab63908a7b3cc9163bcb835fdf8b9f6301c73216f3f1eb3992dd",
}
EXPECTED_CANDIDATE_ACTIVE_VOCAB = 248_077
EXPECTED_CANDIDATE_EOS = 248_046
EXPECTED_CANDIDATE_PAD = 248_044
EXPECTED_BASELINE_ACTIVE_FINGERPRINT = "f6eee1883c5b974ef0147007ab0d9d2eaf7e13484043d7bb98e9cc3e4c877dfc"
EXPECTED_BASELINE_VOCAB_SHA256 = "fe7c58c0bc9d9ede7c9109b42a619d8eee4a570c78f1b7f44b3199c8fa63dc69"
EXPECTED_BASELINE_MERGES_SHA256 = "757be9325462bc729dfc66f19a1e21c24c456cb4e81a678e0ae7bed95104db49"
EXPECTED_BASELINE_MERGES_COUNT = 514_906
EXPECTED_BASELINE_NATIVE_TRAIN_TOKENS = 140_119
EXPECTED_BASELINE_NATIVE_TEST_TOKENS = 39_480
EXPECTED_CANDIDATE_NATIVE_TRAIN_TOKENS = 137_360
EXPECTED_CANDIDATE_NATIVE_TEST_TOKENS = 38_779
EXPECTED_PUBLISHED_TRAIN_TOKENS_WITH_EOS = 150_122
EXPECTED_PUBLISHED_TEST_TOKENS_WITH_EOS = 42_560
EXPECTED_PUBLISHED_TRAIN_STREAM_SHA256 = "09968b65c099f31615f899456d36a7b4dd1ffc30f33d110dabffe92bab14b80c"
EXPECTED_PUBLISHED_TEST_STREAM_SHA256 = "566855e61c31bb8c949e52dc7e645a2ae537e1993f14f0a3c7e01f5f4928daed"
EXPECTED_TRAIN_CSV_SHA256 = "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
EXPECTED_TEST_CSV_SHA256 = "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"
EXPECTED_TOKENIZERS_VERSION = "0.23.2"
EXPECTED_TOTAL_UPDATES = 471
EXPECTED_SEEDS = (1234, 2243, 3252)
BLOCK_SIZE = 64
EPOCHS = 3
EPOCH_SEED_STRIDE = 1_000_003
LANE_ORDER = {
    1234: ("baseline", "candidate"),
    2243: ("candidate", "baseline"),
    3252: ("candidate", "baseline"),
}
PINNED_SOURCE_FILES = (
    ".omc/plans/tokenizer_banking77.md",
    "scripts/tokenizer_banking77.py",
    "tests/test_tokenizer_banking77.py",
    "src/hagi/__init__.py",
    "src/hagi/config.py",
    "src/hagi/data/__init__.py",
    "src/hagi/data/artifacts.py",
    "src/hagi/model/__init__.py",
    "src/hagi/model/adaptive.py",
    "src/hagi/model/attention.py",
    "src/hagi/model/block.py",
    "src/hagi/model/cortex.py",
    "src/hagi/model/decision.py",
    "src/hagi/model/embedding.py",
    "src/hagi/model/ffn.py",
    "src/hagi/model/head.py",
    "src/hagi/model/kv_cache.py",
    "src/hagi/model/merge.py",
    "src/hagi/model/model.py",
    "src/hagi/model/multimodal.py",
    "src/hagi/model/norms.py",
    "src/hagi/model/outputs.py",
    "src/hagi/model/rope.py",
    "src/hagi/model/ternary.py",
    "src/hagi/train/__init__.py",
    "src/hagi/train/optim.py",
    "src/hagi/version.py",
)


@dataclass(frozen=True)
class Document:
    source_index: int
    text: str
    label: int
    intent: str
    utf8_bytes: int


@dataclass(frozen=True)
class Corpus:
    train: tuple[Document, ...]
    test: tuple[Document, ...]
    manifest_sha256: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class TokenizedCorpus:
    train: tuple[tuple[int, ...], ...]
    test: tuple[tuple[int, ...], ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class PaddedExposure:
    input_ids: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor
    lengths: tuple[int, ...]
    native_tokens: int


class TokenizerAdapter(Protocol):
    name: str
    version: str
    vocab_size: int
    eos_id: int
    pad_id: int

    def encode_batch(self, texts: list[str]) -> list[list[int]]: ...
    def encode_one(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def provenance(self) -> dict[str, Any]: ...


class GigaAdapter:
    def __init__(
        self,
        native: Any,
        *,
        version: str,
        expected_sha256: str | None,
        tokenizer_path: Path | None = None,
        tokenizer_config_path: Path | None = None,
    ) -> None:
        self.native = native
        self.name = EXPECTED_BASELINE_NAME
        self.version = str(version)
        self.vocab_size = int(native.vocab_size)
        self.eos_id = EXPECTED_BASELINE_EOS
        self.pad_id = 0
        self.expected_sha256 = expected_sha256
        self.tokenizer_path = tokenizer_path
        self.tokenizer_config_path = tokenizer_config_path
        if self.version != EXPECTED_GIGATOKEN_VERSION or self.vocab_size != EXPECTED_BASELINE_VOCAB:
            raise ValueError("installed gigatoken baseline version or vocabulary mismatch")

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        rows = self.native.encode_batch_list(texts)
        if len(rows) != len(texts):
            raise ValueError("gigatoken returned a different number of rows")
        result: list[list[int]] = []
        for row in rows:
            if any(not isinstance(value, Integral) or isinstance(value, bool) for value in row):
                raise ValueError("gigatoken returned a non-integer token id")
            result.append([int(value) for value in row])
        return result

    def decode(self, ids: list[int]) -> str:
        value = self.native.decode(ids)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="strict")
        if isinstance(value, str):
            return value
        raise TypeError("gigatoken.decode returned neither bytes nor str")

    def encode_one(self, text: str) -> list[int]:
        return self.encode_batch([text])[0]

    def provenance(self) -> dict[str, Any]:
        backend = self.native._backend
        vocab = backend.vocab
        merges = backend.merges
        vocab_digest = hashlib.sha256()
        for token_id in sorted(int(key) for key in vocab):
            piece = vocab[token_id]
            if not isinstance(piece, bytes):
                raise ValueError("gigatoken vocabulary pieces must be bytes")
            vocab_digest.update(token_id.to_bytes(8, "little"))
            vocab_digest.update(len(piece).to_bytes(8, "little"))
            vocab_digest.update(piece)
        merges_digest = hashlib.sha256()
        for pair in merges:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("gigatoken merges must be ordered byte pairs")
            for piece in pair:
                merges_digest.update(len(piece).to_bytes(8, "little"))
                merges_digest.update(piece)
        combined = hashlib.sha256(vocab_digest.digest() + merges_digest.digest()).hexdigest()
        return {
            "backend_record": _distribution_provenance("gigatoken"),
            "tokenizer_file": (
                _file_provenance(self.tokenizer_path, self.expected_sha256)
                if self.tokenizer_path is not None
                else None
            ),
            "tokenizer_config_file": (
                _file_provenance(
                    self.tokenizer_config_path,
                    EXPECTED_BASELINE_TOKENIZER_CONFIG_SHA256,
                )
                if self.tokenizer_config_path is not None
                else None
            ),
            "active_vocab_size": self.vocab_size,
            "active_vocab_sha256": vocab_digest.hexdigest(),
            "merges_count": len(merges),
            "merges_sha256": merges_digest.hexdigest(),
            "fingerprint_sha256": combined,
            "eos_token_id": self.eos_id,
        }


class HFAdapter:
    def __init__(
        self,
        native: Any,
        *,
        tokenizer_path: Path,
        config_path: Path,
        version: str,
        expected_sha256: str | None,
        expected_config_sha256: str | None,
    ) -> None:
        self.native = native
        self.name = EXPECTED_CANDIDATE_NAME
        self.version = str(version)
        self.tokenizer_path = tokenizer_path
        self.config_path = config_path
        self.expected_sha256 = expected_sha256
        self.expected_config_sha256 = expected_config_sha256
        vocab = native.get_vocab(with_added_tokens=True)
        self.vocab_size = len(vocab)
        self.eos_id = int(vocab["<|im_end|>"])
        self.pad_id = int(vocab["<|endoftext|>"])
        config: dict[str, Any] = {}
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("eos_token") != "<|im_end|>" or config.get("pad_token") != "<|endoftext|>":
                raise ValueError("candidate tokenizer special-token config mismatch")
        elif expected_config_sha256 is not None:
            raise FileNotFoundError(config_path)
        if (
            self.version != EXPECTED_TOKENIZERS_VERSION
            or self.vocab_size != EXPECTED_CANDIDATE_ACTIVE_VOCAB
            or self.eos_id != EXPECTED_CANDIDATE_EOS
            or self.pad_id != EXPECTED_CANDIDATE_PAD
        ):
            raise ValueError("installed tokenizers candidate version, vocabulary, or special IDs mismatch")

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        rows = self.native.encode_batch(texts, add_special_tokens=False)
        if len(rows) != len(texts):
            raise ValueError("Hugging Face tokenizer returned a different number of rows")
        result: list[list[int]] = []
        for row in rows:
            if any(not isinstance(value, Integral) or isinstance(value, bool) for value in row.ids):
                raise ValueError("Hugging Face tokenizer returned a non-integer token id")
            if any(int(value) != 0 for value in row.special_tokens_mask):
                raise ValueError("Hugging Face tokenizer injected a special token")
            result.append([int(value) for value in row.ids])
        return result

    def decode(self, ids: list[int]) -> str:
        value = self.native.decode(ids)
        if not isinstance(value, str):
            raise TypeError("Hugging Face tokenizer.decode returned a non-string")
        return value

    def encode_one(self, text: str) -> list[int]:
        return self.encode_batch([text])[0]

    def provenance(self) -> dict[str, Any]:
        vocab = self.native.get_vocab(with_added_tokens=True)
        digest = hashlib.sha256()
        for token, token_id in sorted(vocab.items(), key=lambda item: (int(item[1]), item[0])):
            digest.update(int(token_id).to_bytes(8, "little"))
            raw = token.encode("utf-8")
            digest.update(len(raw).to_bytes(8, "little"))
            digest.update(raw)
        return {
            "tokenizer_file": _file_provenance(self.tokenizer_path, self.expected_sha256),
            "config_file": _file_provenance(self.config_path, self.expected_config_sha256),
            "package_record": _distribution_provenance("tokenizers"),
            "active_vocab_size": self.vocab_size,
            "active_vocab_sha256": digest.hexdigest(),
            "eos_token_id": self.eos_id,
            "pad_token_id": self.pad_id,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_provenance(path: Path, expected: str | None = None) -> dict[str, Any]:
    actual = _sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"pinned tokenizer asset sha256 mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": actual, "byte_count": path.stat().st_size}


def _package_record(name: str) -> dict[str, Any]:
    """Hash the installed distribution RECORD without rereading its payload files."""
    dist = metadata.distribution(name)
    record = dist.read_text("RECORD") or ""
    payload = record.encode("utf-8")
    return {
        "name": name,
        "version": dist.version,
        "files": len(dist.files or []),
        "record_sha256": hashlib.sha256(payload).hexdigest(),
        "record_byte_count": len(payload),
    }


def _distribution_provenance(name: str) -> dict[str, Any]:
    return _package_record(name)


def parse_csv_rows(text: str, split: str, *, expected_sha256: str | None = None) -> tuple[list[dict[str, str]], str]:
    """Parse provenance CSV strictly and verify its exact byte hash when pinned."""
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if expected_sha256 is not None and text_hash != expected_sha256:
        raise ValueError(f"{split} provenance CSV sha256 mismatch")
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if reader.fieldnames != ["text", "category"]:
        raise ValueError(f"{split} CSV has unexpected columns")
    rows: list[dict[str, str]] = []
    for row in reader:
        if None in row or set(row) != {"text", "category"}:
            raise ValueError(f"{split} CSV row is malformed")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"{split} row has empty text")
        if not isinstance(row["category"], str) or not row["category"]:
            raise ValueError(f"{split} row has empty category")
        rows.append({"text": row["text"], "category": row["category"]})
    return rows, text_hash


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object JSONL record")
            records.append(value)
    return records


def document_stream_sha256(documents: Iterable[tuple[str, ...]]) -> str:
    digest = hashlib.sha256()
    for index, values in enumerate(documents):
        payload = "".join(values).encode("utf-8")
        digest.update(index.to_bytes(8, "little"))
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _document_stream_hashes(corpus: Corpus) -> dict[str, str]:
    return {
        "train": document_stream_sha256(
            [(document.text,) for document in corpus.train]
        ),
        "test": document_stream_sha256(
            [(document.text,) for document in corpus.test]
        ),
    }


def _normalized_overlap_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_corpus(artifact: str | Path) -> Corpus:
    root = Path(artifact).resolve(strict=True)
    if _sha256_file(root / "manifest.json") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("pinned Banking77 manifest sha256 mismatch")
    manifest = load_published_artifact(root)
    expected = {
        "artifact_id": "banking77", "revision": EXPECTED_REVISION, "dataset": EXPECTED_DATASET,
        "license": EXPECTED_LICENSE, "train_rows": EXPECTED_TRAIN_ROWS, "test_rows": EXPECTED_TEST_ROWS,
        "category_count": EXPECTED_CATEGORIES, "quarantine_count": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key} mismatch")
    categories = json.loads((root / "metadata/categories.json").read_text(encoding="utf-8"))
    if not isinstance(categories, list) or len(categories) != EXPECTED_CATEGORIES or len(set(categories)) != 77:
        raise ValueError("Banking77 categories mismatch")
    labels_by_intent = {str(value): index for index, value in enumerate(categories)}
    splits: dict[str, tuple[Document, ...]] = {}
    hashes: dict[str, str] = {}
    expected_csv_hashes = {
        "train": EXPECTED_TRAIN_CSV_SHA256,
        "test": EXPECTED_TEST_CSV_SHA256,
    }
    for split, expected_rows in (("train", EXPECTED_TRAIN_ROWS), ("test", EXPECTED_TEST_ROWS)):
        csv_path = root / "provenance" / f"{split}.csv"
        csv_bytes = csv_path.read_bytes()
        rows, csv_hash = parse_csv_rows(
            csv_bytes.decode("utf-8"),
            split,
            expected_sha256=expected_csv_hashes[split],
        )
        if csv_hash != expected_csv_hashes[split]:
            raise ValueError(f"{split} provenance CSV hash mismatch")
        labels = _read_jsonl(root / "labels" / f"{split}.jsonl")
        if len(rows) != expected_rows or len(labels) != expected_rows:
            raise ValueError(f"{split} row count mismatch")
        documents: list[Document] = []
        for source_index, (row, label) in enumerate(zip(rows, labels, strict=True), start=2):
            text = row["text"]
            category = row["category"]
            if category not in labels_by_intent:
                raise ValueError(f"{split} row has an unknown category")
            expected_label = labels_by_intent[category]
            if (
                type(label.get("source_row")) is not int
                or label.get("source_row") != source_index
                or label.get("intent") != category
                or type(label.get("label")) is not int
                or label.get("label") != expected_label
            ):
                raise ValueError(f"{split} label schema or source alignment mismatch")
            if label.get("text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
                raise ValueError(f"{split} text sha256 mismatch")
            documents.append(
                Document(source_index - 2, text, expected_label, category, len(text.encode("utf-8")))
            )
        splits[split] = tuple(documents)
        hashes[split] = csv_hash
    train_texts = {document.text for document in splits["train"]}
    test_texts = {document.text for document in splits["test"]}
    exact_overlap = train_texts & test_texts
    if exact_overlap:
        raise ValueError("Banking77 train/test exact text overlap")
    normalized_train = {_normalized_overlap_text(document.text) for document in splits["train"]}
    normalized_test = {_normalized_overlap_text(document.text) for document in splits["test"]}
    normalized_overlap = normalized_train & normalized_test
    normalized_overlap_digest = hashlib.sha256(
        "\n".join(sorted(normalized_overlap)).encode("utf-8")
    ).hexdigest()
    return Corpus(
        splits["train"],
        splits["test"],
        EXPECTED_MANIFEST_SHA256,
        {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "provenance_csv_sha256": hashes,
            "quarantine_count": 0,
            "exact_overlap_count": 0,
            "normalized_overlap_policy": "NFKC + casefold + whitespace-collapse; diagnostic only",
            "normalized_overlap_count": len(normalized_overlap),
            "normalized_overlap_sha256": normalized_overlap_digest,
            "train_content_bytes": sum(document.utf8_bytes for document in splits["train"]),
            "test_content_bytes": sum(document.utf8_bytes for document in splits["test"]),
        },
    )


def create_baseline_adapter(tokenizer_dir: str | Path) -> GigaAdapter:
    """Load the pinned baseline from a local directory; never a model id."""
    import gigatoken

    root = Path(tokenizer_dir).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    tokenizer_path = root / "tokenizer.json"
    config_path = root / "tokenizer_config.json"
    _file_provenance(tokenizer_path, EXPECTED_BASELINE_TOKENIZER_JSON_SHA256)
    _file_provenance(config_path, EXPECTED_BASELINE_TOKENIZER_CONFIG_SHA256)
    return GigaAdapter(
        gigatoken.Tokenizer.from_json(tokenizer_path.read_bytes()),
        version=metadata.version("gigatoken"),
        expected_sha256=EXPECTED_BASELINE_TOKENIZER_JSON_SHA256,
        tokenizer_path=tokenizer_path,
        tokenizer_config_path=config_path,
    )


def create_candidate_adapter(tokenizer_dir: str | Path) -> HFAdapter:
    from tokenizers import Tokenizer

    root = Path(tokenizer_dir).resolve(strict=True)
    for name, expected in EXPECTED_CANDIDATE_ASSET_SHA256.items():
        _file_provenance(root / name, expected)
    tokenizer_path = root / "tokenizer.json"
    config_path = root / "tokenizer_config.json"
    return HFAdapter(
        Tokenizer.from_file(str(tokenizer_path)),
        tokenizer_path=tokenizer_path,
        config_path=config_path,
        version=metadata.version("tokenizers"),
        expected_sha256=EXPECTED_CANDIDATE_TOKENIZER_SHA256,
        expected_config_sha256=EXPECTED_CANDIDATE_CONFIG_SHA256,
    )


def _token_stream_sha256(rows: tuple[tuple[int, ...], ...]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(len(row).to_bytes(8, "little"))
        for token in row:
            digest.update(int(token).to_bytes(4, "little"))
    return digest.hexdigest()


def tokenize_corpus(corpus: Corpus, adapter: TokenizerAdapter, *, batch_size: int = 256) -> TokenizedCorpus:
    result: dict[str, tuple[tuple[int, ...], ...]] = {}
    for split, documents in (("train", corpus.train), ("test", corpus.test)):
        encoded: list[tuple[int, ...]] = []
        for start in range(0, len(documents), batch_size):
            chunk = documents[start : start + batch_size]
            rows = adapter.encode_batch([document.text for document in chunk])
            if len(rows) != len(chunk):
                raise ValueError(f"{split} tokenizer batch count mismatch")
            for document, row in zip(chunk, rows, strict=True):
                if not row or any(type(value) is not int or value < 0 for value in row):
                    raise ValueError(f"{split} tokenizer row is empty or malformed")
                if max(row) >= EXPECTED_MODEL_VOCAB:
                    raise ValueError(f"{split} content token is outside model vocabulary")
                if adapter.eos_id in row or adapter.pad_id in row:
                    raise ValueError(f"{split} content contains a reserved tokenizer token")
                batched = list(row)
                direct = adapter.encode_one(document.text)
                if batched != direct:
                    raise ValueError(f"{split} direct/batch tokenization differs")
                if adapter.decode(direct) != document.text:
                    raise ValueError(f"{split} tokenizer round-trip is not exact")
                if len(row) > 128:
                    raise ValueError(f"{split} document exceeds max sequence length")
                encoded.append(tuple(row))
        result[split] = tuple(encoded)
    return TokenizedCorpus(
        result["train"],
        result["test"],
        {
            "adapter": adapter.provenance(),
            "round_trip_exact": True,
            "stream_sha256": {
                "train": _token_stream_sha256(result["train"]),
                "test": _token_stream_sha256(result["test"]),
            },
            "token_count": {
                "train": sum(len(row) for row in result["train"]),
                "test": sum(len(row) for row in result["test"]),
            },
            "max_length": {
                "train": max(len(row) for row in result["train"]),
                "test": max(len(row) for row in result["test"]),
            },
        },
    )


def verify_published_baseline_streams(
    artifact: str | Path,
    tokenized: TokenizedCorpus,
) -> dict[str, dict[str, Any]]:
    """Rebuild the pinned Gemma shards byte-for-byte before using them as baseline."""
    root = Path(artifact).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    rows_by_split = {"train": tokenized.train, "test": tokenized.test}
    expected_counts = {
        "train": EXPECTED_PUBLISHED_TRAIN_TOKENS_WITH_EOS,
        "test": EXPECTED_PUBLISHED_TEST_TOKENS_WITH_EOS,
    }
    expected_hashes = {
        "train": EXPECTED_PUBLISHED_TRAIN_STREAM_SHA256,
        "test": EXPECTED_PUBLISHED_TEST_STREAM_SHA256,
    }
    for split, native_rows in rows_by_split.items():
        packed = np.fromiter(
            (int(token) for row in native_rows for token in (*row, EXPECTED_BASELINE_EOS)),
            dtype="<u4",
        )
        if int(packed.size) != expected_counts[split]:
            raise ValueError(f"published {split} token count mismatch")
        entries = sorted(
            (
                entry
                for entry in manifest.get("files", [])
                if entry.get("kind") == "tokens" and str(entry.get("path", "")).startswith(f"{split}/")
            ),
            key=lambda entry: str(entry["path"]),
        )
        if not entries:
            raise ValueError(f"published {split} token shards are missing")
        cursor = 0
        for entry in entries:
            count = int(entry["token_count"])
            actual = (root / str(entry["path"])).read_bytes()
            expected = np.asarray(packed[cursor : cursor + count], dtype="<u4").tobytes()
            if actual != expected or hashlib.sha256(actual).hexdigest() != entry["sha256"]:
                raise ValueError(f"published {split} token shard replay mismatch")
            cursor += count
        concatenated = np.asarray(packed, dtype="<u4").tobytes()
        digest = hashlib.sha256(concatenated).hexdigest()
        if cursor != packed.size or digest != expected_hashes[split]:
            raise ValueError(f"published {split} concatenated token hash mismatch")
        result[split] = {
            "verified": True,
            "token_count": int(packed.size),
            "sha256": digest,
            "shard_count": len(entries),
        }
    return result


def build_source_blocks(
    rows: tuple[int, ...],
    *,
    seed: int,
    epochs: int = EPOCHS,
    block_size: int = BLOCK_SIZE,
) -> list[tuple[int, ...]]:
    if len(rows) != EXPECTED_TRAIN_ROWS or sorted(rows) != list(range(EXPECTED_TRAIN_ROWS)):
        raise ValueError("source rows must be the exact ordered train indices")
    if epochs != EPOCHS or block_size != BLOCK_SIZE or seed not in EXPECTED_SEEDS:
        raise ValueError("source block plan differs from the pinned protocol")
    result: list[tuple[int, ...]] = []
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch * EPOCH_SEED_STRIDE)
        permutation = tuple(int(value) for value in torch.randperm(len(rows), generator=generator))
        result.extend(
            tuple(permutation[start : start + block_size])
            for start in range(0, len(permutation), block_size)
        )
    return result


def build_padded_exposure(documents: tuple[tuple[int, ...], ...], *, eos_id: int, pad_id: int, vocab_size: int) -> PaddedExposure:
    if not documents or any(not row for row in documents):
        raise ValueError("exposure documents must be non-empty")
    lengths = tuple(len(row) for row in documents)
    width = max(lengths)
    inputs = torch.full((len(documents), width), pad_id, dtype=torch.long)
    targets = torch.full((len(documents), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(documents), width), dtype=torch.bool)
    for index, row in enumerate(documents):
        if eos_id in row:
            raise ValueError("content token equals EOS")
        if min(row) < 0 or max(row) >= vocab_size or pad_id in row:
            raise ValueError("content token is outside the model or equals PAD")
        inputs[index, 0] = eos_id
        inputs[index, 1:len(row)] = torch.tensor(row[:-1], dtype=torch.long)
        targets[index, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, :len(row)] = True
    return PaddedExposure(inputs, targets, mask, lengths, sum(lengths))


def byte_normalized_loss(output: Any, block_bytes: int, *, expected_tokens: int | None = None) -> torch.Tensor:
    ce = getattr(output, "ce", None)
    count = int(getattr(output, "n_tokens", 0))
    if ce is None or count < 1 or block_bytes < 1 or (expected_tokens is not None and count != expected_tokens):
        raise ValueError("byte-normalized loss count mismatch")
    if not bool(torch.isfinite(ce)):
        raise ValueError("byte-normalized loss is non-finite")
    return ce * count / block_bytes


def _partition_parameters(model: nn.Module, cfg: Config) -> list[dict[str, Any]]:
    channel_ids = {id(parameter) for parameter in _muon_parameters(model)}
    body: list[nn.Parameter] = []
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in channel_ids:
            body.append(parameter)
        elif parameter.ndim >= 2 and "norm" not in name.lower():
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    assigned = [id(parameter) for group in (body, decay, no_decay) for parameter in group]
    if len(assigned) != len(set(assigned)) or set(assigned) != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
        raise RuntimeError("optimizer parameter partition is not a bijection")
    groups: list[dict[str, Any]] = []
    if body:
        groups.append({"params": body, "weight_decay": cfg.train.adam.weight_decay, "lr": cfg.train.learning_rate * cfg.train.adam.body_lr_scale, "_body": True, "fused": False})
    if decay:
        groups.append({"params": decay, "weight_decay": cfg.train.adam.weight_decay, "lr": cfg.train.learning_rate, "fused": False})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "lr": cfg.train.learning_rate, "fused": False})
    return groups


class ByteNormalizedTrainer:
    """Opt-in FP32 AdamW trainer for one fixed source-byte block."""

    def __init__(self, model: nn.Module, cfg: Config, *, max_steps: int = EXPECTED_TOTAL_UPDATES) -> None:
        if cfg.train.precision != "fp32" or cfg.train.use_muon or cfg.train.compile_model or cfg.model.head.unigram_prior or cfg.model.head.sampled_softmax_k:
            raise ValueError("byte-normalized trainer requires the pinned production-isolated options")
        self.model = model
        self.cfg = cfg
        self.step = 0
        self.max_steps = int(max_steps)
        self.optimizer = torch.optim.AdamW(
            _partition_parameters(model, cfg), lr=cfg.train.learning_rate,
            betas=(cfg.train.adam.beta1, cfg.train.adam.beta2), eps=cfg.train.adam.eps, fused=False,
        )

    def train_step(self, exposure: PaddedExposure, *, block_bytes: int) -> dict[str, Any]:
        self.model.train()
        device = next(self.model.parameters()).device
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(exposure.input_ids.to(device), exposure.targets.to(device), loss_mask=exposure.loss_mask.to(device))
        objective = byte_normalized_loss(output, block_bytes, expected_tokens=exposure.native_tokens)
        objective.backward()
        pre_step_grad_norm = float(
            torch.linalg.vector_norm(
                torch.stack(
                    [
                        parameter.grad.detach().float().norm()
                        for parameter in self.model.parameters()
                        if parameter.requires_grad
                    ]
                )
            )
        )
        gradients_finite = all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        metrics_finite = bool(torch.isfinite(objective).item()) and math.isfinite(pre_step_grad_norm)
        if not (gradients_finite and metrics_finite):
            self.optimizer.zero_grad(set_to_none=True)
            return {
                "trainer_step": self.step,
                "update_applied": False,
                "loss": float(objective.detach().cpu()),
                "pre_step_grad_norm": pre_step_grad_norm,
                "metrics_finite": metrics_finite,
                "gradients_finite": gradients_finite,
            }
        base_lr = learning_rate_at(self.step, self.cfg.train.learning_rate, self.cfg)
        for group in self.optimizer.param_groups:
            group["lr"] = base_lr * (self.cfg.train.adam.body_lr_scale if group.get("_body") else 1.0)
        self.optimizer.step()
        if hasattr(self.model, "commit_controller_updates"):
            self.model.commit_controller_updates()
        post_update_finite = full_state_finite(self.model, self.optimizer)
        if not post_update_finite:
            raise RuntimeError("post-update model or optimizer state is non-finite")
        record = {
            "trainer_step": self.step,
            "update_applied": True,
            "loss": float(objective.detach().cpu()),
            "ce": float(output.ce.detach().cpu()),
            "pre_step_grad_norm": pre_step_grad_norm,
            "lr": base_lr,
            "metrics_finite": metrics_finite,
            "gradients_finite": gradients_finite,
            "post_update_state_finite": post_update_finite,
        }
        self.step += 1
        return record


def make_matched_config(*, seed: int, tokenizer_name: str, max_updates: int = EXPECTED_TOTAL_UPDATES) -> Config:
    cfg = Config()
    model = cfg.model
    model.vocab_size = EXPECTED_MODEL_VOCAB
    model.hidden_size = 128
    model.num_layers = 4
    model.loop_depth = 1
    model.init_seed = seed
    model.init_orthogonal = False
    model.attention.num_query_heads = 4
    model.attention.num_kv_heads = 2
    model.attention.head_dim = 32
    model.attention.max_seq_len = 128
    model.attention.sink_len = 0
    model.sliding.window = 0
    model.embedding.tie_lm_head = True
    model.embedding.conv_kernel = 4
    model.embedding.init_std = 0.02
    model.ffn.intermediate_size = 256
    model.ffn.multiple_of = 32
    model.ternary.enabled = True
    model.head.unigram_prior = False
    model.head.z_loss_weight = 0.0
    model.head.ce_chunk_rows = 64
    model.head.ce_save_logits = False
    model.head.sampled_softmax_k = 0
    model.head.logit_scale_max = 0.0
    train = cfg.train
    train.batch_size = BLOCK_SIZE
    train.grad_accum_steps = 1
    train.max_steps = int(max_updates)
    train.learning_rate = 0.003
    # The byte-normalized research objective deliberately has no global gradient
    # clip; Config requires a positive production default even though this
    # isolated trainer does not consume it.
    train.max_grad_norm = 1.0
    train.schedule.warmup_steps = 20
    train.schedule.decay_fraction = 0.2
    train.schedule.min_lr_ratio = 0.02
    train.schedule.inverse_sqrt_stable = True
    train.schedule.inverse_sqrt_tau = 0
    train.precision = "fp32"
    train.grad_checkpointing = False
    train.compile_model = False
    train.use_muon = False
    train.ternary_step_cache = False
    train.ce_keep_rate = 1.0
    train.adam.body_lr_scale = 1.0
    train.adam.weight_decay = 0.01
    train.z_loss_weight = 0.0
    train.tokenizer = tokenizer_name
    train.data.seq_len = 128
    train.data.eos_token_id = (
        EXPECTED_BASELINE_EOS
        if tokenizer_name in {EXPECTED_BASELINE_NAME, "common"}
        else EXPECTED_CANDIDATE_EOS
    )
    train.data.pad_token_id = (
        0 if tokenizer_name in {EXPECTED_BASELINE_NAME, "common"} else EXPECTED_CANDIDATE_PAD
    )
    validate_config(cfg)
    return cfg


def _digest_value(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T" + str(tensor.dtype).encode() + np.asarray(tensor.shape, dtype=np.int64).tobytes() + tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"A" + str(array.dtype).encode() + np.asarray(array.shape, dtype=np.int64).tobytes() + array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"D")
        for key in sorted(value):
            encoded = str(key).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little") + encoded)
            _digest_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"L" + len(value).to_bytes(8, "little"))
        for item in value:
            _digest_value(digest, item)
    else:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        digest.update(b"S" + len(encoded).to_bytes(8, "little") + encoded)


def state_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little") + encoded)
        _digest_value(digest, state[name])
    return digest.hexdigest()


def _finite_state(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_state(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_state(item) for item in value)
    if isinstance(value, Real):
        return math.isfinite(float(value))
    return True


def full_state_finite(model: nn.Module, optimizer: torch.optim.Optimizer) -> bool:
    return _finite_state(model.state_dict()) and _finite_state(optimizer.state_dict())


def _common_state_hash(model: HAGI) -> str:
    return state_digest({name: tensor for name, tensor in model.state_dict().items() if not name.startswith("decision_head.")})


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(
            value.detach().cpu().item()
            if value.numel() == 1
            else value.detach().cpu().tolist()
        )
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (Integral, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (Real, np.floating)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("report contains non-finite JSON number")
        return number
    return value


def _write_bytes_exclusive(path: str | Path, payload: bytes) -> None:
    """Write a new regular file without replacing any existing path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_chain(target, "exclusive evidence path")
    with target.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_file_no_replace(temporary: str | Path, target: str | Path) -> None:
    """Atomically publish a complete file and never replace ``target``.

    This is the stdlib equivalent of ``atomicwrites.move_atomic(overwrite=False)``:
    a same-directory hard link appears atomically and fails with ``EEXIST`` when
    the destination exists. There is deliberately no overwrite fallback.
    """
    temporary_path = Path(temporary)
    target_path = Path(target)
    if temporary_path.parent != target_path.parent:
        raise ValueError("no-replace publication requires one source directory")
    _reject_link_chain(target_path, "publication target")
    try:
        os.link(temporary_path, target_path)
    except FileExistsError as exc:
        raise FileExistsError(f"publication target already exists: {target_path}") from exc
    # The destination is already complete. A cleanup failure must not turn a
    # successfully published artifact into a reported failure.
    try:
        temporary_path.unlink(missing_ok=True)
    except OSError:
        pass


def _write_bytes_atomic_no_replace(path: str | Path, payload: bytes) -> None:
    """Fsync a sibling temporary and publish it without overwriting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_chain(target, "atomic evidence path")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_file_no_replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_ledger(path: str | Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            for record in records:
                try:
                    payload = (
                        json.dumps(
                            _json_safe(record),
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode("utf-8")
                except ValueError as exc:
                    raise ValueError("ledger requires finite JSON values") from exc
                handle.write(payload)
                digest.update(payload)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        _publish_file_no_replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(target.resolve()), "sha256": digest.hexdigest(), "record_count": count}


def _git_status() -> str:
    result = subprocess.run(
        ["git", "status", "--short"], cwd=_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError("git status provenance unavailable")
    return result.stdout


def _source_provenance(snapshot: Path | None = None) -> dict[str, Any]:
    files: dict[str, Any] = {}
    tree = hashlib.sha256()
    for relative in PINNED_SOURCE_FILES:
        path = _ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        files[relative] = {"sha256": digest, "byte_count": len(payload)}
        tree.update(relative.encode() + b"\0" + digest.encode() + b"\n")
        if snapshot is not None:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_bytes_exclusive(target, payload)
    head = "unavailable"
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            head = result.stdout.strip()
    except OSError as exc:
        raise RuntimeError("git provenance unavailable") from exc
    return {"git_head": head, "git_status": _git_status(), "tree_sha256": tree.hexdigest(), "files": files}


def _extend_source_evidence(
    base: dict[str, Any],
    *,
    root: Path,
    baseline_tokenizer_dir: Path,
    candidate_tokenizer_dir: Path,
    snapshot: Path | None,
) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    relative_paths = tuple(
        sorted(
            {
                "manifest.json",
                *(
                    str(entry["path"])
                    for entry in manifest.get("files", [])
                    if isinstance(entry.get("path"), str) and entry["path"]
                ),
            }
        )
    )
    runtime_paths = {
        **{f"artifact/{name}": root / name for name in relative_paths},
        **{
            f"tokenizer_baseline/{name}": baseline_tokenizer_dir / name
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
        **{
            f"tokenizer_candidate/{name}": candidate_tokenizer_dir / name
            for name in EXPECTED_CANDIDATE_ASSET_SHA256
        },
    }
    files = dict(base["files"])
    tree = hashlib.sha256(base["tree_sha256"].encode("ascii"))
    for logical, path in sorted(runtime_paths.items()):
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        files[logical] = {"sha256": digest, "byte_count": len(payload)}
        tree.update(logical.encode() + b"\0" + digest.encode() + b"\n")
        if snapshot is not None:
            target = snapshot / "runtime_inputs" / logical
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_bytes_exclusive(target, payload)
    return {**base, "tree_sha256": tree.hexdigest(), "files": files}


def _device_provenance(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {"requested": str(device), "cuda": torch.cuda.is_available()}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update({
            "index": index,
            "name": properties.name,
            "capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        })
    return result


def _effective_runtime_state() -> dict[str, bool | str]:
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "flash_sdp": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp": bool(torch.backends.cuda.math_sdp_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _environment_provenance(device: torch.device) -> dict[str, Any]:
    names = (
        "CUBLAS_WORKSPACE_CONFIG",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda_runtime": torch.version.cuda,
        "selected_environment": {name: os.environ.get(name) for name in names},
        "device": _device_provenance(device),
        "effective_runtime": _effective_runtime_state(),
        "packages": {
            name: _distribution_provenance(name)
            for name in ("torch", "numpy", "gigatoken", "tokenizers")
        },
    }


def configure_deterministic_runtime(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8"):
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before CUDA initialization")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except (RuntimeError, NotImplementedError) as exc:
        raise RuntimeError("required deterministic runtime control is unsupported") from exc
    result = _effective_runtime_state()
    if not (
        result["deterministic_algorithms"]
        and not result["deterministic_warn_only"]
        and not result["tf32_matmul"]
        and not result["tf32_cudnn"]
        and result["float32_matmul_precision"] == "highest"
        and not result["flash_sdp"]
        and not result["mem_efficient_sdp"]
        and result["math_sdp"]
        and result["cublas_workspace_config"] == ":4096:8"
    ):
        raise RuntimeError(f"required deterministic runtime controls are not active: {result}")
    return result


def _config_outside_tokenizer_fields_matches(config: Config, common: Config) -> bool:
    left = asdict(config)
    right = asdict(common)
    for payload in (left, right):
        payload["train"]["tokenizer"] = "<tokenizer>"
        payload["train"]["data"]["eos_token_id"] = -1
        payload["train"]["data"]["pad_token_id"] = -1
    return left == right


def _device_operating_state(device: torch.device) -> dict[str, float | int | None]:
    if device.type != "cuda":
        return {"temperature_c": None, "power_w": None}
    index = device.index if device.index is not None else torch.cuda.current_device()
    result: dict[str, float | int | None] = {
        "temperature_c": None,
        "power_w": None,
    }
    for key, reader in (("temperature_c", torch.cuda.temperature), ("power_w", torch.cuda.power_draw)):
        try:
            value = reader(index)
        except (ImportError, RuntimeError, OSError):
            # ROCm telemetry is optional: this Windows build may omit amdsmi.
            continue
        if value is not None:
            result[key] = float(value)
    return result


def _validate_baseline_tokenizer_provenance(provenance: dict[str, Any]) -> None:
    expected = {
        "active_vocab_size": EXPECTED_BASELINE_VOCAB,
        "active_vocab_sha256": EXPECTED_BASELINE_VOCAB_SHA256,
        "merges_count": EXPECTED_BASELINE_MERGES_COUNT,
        "merges_sha256": EXPECTED_BASELINE_MERGES_SHA256,
        "fingerprint_sha256": EXPECTED_BASELINE_ACTIVE_FINGERPRINT,
        "eos_token_id": EXPECTED_BASELINE_EOS,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"baseline tokenizer active {key} mismatch")
    package = provenance.get("backend_record")
    if not isinstance(package, dict) or package.get("version") != EXPECTED_GIGATOKEN_VERSION:
        raise ValueError("baseline tokenizer package provenance mismatch")


def _write_npy_atomic(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_chain(path, "count evidence path")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_file_no_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "byte_count": path.stat().st_size,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def _prior_diagnostics(
    train: tuple[tuple[int, ...], ...],
    test: tuple[tuple[int, ...], ...],
    corpus: Corpus,
    *,
    counts_dir: Path,
    lane: str,
) -> dict[str, Any]:
    train_counts = np.bincount(
        np.fromiter((token for row in train for token in row), dtype=np.int64),
        minlength=EXPECTED_MODEL_VOCAB,
    ).astype("<u8", copy=False)
    test_counts = np.bincount(
        np.fromiter((token for row in test for token in row), dtype=np.int64),
        minlength=EXPECTED_MODEL_VOCAB,
    ).astype("<u8", copy=False)
    probabilities = (train_counts.astype(np.float64) + 1.0) / (
        float(train_counts.sum()) + EXPECTED_MODEL_VOCAB
    )
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    nll = float(-(test_counts * np.log(probabilities)).sum())
    byte_count = sum(document.utf8_bytes for document in corpus.test)
    return {
        "train_counts": _write_npy_atomic(counts_dir / f"{lane}-train.npy", train_counts),
        "test_counts": _write_npy_atomic(counts_dir / f"{lane}-test.npy", test_counts),
        "train_count_mass": int(train_counts.sum()),
        "test_count_mass": int(test_counts.sum()),
        "smoothed_prior_entropy_nats": entropy,
        "prior_only_test_nll": nll,
        "prior_only_test_bpb": nll / (byte_count * math.log(2.0)),
    }


def _exact_exposure(
    rows: tuple[tuple[int, ...], ...],
    *,
    eos_id: int,
    vocab_size: int,
) -> PaddedExposure:
    if not rows:
        raise ValueError("exact-length evaluation batch is empty")
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("exact-length evaluation batch contains mixed lengths")
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for row in rows:
        if eos_id in row or min(row) < 0 or max(row) >= vocab_size:
            raise ValueError("evaluation content contains a reserved or out-of-range token")
        inputs.append([eos_id, *row[:-1]])
        targets.append(list(row))
    return PaddedExposure(
        input_ids=torch.tensor(inputs, dtype=torch.long),
        targets=torch.tensor(targets, dtype=torch.long),
        loss_mask=torch.ones((len(rows), width), dtype=torch.bool),
        lengths=(width,) * len(rows),
        native_tokens=len(rows) * width,
    )


def _evaluate(
    model: HAGI,
    rows: tuple[tuple[int, ...], ...],
    documents: tuple[Document, ...],
    *,
    eos_id: int,
    device: torch.device,
    max_batch_size: int = 64,
) -> dict[str, Any]:
    if len(rows) != len(documents) or not rows or max_batch_size < 1:
        raise ValueError("evaluation rows/documents or batch size mismatch")
    model.eval()
    total_nll = np.float64(0.0)
    token_count = 0
    byte_count = 0
    batch_count = 0
    indices_by_length: dict[int, list[int]] = {}
    for index, (row, document) in enumerate(zip(rows, documents, strict=True)):
        if not row or len(row) > model.cfg.model.attention.max_seq_len:
            raise ValueError("evaluation document length is outside the pinned model context")
        indices_by_length.setdefault(len(row), []).append(index)
    with torch.no_grad():
        for length in sorted(indices_by_length):
            indices = indices_by_length[length]
            for start in range(0, len(indices), max_batch_size):
                selected = indices[start : start + max_batch_size]
                exposure = _exact_exposure(
                    tuple(rows[index] for index in selected),
                    eos_id=eos_id,
                    vocab_size=model.cfg.model.vocab_size,
                )
                output = model(
                    exposure.input_ids.to(device),
                    exposure.targets.to(device),
                )
                if output.ce is None or output.n_tokens != exposure.native_tokens:
                    raise ValueError("evaluation token count mismatch")
                value = float(output.ce.detach().cpu()) * exposure.native_tokens
                if not math.isfinite(value):
                    raise ValueError("evaluation produced non-finite NLL")
                total_nll += np.float64(value)
                token_count += exposure.native_tokens
                byte_count += sum(documents[index].utf8_bytes for index in selected)
                batch_count += 1
    if token_count < 1 or byte_count < 1:
        raise ValueError("evaluation selected no content or bytes")
    return {
        "total_nll": float(total_nll),
        "token_count": token_count,
        "byte_count": byte_count,
        "batch_count": batch_count,
        "native_token_ce": float(total_nll) / token_count,
        "tokens_per_byte": token_count / byte_count,
        "bits_per_byte": float(total_nll) / (byte_count * math.log(2.0)),
        "round_trip_exact": True,
    }


def _model_bytes(model: nn.Module) -> dict[str, int]:
    unique = {id(parameter): parameter for parameter in model.parameters()}
    return {"parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in unique.values()), "model_state_bytes": sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())}


def _optimizer_bytes(optimizer: torch.optim.Optimizer) -> int:
    def tensor_bytes(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(tensor_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(tensor_bytes(item) for item in value)
        return 0

    return tensor_bytes(optimizer.state_dict()["state"])


def _block_digest(
    rows: tuple[int, ...],
    documents: tuple[Document, ...],
    tokenized_train: tuple[tuple[int, ...], ...],
) -> str:
    digest = hashlib.sha256()
    for source_index in rows:
        payload = documents[source_index].text.encode("utf-8")
        digest.update(source_index.to_bytes(8, "little"))
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        for token in tokenized_train[source_index]:
            digest.update(int(token).to_bytes(4, "little"))
    return digest.hexdigest()


def run_lane(
    corpus: Corpus,
    data: TokenizedCorpus,
    *,
    seed: int,
    lane: str,
    adapter: TokenizerAdapter,
    common_state: dict[str, torch.Tensor],
    blocks: list[tuple[int, ...]],
    max_updates: int,
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    cfg = make_matched_config(seed=seed, tokenizer_name=adapter.name, max_updates=max_updates)
    operating_before = _device_operating_state(device)
    torch.manual_seed(seed)
    model = HAGI(cfg).to(device)
    model.load_state_dict(common_state, strict=True)
    initial_hash = _common_state_hash(model)
    if initial_hash != state_digest(common_state):
        raise ValueError("matched common initialization hash mismatch")
    initial = _evaluate(model, data.test, corpus.test, eos_id=adapter.eos_id, device=device)
    trainer = ByteNormalizedTrainer(model, cfg, max_steps=max_updates)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    records: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    step_times: list[float] = []
    for step, source_rows in enumerate(blocks[:max_updates]):
        encoded = tuple(data.train[index] for index in source_rows)
        encoded_with_rows = sorted(
            zip(source_rows, encoded, strict=True),
            key=lambda item: (len(item[1]), item[0]),
        )
        exposure = build_padded_exposure(
            tuple(tokens for _, tokens in encoded_with_rows),
            eos_id=adapter.eos_id,
            pad_id=cfg.train.data.pad_token_id,
            vocab_size=cfg.model.vocab_size,
        )
        block_bytes = sum(corpus.train[index].utf8_bytes for index in source_rows)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        metrics = trainer.train_step(exposure, block_bytes=block_bytes)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - started)
        record = {
            "schema_version": 1,
            "seed": seed,
            "lane": lane,
            "step": step,
            "trainer_step": metrics["trainer_step"],
            "source_rows": list(source_rows),
            "row_count": len(source_rows),
            "byte_count": block_bytes,
            "token_count": exposure.native_tokens,
            "microbatch_count": 1,
            "microbatch_token_counts": [exposure.native_tokens],
            "block_digest": _block_digest(source_rows, corpus.train, data.train),
            **metrics,
        }
        records.append(record)
        if step in {0, max_updates // 2, max_updates - 1} or (step + 1) % 128 == 0:
            finite = full_state_finite(model, trainer.optimizer)
            snapshots.append(
                {
                    "step": step,
                    "finite": finite,
                    "model_digest": state_digest(model.state_dict()),
                    "optimizer_digest": state_digest(trainer.optimizer.state_dict()),
                }
            )
            if not finite:
                raise RuntimeError("post-update model or optimizer state is non-finite")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    final = _evaluate(model, data.test, corpus.test, eos_id=adapter.eos_id, device=device)
    evaluation_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    ledger = write_jsonl_ledger(
        run_dir / "ledgers" / f"seed-{seed}-{lane}.jsonl",
        records,
    )
    memory = _model_bytes(model) | {
        "optimizer_state_bytes": _optimizer_bytes(trainer.optimizer),
        "cuda_training_peak_allocated": training_peak,
        "cuda_evaluation_peak_allocated": evaluation_peak,
    }
    expected_steps = list(range(max_updates))
    record_integrity = (
        len(records) == max_updates
        and ledger["record_count"] == max_updates
        and [int(record["step"]) for record in records] == expected_steps
        and [int(record["trainer_step"]) for record in records] == expected_steps
        and all(bool(record["update_applied"]) for record in records)
        and all(bool(record["metrics_finite"]) for record in records)
        and all(bool(record["gradients_finite"]) for record in records)
        and all(bool(record["post_update_state_finite"]) for record in records)
    )
    operating_after = _device_operating_state(device)
    config_payload = asdict(cfg)
    return {
        "name": lane,
        "seed": seed,
        "configuration": cfg,
        "configuration_sha256": state_digest(config_payload),
        "initial": initial,
        "final": final,
        "steps_completed": len(records),
        "updates_applied": sum(bool(record["update_applied"]) for record in records),
        "rejected_updates": sum(not bool(record["update_applied"]) for record in records),
        "step_times_seconds": step_times,
        "mean_post_warmup_seconds": (
            statistics.fmean(step_times[20:]) if len(step_times) > 20 else None
        ),
        "median_post_warmup_seconds": (
            statistics.median(step_times[20:]) if len(step_times) > 20 else None
        ),
        "std_post_warmup_seconds": (
            statistics.stdev(step_times[20:]) if len(step_times) > 20 else None
        ),
        "p95_post_warmup_seconds": (
            sorted(step_times[20:])[
                max(0, math.ceil(0.95 * len(step_times[20:])) - 1)
            ]
            if len(step_times) > 20
            else None
        ),
        "ledger": ledger,
        "state_checkpoints": snapshots,
        "memory": memory,
        "device_operating_state": {"before": operating_before, "after": operating_after},
        "initial_common_state_sha256": initial_hash,
        "record_integrity": record_integrity,
        "finite": record_integrity and all(item["finite"] for item in snapshots),
    }


def _protocol_metadata(
    *,
    requested_seeds: list[int],
    updates_per_seed: int,
    full_protocol_executed: bool,
) -> dict[str, Any]:
    return {
        "requested_protocol": {
            "seeds": list(requested_seeds),
            "updates_per_seed": int(updates_per_seed),
        },
        "target_protocol": {
            "seeds": list(EXPECTED_SEEDS),
            "updates_per_seed": EXPECTED_TOTAL_UPDATES,
        },
        "full_protocol_executed": bool(full_protocol_executed),
    }


def evaluate_gate_input(
    *,
    seeds: list[int],
    updates_per_seed: list[int],
    bpb_improvements: list[float],
    timing_ratios: list[float],
    memory_ratios: list[float],
    roundtrip_exact: bool,
    runs_finite: bool,
    matched_configs: bool,
    provenance_stable: bool,
    deterministic: bool,
) -> dict[str, Any]:
    expected_count = len(seeds)
    vector_shapes_valid = (
        expected_count == 3
        and len(updates_per_seed) == expected_count
        and len(bpb_improvements) == expected_count
        and len(timing_ratios) == expected_count
        and len(memory_ratios) == expected_count
    )
    exact_protocol = (
        seeds == list(EXPECTED_SEEDS)
        and updates_per_seed == [EXPECTED_TOTAL_UPDATES] * 3
    )
    finite_vectors = vector_shapes_valid and all(
        math.isfinite(float(value)) for value in bpb_improvements + timing_ratios + memory_ratios
    )
    systems_ratios_positive = vector_shapes_valid and all(
        float(value) > 0.0 for value in timing_ratios + memory_ratios
    )
    correctness = bool(
        vector_shapes_valid
        and finite_vectors
        and roundtrip_exact
        and runs_finite
        and matched_configs
    )
    provenance = bool(provenance_stable and deterministic)
    execution_valid = bool(exact_protocol and correctness and provenance)
    checks = {
        "vector_shapes_valid": vector_shapes_valid,
        "exact_protocol": exact_protocol,
        "exact_roundtrip": roundtrip_exact,
        "current_runs_finite": runs_finite,
        "six_complete_finite_runs": exact_protocol and runs_finite,
        "matched_configs": matched_configs,
        "bpb_all_seeds_positive": vector_shapes_valid
        and all(value > 0.0 for value in bpb_improvements),
        "bpb_median": vector_shapes_valid
        and statistics.median(bpb_improvements) >= 0.01,
        "bpb_mean": vector_shapes_valid
        and statistics.fmean(bpb_improvements) >= 0.005,
        "systems_ratios_positive": systems_ratios_positive,
        "timing": vector_shapes_valid
        and systems_ratios_positive
        and all(value <= 1.25 for value in timing_ratios),
        "memory": vector_shapes_valid
        and systems_ratios_positive
        and all(value <= 1.25 for value in memory_ratios),
        "provenance_stable": provenance_stable,
        "deterministic": deterministic,
    }
    scientific_checks = (
        "vector_shapes_valid",
        "exact_protocol",
        "exact_roundtrip",
        "six_complete_finite_runs",
        "matched_configs",
        "bpb_all_seeds_positive",
        "bpb_median",
        "bpb_mean",
        "provenance_stable",
        "deterministic",
    )
    systems_checks = ("systems_ratios_positive", "timing", "memory")
    quality_supported = execution_valid and all(checks[name] for name in scientific_checks)
    systems_supported = execution_valid and all(checks[name] for name in systems_checks)
    eligible = quality_supported and systems_supported
    return {
        "status": "eligible-for-next-slice" if eligible else "research-only",
        "execution_valid": execution_valid,
        "quality_supported": quality_supported,
        "systems_supported": systems_supported,
        "checks": checks,
        "bpb_improvements": bpb_improvements,
        "mean_bpb_improvement": (
            statistics.fmean(bpb_improvements) if bpb_improvements else 0.0
        ),
        "median_bpb_improvement": (
            statistics.median(bpb_improvements) if bpb_improvements else 0.0
        ),
        "timing_ratios": timing_ratios,
        "memory_ratios": memory_ratios,
        "production_promotion": False,
    }


def _lane_order(seed: int) -> tuple[str, str]:
    if seed not in LANE_ORDER:
        raise ValueError("non-pinned seed cannot use the eligibility protocol")
    return LANE_ORDER[seed]


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    runtime = configure_deterministic_runtime(device)
    environment_before = _environment_provenance(device)
    source_before = _source_provenance(Path(args.run) / "source_snapshot")
    corpus = load_corpus(args.artifact)
    document_stream_before = _document_stream_hashes(corpus)
    adapters = {
        "baseline": create_baseline_adapter(args.baseline_tokenizer),
        "candidate": create_candidate_adapter(args.candidate_tokenizer),
    }
    baseline_provenance = adapters["baseline"].provenance()
    _validate_baseline_tokenizer_provenance(baseline_provenance)
    source_before = _extend_source_evidence(
        source_before,
        root=Path(args.artifact).resolve(strict=True),
        baseline_tokenizer_dir=Path(args.baseline_tokenizer).resolve(strict=True),
        candidate_tokenizer_dir=Path(args.candidate_tokenizer).resolve(strict=True),
        snapshot=Path(args.run) / "source_snapshot",
    )

    tokenized = {
        name: tokenize_corpus(corpus, adapter)
        for name, adapter in adapters.items()
    }
    baseline_stream_replay = verify_published_baseline_streams(
        args.artifact,
        tokenized["baseline"],
    )
    observed_token_counts = {
        "baseline": {
            "train": sum(len(row) for row in tokenized["baseline"].train),
            "test": sum(len(row) for row in tokenized["baseline"].test),
        },
        "candidate": {
            "train": sum(len(row) for row in tokenized["candidate"].train),
            "test": sum(len(row) for row in tokenized["candidate"].test),
        },
    }
    expected_token_counts = {
        "baseline": {
            "train": EXPECTED_BASELINE_NATIVE_TRAIN_TOKENS,
            "test": EXPECTED_BASELINE_NATIVE_TEST_TOKENS,
        },
        "candidate": {
            "train": EXPECTED_CANDIDATE_NATIVE_TRAIN_TOKENS,
            "test": EXPECTED_CANDIDATE_NATIVE_TEST_TOKENS,
        },
    }
    if observed_token_counts != expected_token_counts:
        raise ValueError(
            f"pinned tokenizer content token count mismatch: "
            f"{observed_token_counts!r} != {expected_token_counts!r}"
        )
    if (
        observed_token_counts["baseline"]["train"] + EXPECTED_TRAIN_ROWS
        != EXPECTED_PUBLISHED_TRAIN_TOKENS_WITH_EOS
        or observed_token_counts["baseline"]["test"] + EXPECTED_TEST_ROWS
        != EXPECTED_PUBLISHED_TEST_TOKENS_WITH_EOS
    ):
        raise ValueError("published baseline counts are not content plus one EOS per row")

    prior = {
        name: _prior_diagnostics(
            data.train,
            data.test,
            corpus,
            counts_dir=Path(args.run) / "counts",
            lane=name,
        )
        for name, data in tokenized.items()
    }
    matched_configs = True
    results: list[dict[str, Any]] = []
    for seed in args.seeds:
        order = _lane_order(seed)
        blocks = build_source_blocks(tuple(range(EXPECTED_TRAIN_ROWS)), seed=seed)
        torch.manual_seed(seed)
        common_config = make_matched_config(
            seed=seed,
            tokenizer_name="common",
            max_updates=args.max_updates_per_seed,
        )
        common_model = HAGI(common_config)
        common_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in common_model.state_dict().items()
        }
        common_state_sha256 = state_digest(common_state)
        del common_model
        lanes: dict[str, Any] = {}
        for lane in order:
            lanes[lane] = run_lane(
                corpus,
                tokenized[lane],
                seed=seed,
                lane=lane,
                adapter=adapters[lane],
                common_state=common_state,
                blocks=blocks,
                max_updates=args.max_updates_per_seed,
                device=device,
                run_dir=Path(args.run),
            )
            if lanes[lane]["initial_common_state_sha256"] != common_state_sha256:
                raise ValueError(f"seed {seed} lane {lane} common-state hash mismatch")
            if not _config_outside_tokenizer_fields_matches(
                lanes[lane]["configuration"],
                common_config,
            ):
                matched_configs = False
                raise ValueError(
                    f"seed {seed} lane {lane} config differs outside tokenizer/EOS/PAD"
                )
        results.append(
            {
                "seed": seed,
                "lane_order": list(order),
                "source_block_plan_sha256": state_digest({"blocks": blocks}),
                "common_state_sha256": common_state_sha256,
                "lanes": lanes,
            }
        )

    source_after = _extend_source_evidence(
        _source_provenance(),
        root=Path(args.artifact).resolve(strict=True),
        baseline_tokenizer_dir=Path(args.baseline_tokenizer).resolve(strict=True),
        candidate_tokenizer_dir=Path(args.candidate_tokenizer).resolve(strict=True),
        snapshot=None,
    )
    environment_after = _environment_provenance(device)
    document_stream_after = _document_stream_hashes(corpus)
    document_stream_stable = document_stream_before == document_stream_after
    source_stable = source_before["tree_sha256"] == source_after["tree_sha256"]
    environment_stable = environment_before == environment_after
    provenance_stable = (
        source_stable and environment_stable and document_stream_stable
    )

    improvements = [
        float(item["lanes"]["baseline"]["final"]["bits_per_byte"])
        - float(item["lanes"]["candidate"]["final"]["bits_per_byte"])
        for item in results
    ]
    failure_ratio = 1.0e300
    timing_ratios: list[float] = []
    memory_ratios: list[float] = []
    for item in results:
        candidate_time = item["lanes"]["candidate"]["mean_post_warmup_seconds"]
        baseline_time = item["lanes"]["baseline"]["mean_post_warmup_seconds"]
        timing_ratios.append(
            float(candidate_time) / float(baseline_time)
            if candidate_time is not None
            and baseline_time is not None
            and float(candidate_time) > 0.0
            and float(baseline_time) > 0.0
            else failure_ratio
        )
        candidate_peak = int(
            item["lanes"]["candidate"]["memory"]["cuda_training_peak_allocated"]
        )
        baseline_peak = int(
            item["lanes"]["baseline"]["memory"]["cuda_training_peak_allocated"]
        )
        memory_ratios.append(
            candidate_peak / baseline_peak
            if candidate_peak > 0 and baseline_peak > 0
            else failure_ratio
        )

    updates_per_seed = [
        min(
            int(item["lanes"]["baseline"]["steps_completed"]),
            int(item["lanes"]["candidate"]["steps_completed"]),
        )
        for item in results
    ]
    gate = evaluate_gate_input(
        seeds=list(args.seeds),
        updates_per_seed=updates_per_seed,
        bpb_improvements=improvements,
        timing_ratios=timing_ratios,
        memory_ratios=memory_ratios,
        roundtrip_exact=all(
            item.provenance["round_trip_exact"] for item in tokenized.values()
        ),
        runs_finite=all(
            lane["finite"]
            for item in results
            for lane in item["lanes"].values()
        ),
        matched_configs=matched_configs,
        provenance_stable=provenance_stable,
        deterministic=runtime["deterministic_algorithms"],
    )
    return {
        "schema_version": 1,
        "experiment": "banking77-tokenizer-frontier",
        "execution_valid": gate["execution_valid"],
        "quality_supported": gate["quality_supported"],
        "systems_supported": gate["systems_supported"],
        "status": gate["status"],
        "production_promotion": False,
        "gate": gate,
        "protocol": {
            "plan": ".omc/plans/tokenizer_banking77.md",
            **_protocol_metadata(
                requested_seeds=list(args.seeds),
                updates_per_seed=int(args.max_updates_per_seed),
                full_protocol_executed=bool(gate["checks"]["exact_protocol"]),
            ),
            "epochs": EPOCHS,
            "block_size": BLOCK_SIZE,
            "objective": "sum_native_nll_divided_by_exact_utf8_block_bytes",
            "gradient_clipping_enabled": False,
            "estimand": "no_prior_same_from_scratch_segmentation_screen",
            "unigram_prior_applied": False,
            "primary_metric": "same_text_bits_per_byte",
            "all_three_bpb_deltas_required_positive": True,
            "minimum_median_bpb_improvement": 0.01,
            "minimum_mean_bpb_improvement": 0.005,
            "maximum_timing_ratio": 1.25,
            "maximum_memory_ratio": 1.25,
        },
        "runtime": runtime,
        "environment_before": environment_before,
        "environment_after": environment_after,
        "environment_stable": environment_stable,
        "environment": environment_after,
        "artifact": {
            **corpus.provenance,
            "published_baseline_stream_replay": baseline_stream_replay,
        },
        "document_stream_sha256": document_stream_before,
        "document_stream_after_sha256": document_stream_after,
        "document_stream_stable": document_stream_stable,
        "train_native_tokens": {
            name: observed_token_counts[name]["train"]
            for name in observed_token_counts
        },
        "test_native_tokens": {
            name: observed_token_counts[name]["test"]
            for name in observed_token_counts
        },
        "train_fertility": {
            name: observed_token_counts[name]["train"]
            / corpus.provenance["train_content_bytes"]
            for name in observed_token_counts
        },
        "test_fertility": {
            name: observed_token_counts[name]["test"]
            / corpus.provenance["test_content_bytes"]
            for name in observed_token_counts
        },
        "tokenizer": {
            "baseline": {
                "provenance": baseline_provenance,
                "corpus": tokenized["baseline"].provenance,
            },
            "candidate": {
                "provenance": adapters["candidate"].provenance(),
                "corpus": tokenized["candidate"].provenance,
            },
        },
        "prior_diagnostics": prior,
        "source_before": source_before,
        "source_after": source_after,
        "source_stable": source_stable,
        "seeds": results,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Config):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported report type: {type(value)!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--baseline-tokenizer",
        type=Path,
        required=True,
        help="local pinned Gemma tokenizer snapshot directory; model IDs are rejected",
    )
    parser.add_argument(
        "--candidate-tokenizer",
        type=Path,
        required=True,
        help="local pinned candidate asset directory; no network identity is inferred",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EXPECTED_SEEDS))
    parser.add_argument("--max-updates-per-seed", type=int, default=EXPECTED_TOTAL_UPDATES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(seed not in EXPECTED_SEEDS for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique pinned values")
    if not 1 <= args.max_updates_per_seed <= EXPECTED_TOTAL_UPDATES:
        parser.error("max-updates-per-seed must be in [1, 471]")
    if args.device not in {"cpu", "cuda"}:
        parser.error("device must be cpu or cuda")
    return args


def prepare_run_paths(
    run: str | Path, output: str | Path
) -> tuple[Path, Path, Path]:
    """Exclusively reserve a clean run directory and report output.

    The run leaf is created with ``exist_ok=False``. The report lock uses
    ``O_CREAT | O_EXCL`` so concurrent invocations cannot share final paths.
    See https://docs.python.org/3/library/os.html#os.open and
    https://docs.python.org/3/library/os.html#os.mkdir.
    """
    run_path = Path(os.path.abspath(run))
    output_path = Path(os.path.abspath(output))
    if run_path == output_path:
        raise ValueError("run and report output paths must differ")
    if output_path.parent != run_path:
        raise ValueError("report output must be a direct child of the reserved run directory")
    _reject_link_chain(run_path, "tokenizer run path")
    _reject_link_chain(output_path, "tokenizer report path")
    if output_path.exists():
        raise FileExistsError(f"report output already exists: {output_path}")
    if run_path.exists():
        raise FileExistsError(f"run path is not exclusively reservable: {run_path}")

    run_path.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_chain(run_path.parent, "tokenizer run parent")

    lock_path = output_path.with_name(f".{output_path.name}.lock")
    descriptor: int | None = None
    try:
        run_path.mkdir(exist_ok=False)
        _reject_link_chain(run_path, "tokenizer run path")
        _reject_link_chain(lock_path, "tokenizer report lock")
        descriptor = os.open(
            lock_path,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_BINARY", 0),
            0o600,
        )
        payload = json.dumps(
            {"pid": os.getpid(), "run_path": str(run_path)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        os.write(descriptor, payload + b"\n")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
    except BaseException:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException:
                pass
        try:
            lock_path.unlink(missing_ok=True)
        except BaseException:
            pass
        try:
            run_path.rmdir()
        except BaseException:
            pass
        raise
    return run_path, output_path, lock_path


def assert_output_reserved(
    output: str | Path, lock: str | Path, run: str | Path
) -> None:
    """Verify the report reservation survived inside the exclusive run root."""
    output_path = Path(output)
    lock_path = Path(lock)
    run_path = Path(run)
    if output_path.parent != run_path or lock_path.parent != run_path:
        raise ValueError("report and reservation lock must remain inside the run root")
    _reject_link_chain(run_path, "tokenizer run path")
    _reject_link_chain(output_path, "tokenizer report path")
    _reject_link_chain(lock_path, "tokenizer report lock")
    if output_path.exists():
        raise FileExistsError(f"report output appeared during run: {output_path}")
    if not lock_path.is_file() or lock_path.is_symlink():
        raise FileNotFoundError(f"report reservation lock is missing: {lock_path}")


def _report_payload(report: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            _json_safe(report),
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def _regular_file_sha256_or_none(path: str | Path) -> str | None:
    try:
        candidate = Path(path)
        _reject_link_chain(candidate, "terminal evidence path")
        if not candidate.is_file() or candidate.is_symlink():
            return None
        return _sha256_file(candidate)
    except (OSError, ValueError):
        return None


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"report output already exists: {target}")
    _reject_link_chain(target, "tokenizer report path")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _report_payload(report)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_file_no_replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_failure_status(
    output: str | Path,
    run: str | Path,
    lock: str | Path,
    error: BaseException,
) -> Path:
    """Preserve a failed run with an explicit non-quality terminal marker."""
    output_path = Path(output)
    run_path = Path(run)
    lock_path = Path(lock)
    if output_path.parent != run_path or lock_path.parent != run_path:
        raise ValueError("failure evidence must remain inside the run root")
    if not lock_path.is_file():
        raise ValueError("cannot write failure status without the run reservation lock")
    failure_path = run_path / "failure.json"
    payload = {
        "schema_version": 1,
        "experiment": "banking77-tokenizer-frontier",
        "status": "failed-before-report",
        "quality_supported": False,
        "systems_supported": False,
        "production_promotion": False,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "pid": os.getpid(),
    }
    _write_bytes_atomic_no_replace(
        failure_path,
        (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    return failure_path


def _remove_reservation_lock(lock: str | Path) -> None:
    """Best-effort cleanup; report/failure evidence remains authoritative."""
    try:
        Path(lock).unlink(missing_ok=True)
    except BaseException:
        # The complete report or failure marker, not the advisory lock, is the
        # terminal evidence. Lock cleanup is best effort after that boundary.
        pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.run, args.output, report_lock = prepare_run_paths(args.run, args.output)
    try:
        report = run_experiment(args)
        assert_output_reserved(args.output, report_lock, args.run)
    except BaseException as error:
        try:
            write_failure_status(args.output, args.run, report_lock, error)
        except BaseException as status_error:
            error.add_note(f"failed to write terminal run status: {status_error}")
        else:
            _remove_reservation_lock(report_lock)
        raise

    expected_report_sha256 = hashlib.sha256(_report_payload(report)).hexdigest()
    try:
        write_report(args.output, report)
    except BaseException as error:
        if _regular_file_sha256_or_none(args.output) != expected_report_sha256:
            try:
                write_failure_status(args.output, args.run, report_lock, error)
            except BaseException as status_error:
                error.add_note(f"failed to write terminal run status: {status_error}")
            else:
                _remove_reservation_lock(report_lock)
            raise
        # The report hard link is the atomic success boundary. A signal after
        # that syscall must not create a contradictory failure marker.
    _remove_reservation_lock(report_lock)
    return 0 if report["execution_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
