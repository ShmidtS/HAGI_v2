"""Real Banking77 gate for the opt-in finite-option DecisionPlane.

This runner is deliberately separate from ``decision_plane_ab.py`` and from the
production packed-data loader. Banking77 is a document-level classification
corpus: each official row owns one finite-option label, while a packed LM window
may cross document boundaries. The experiment therefore uses exact-length
microbatches, no padding and no truncation, and reads the final content-token
state through ``DecisionHead``.

The full gate is pre-registered in ``.omc/plans/decision_banking77.md`` before
implementation or result inspection. Reduced rows, fewer than three seeds, or
a different protocol always report ``quality_supported=false``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import distribution, version
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _path in (_SRC, _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from hagi.config import Config, validate_config  # noqa: E402
from hagi.data.artifacts import load_published_artifact, write_json_atomic  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.loop import Trainer  # noqa: E402
from scripts.decision_plane_ab import _common_state, decision_metrics, load_common_state  # noqa: E402

EXPECTED_ARTIFACT_ID = "banking77"
EXPECTED_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
EXPECTED_DATASET = "PolyAI-LDN/task-specific-datasets/banking_data"
EXPECTED_LICENSE = "CC-BY-4.0"
EXPECTED_TOKENIZER = "google/gemma-4-E2B-it"
EXPECTED_GIGATOKEN_VERSION = "0.10.0"
EXPECTED_VOCAB_SIZE = 262_144
EXPECTED_MANIFEST_SHA256 = "44d50edd994e7c32f30066a26052e15f902c74b79a7498ba60f475c76b453f3b"
EXPECTED_TRAIN_TEXT_SHA256 = "881989b4e2ef96bb3f2981fbf3765cfea0efddb30a784ed2c4709ad77be524e4"
EXPECTED_TEST_TEXT_SHA256 = "b4148261b6025a32f2c8cd3318ae48bc6225f74603ae022346c9ca371f29a06c"
EXPECTED_TRAIN_ROWS = 10_003
EXPECTED_TEST_ROWS = 3_080
EXPECTED_CATEGORIES = 77
MAX_ECE = 0.15
MAX_ACCURACY_REGRESSION = 0.0
SEEDS = (1234, 2243, 3252)
PINNED_SHARD_TOKENS = 100_000
LANE_SPECS = ("majority_or_uniform", "frozen_probe", "decision_only", "end_to_end")
PINNED_SOURCE_FILES = (
    ".omc/plans/decision_banking77.md",
    "pyproject.toml",
    "scripts/decision_plane_banking77.py",
    "scripts/decision_plane_ab.py",
    "scripts/prepare_banking77.py",
    "src/hagi/config.py",
    "src/hagi/data/artifacts.py",
    "src/hagi/model/adaptive.py",
    "src/hagi/model/attention.py",
    "src/hagi/model/block.py",
    "src/hagi/model/cortex.py",
    "src/hagi/model/decision.py",
    "src/hagi/model/embedding.py",
    "src/hagi/model/ffn.py",
    "src/hagi/model/head.py",
    "src/hagi/model/kv_cache.py",
    "src/hagi/model/model.py",
    "src/hagi/model/multimodal.py",
    "src/hagi/model/norms.py",
    "src/hagi/model/outputs.py",
    "src/hagi/model/rope.py",
    "src/hagi/model/ternary.py",
    "src/hagi/train/loop.py",
    "src/hagi/train/optim.py",
    "src/hagi/version.py",
    "tests/test_decision_plane.py",
    "tests/test_decision_plane_ab.py",
    "tests/test_decision_plane_banking77.py",
    "tests/test_loop.py",
)


@dataclass(frozen=True)
class DecisionDocument:
    """One exact-length compact-token document and its finite option."""

    content_ids: tuple[int, ...]
    label: int


@dataclass(frozen=True)
class DecisionData:
    """Validated Banking77 data reduced to a deterministic experiment contract."""

    train: tuple[DecisionDocument, ...]
    test: tuple[DecisionDocument, ...]
    num_options: int
    native_vocab_size: int
    train_native_unique: int
    test_token_count: int
    test_oov_token_count: int
    manifest_sha256: str
    train_text_sha256: str
    test_text_sha256: str
    tokenizer_version: str
    tokenizer_fingerprint: dict[str, Any]
    quality_eligible: bool
    pinned_token_streams_verified: bool = False
    pinned_token_stream_metadata: tuple[tuple[str, dict[str, object]], ...] = ()


@dataclass(frozen=True)
class TrainVocabulary:
    """Train-only compact vocabulary with explicit PAD/EOS/UNK ownership."""

    old_to_new: np.ndarray
    new_to_old: np.ndarray
    counts: np.ndarray
    pad_id: int = 0
    eos_id: int = 1
    unk_id: int = 2
    sha256: str = ""

    @property
    def vocab_size(self) -> int:
        return int(self.new_to_old.shape[0])

    def map_ids(self, values: list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.int64)
        if array.ndim != 1:
            raise ValueError("token ids must be one-dimensional")
        if array.size and (array.min() < 0 or array.max() >= self.old_to_new.shape[0]):
            raise ValueError("native token id is outside the declared vocabulary")
        mapped = self.old_to_new[array]
        mapped = np.where(mapped < 0, self.unk_id, mapped)
        return mapped.astype(np.int64)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_provenance(snapshot_dir: Path | None = None) -> dict[str, Any]:
    """Hash the exact protocol/source files and optionally preserve their bytes."""
    files: dict[str, dict[str, Any]] = {}
    tree_digest = hashlib.sha256()
    for relative in sorted(PINNED_SOURCE_FILES):
        source = _ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"pinned experiment source is missing: {relative}")
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        files[relative] = {"sha256": digest, "byte_count": len(payload)}
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\n")
        if snapshot_dir is not None:
            target = snapshot_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
    git_head = "unavailable"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            git_head = result.stdout.strip()
    except OSError:
        pass
    return {
        "git_head": git_head,
        "tree_sha256": tree_digest.hexdigest(),
        "files": files,
        "snapshot_dir": str(snapshot_dir) if snapshot_dir is not None else None,
    }


def _hash_length_prefixed_bytes(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "little"))
    digest.update(value)


def _tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    """Hash the exact active native vocabulary and merge table."""
    backend = tokenizer._backend
    vocab = backend.vocab
    merges = backend.merges
    if not isinstance(vocab, dict) or not isinstance(merges, list):
        raise ValueError("tokenizer backend must expose dict vocab and list merges")
    vocab_digest = hashlib.sha256()
    for token_id in sorted(int(key) for key in vocab):
        piece = vocab[token_id]
        if not isinstance(piece, bytes):
            raise ValueError("tokenizer vocabulary pieces must be bytes")
        vocab_digest.update(token_id.to_bytes(8, "little", signed=False))
        _hash_length_prefixed_bytes(vocab_digest, piece)
    merges_digest = hashlib.sha256()
    for pair in merges:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("tokenizer merges must be ordered byte pairs")
        _hash_length_prefixed_bytes(merges_digest, pair[0])
        _hash_length_prefixed_bytes(merges_digest, pair[1])
    combined = hashlib.sha256()
    combined.update(vocab_digest.digest())
    combined.update(merges_digest.digest())
    return {
        "vocab_size": len(vocab),
        "merges_count": len(merges),
        "vocab_sha256": vocab_digest.hexdigest(),
        "merges_sha256": merges_digest.hexdigest(),
        "fingerprint_sha256": combined.hexdigest(),
    }


def _distribution_provenance(name: str) -> dict[str, Any]:
    dist = distribution(name)
    record = dist.read_text("RECORD") or ""
    return {
        "name": name,
        "version": dist.version,
        "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
        "record_byte_count": len(record.encode("utf-8")),
    }


def _environment_provenance(device: torch.device) -> dict[str, Any]:
    devices: list[dict[str, Any]] = []
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda_runtime": torch.version.cuda,
        "numpy": version("numpy"),
        "gigatoken": version("gigatoken"),
        "distributions": {
            name: _distribution_provenance(name)
            for name in ("gigatoken", "numpy", "torch")
        },
        "device": str(device),
        "devices": devices,
    }


def _config_provenance(cfg: Config) -> dict[str, Any]:
    payload = asdict(cfg)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "config": payload}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    return value


def _update_metrics_finite(metrics: dict[str, Any], *, use_muon: bool) -> bool:
    for key, value in metrics.items():
        if value is None or key in {"step", "tokens", "n_decisions", "update_applied"}:
            continue
        # Trainer deliberately reports body_grad_norm=NaN when Muon is disabled
        # and there is no body group. rest_grad_norm is then the effective gate.
        if key in {"body_grad_norm", "grad_norm"} and not use_muon:
            continue
        if isinstance(value, (int, float, np.number)) and not math.isfinite(float(value)):
            return False
    return True


def _tensor_values_finite(values: Any) -> bool:
    if isinstance(values, torch.Tensor):
        return bool(torch.isfinite(values).all())
    if isinstance(values, dict):
        return all(_tensor_values_finite(item) for item in values.values())
    if isinstance(values, (list, tuple)):
        return all(_tensor_values_finite(item) for item in values)
    if isinstance(values, Real) and not isinstance(values, bool):
        return math.isfinite(float(values))
    return True


def _model_state_finite(model: nn.Module) -> bool:
    return all(_tensor_values_finite(tensor) for tensor in model.state_dict().values())


def _gradient_state_finite(model: nn.Module) -> tuple[bool, int]:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return bool(gradients) and all(_tensor_values_finite(gradient) for gradient in gradients), len(gradients)


def _optimizer_state_finite(optimizer: Any) -> bool:
    states: list[Any] = []
    if optimizer.muon is not None:
        states.extend(optimizer.muon.state.values())
    states.extend(optimizer.adamw.state.values())
    return _tensor_values_finite(states)


def _record_step_finiteness(
    *,
    model: nn.Module,
    trainer: Trainer,
    metrics: dict[str, Any],
    use_muon: bool,
) -> dict[str, bool | int]:
    metrics_finite = _update_metrics_finite(metrics, use_muon=use_muon)
    gradients_finite, gradient_tensors = _gradient_state_finite(model)
    model_finite = _model_state_finite(model)
    optimizer_finite = _optimizer_state_finite(trainer.optimizer)
    update_applied = bool(metrics.get("update_applied", False))
    return {
        "update_applied": update_applied,
        "metrics_finite": metrics_finite,
        "gradients_finite": gradients_finite,
        "gradient_tensor_count": gradient_tensors,
        "model_state_finite": model_finite,
        "optimizer_state_finite": optimizer_finite,
        "all_finite": update_applied and metrics_finite and gradients_finite and model_finite and optimizer_finite,
    }


def _validate_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value.lower()


def _validate_sha256_file(path: Path, expected: str, name: str) -> str:
    actual = _sha256_file(path)
    if actual != _validate_sha256(expected, name):
        raise ValueError(f"{name} sha256 mismatch")
    return actual


def _validate_native_rows(rows: list[list[int]], native_vocab_size: int) -> None:
    if not rows:
        raise ValueError("tokenized split is empty")
    for index, row in enumerate(rows):
        if type(row) is not list or not row:
            raise ValueError(f"tokenized row {index} is empty or malformed")
        if any(type(value) is not int for value in row):
            raise ValueError(f"tokenized row {index} contains a non-integer id")
        if min(row) < 0 or max(row) >= native_vocab_size:
            raise ValueError(f"tokenized row {index} is outside the native vocabulary")


def build_train_vocabulary(rows: list[list[int]], *, native_vocab_size: int) -> TrainVocabulary:
    """Build a deterministic compact map from train tokens only."""
    if type(native_vocab_size) is not int or native_vocab_size < 4:
        raise ValueError("native_vocab_size must be an integer >= 4")
    _validate_native_rows(rows, native_vocab_size)
    native = np.concatenate([np.asarray(row, dtype=np.int64) for row in rows])
    observed = np.unique(native)
    new_to_old = np.concatenate(
        (
            np.full(3, -1, dtype=np.int32),
            observed.astype(np.int32, copy=False),
        )
    )
    old_to_new = np.full(native_vocab_size, -1, dtype=np.int32)
    old_to_new[new_to_old[3:]] = np.arange(3, len(new_to_old), dtype=np.int32)
    native_counts = np.bincount(native, minlength=native_vocab_size).astype(np.int64)
    compact_counts = np.zeros(len(new_to_old), dtype=np.int64)
    # LM targets for `[EOS]+content` are `content+[EOS]`: every content token
    # occurs once and EOS is the terminal target once. The prefix EOS is input,
    # not a target, so it is not counted again.
    compact_counts[1] = len(rows)
    compact_counts[3:] = native_counts[new_to_old[3:]]
    digest = hashlib.sha256()
    digest.update(old_to_new.tobytes())
    digest.update(new_to_old.tobytes())
    digest.update(compact_counts.tobytes())
    return TrainVocabulary(
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        counts=compact_counts,
        sha256=digest.hexdigest(),
    )


def build_packed_shard_payloads(
    native_rows: list[list[int]],
    *,
    split: str,
    eos_id: int = 1,
    shard_tokens: int = PINNED_SHARD_TOKENS,
) -> dict[str, bytes]:
    """Rebuild the publisher's EOS-appended, little-endian token shards."""
    if type(shard_tokens) is not int or shard_tokens < 2:
        raise ValueError("shard_tokens must be >= 2")
    payloads: dict[str, bytes] = {}
    current = bytearray()
    current_count = 0
    shard_index = 0
    for row in native_rows:
        required = len(row) + 1
        if required > shard_tokens:
            raise ValueError(f"{split} document exceeds pinned shard_tokens")
        if current_count and current_count + required > shard_tokens:
            payloads[f"{split}/shard-{shard_index:06d}.bin"] = bytes(current)
            shard_index += 1
            current.clear()
            current_count = 0
        current.extend(np.asarray([*row, eos_id], dtype="<u4").tobytes())
        current_count += required
    if current:
        payloads[f"{split}/shard-{shard_index:06d}.bin"] = bytes(current)
    return payloads


def verify_pinned_token_streams(
    root: Path,
    manifest: dict[str, Any],
    *,
    native_rows: dict[str, list[list[int]]],
    eos_id: int = 1,
) -> dict[str, dict[str, object]]:
    """Require native rows + EOS to reproduce every published token shard."""
    result: dict[str, dict[str, object]] = {}
    for split in ("train", "test"):
        expected = build_packed_shard_payloads(
            native_rows[split],
            split=split,
            eos_id=eos_id,
        )
        entries = sorted(
            (
                entry
                for entry in manifest.get("files", [])
                if entry.get("kind") == "tokens" and str(entry.get("path", "")).startswith(f"{split}/")
            ),
            key=lambda entry: str(entry["path"]),
        )
        actual_paths = [str(entry["path"]) for entry in entries]
        if actual_paths != sorted(expected):
            raise ValueError(f"pinned {split} token shard map mismatch")
        digest = hashlib.sha256()
        total = 0
        for entry, path in zip(entries, actual_paths, strict=True):
            payload = (root / path).read_bytes()
            if payload != expected[path]:
                raise ValueError(f"pinned {split} tokenization drift: {path}")
            token_count = len(payload) // 4
            if int(entry["token_count"]) != token_count:
                raise ValueError(f"pinned {split} token count mismatch: {path}")
            digest.update(payload)
            total += token_count
        source = next(
            (item for item in manifest.get("sources", []) if item.get("name") == split),
            None,
        )
        if source is not None and source.get("output_sha256") != digest.hexdigest():
            raise ValueError(f"pinned {split} concatenated token hash mismatch")
        result[split] = {
            "verified": True,
            "token_count": total,
            "sha256": digest.hexdigest(),
            "shard_count": len(expected),
        }
    return result


def encode_decision_documents(
    native_rows: list[list[int]],
    labels: list[int],
    vocab: TrainVocabulary,
    *,
    num_options: int | None = None,
) -> tuple[DecisionDocument, ...]:
    """Map train-only vocabulary rows into compact document records."""
    if len(native_rows) != len(labels):
        raise ValueError("native rows and labels must have equal length")
    _validate_native_rows(native_rows, vocab.old_to_new.shape[0])
    documents: list[DecisionDocument] = []
    for row_index, (native, label) in enumerate(zip(native_rows, labels, strict=True)):
        if type(label) is not int or (num_options is not None and not 0 <= label < num_options):
            raise ValueError(f"row {row_index} has an invalid label")
        mapped = vocab.map_ids(native)
        documents.append(DecisionDocument(tuple(int(value) for value in mapped), label))
    return tuple(documents)


def _label_count(document: DecisionDocument, num_options: int) -> None:
    if not 0 <= document.label < num_options:
        raise ValueError("document label is outside the decision option range")


def _document_batch(
    documents: list[DecisionDocument],
    *,
    vocab_size: int,
    eos_id: int,
) -> dict[str, Any]:
    width = len(documents[0].content_ids)
    if any(len(document.content_ids) != width for document in documents):
        raise ValueError("exact-length batch contains mixed document lengths")
    if width < 1:
        raise ValueError("document content must be non-empty")
    input_ids: list[list[int]] = []
    targets: list[list[int]] = []
    labels: list[int] = []
    content: list[tuple[int, ...]] = []
    for document in documents:
        if any(value < 0 or value >= vocab_size for value in document.content_ids):
            raise ValueError("compact token is outside model vocabulary")
        row = [eos_id, *document.content_ids]
        input_ids.append(row)
        targets.append([*document.content_ids, eos_id])
        labels.append(document.label)
        content.append(document.content_ids)
    return {
        "content_ids": content,
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "decision_targets": torch.tensor(labels, dtype=torch.long),
    }


def make_exact_length_batches(
    documents: tuple[DecisionDocument, ...],
    *,
    max_batch_size: int,
    seed: int,
    num_options: int | None = None,
) -> list[dict[str, Any]]:
    """Group equal-length documents without padding or truncation."""
    if type(max_batch_size) is not int or max_batch_size < 1:
        raise ValueError("max_batch_size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not documents:
        raise ValueError("cannot batch an empty split")
    by_length: dict[int, list[DecisionDocument]] = {}
    for document in documents:
        if not document.content_ids:
            raise ValueError("document content must be non-empty")
        if num_options is not None:
            _label_count(document, num_options)
        by_length.setdefault(len(document.content_ids), []).append(document)
    generator = torch.Generator().manual_seed(seed)
    batches: list[dict[str, Any]] = []
    vocab_size = max(max(document.content_ids) for document in documents) + 1
    for length in sorted(by_length):
        group = list(by_length[length])
        order = torch.randperm(len(group), generator=generator).tolist()
        group = [group[index] for index in order]
        for start in range(0, len(group), max_batch_size):
            selected = group[start : start + max_batch_size]
            batches.append(_document_batch(selected, vocab_size=vocab_size, eos_id=1))
    return batches


def repeat_epoch_batches(
    documents: tuple[DecisionDocument, ...],
    *,
    epochs: int,
    max_batch_size: int,
    base_seed: int,
    num_options: int | None = None,
) -> list[dict[str, Any]]:
    """Use every document once per epoch, changing only deterministic order."""
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    batches: list[dict[str, Any]] = []
    for epoch in range(epochs):
        batches.extend(
            make_exact_length_batches(
                documents,
                max_batch_size=max_batch_size,
                seed=base_seed + epoch * 1_000_003,
                num_options=num_options,
            )
        )
    return batches


def _accumulation_batches(
    microbatches: list[dict[str, Any]],
    grad_accum_steps: int,
) -> list[list[dict[str, Any]]]:
    if type(grad_accum_steps) is not int or grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be a positive integer")
    return [
        microbatches[start : start + grad_accum_steps]
        for start in range(0, len(microbatches), grad_accum_steps)
    ]


def smoothed_label_prior_logits(
    train_labels: torch.Tensor,
    *,
    num_options: int,
    rows: int | None = None,
) -> torch.Tensor:
    """Return log((count + 1) / (rows + options))."""
    if train_labels.ndim != 1 or train_labels.numel() == 0:
        raise ValueError("train_labels must be a non-empty vector")
    if num_options < 2:
        raise ValueError("num_options must be >= 2")
    if int(train_labels.min()) < 0 or int(train_labels.max()) >= num_options:
        raise ValueError("train label is outside option range")
    observed_rows = int(train_labels.numel())
    denominator_rows = observed_rows if rows is None else int(rows)
    if denominator_rows < observed_rows:
        raise ValueError("rows must be at least the observed label count")
    counts = torch.bincount(train_labels.to(torch.long), minlength=num_options).to(torch.float64)
    probabilities = (counts + 1.0) / float(denominator_rows + num_options)
    return probabilities.log().to(torch.float32)


def make_real_config(
    *,
    seed: int,
    max_steps: int,
    num_options: int,
    vocab_size: int,
    counts_path: Path,
    freeze_base: bool,
) -> Config:
    """Build the exact pre-registered real Banking77 model configuration."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if type(num_options) is not int or num_options < 2:
        raise ValueError("num_options must be >= 2")
    if type(vocab_size) is not int or vocab_size < 4:
        raise ValueError("vocab_size must be >= 4")
    cfg = Config()
    model = cfg.model
    model.vocab_size = vocab_size
    model.hidden_size = 128
    model.num_layers = 4
    model.loop_depth = 1
    model.init_seed = seed
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
    model.head.unigram_prior = True
    model.head.unigram_path = str(counts_path)
    model.head.unigram_smoothing = 1.0
    model.head.z_loss_weight = 0.0
    model.head.ce_chunk_rows = 64
    model.head.ce_save_logits = False
    model.head.sampled_softmax_k = 0
    model.head.sampled_proposal = "uniform"
    model.decision.enabled = True
    model.decision.num_options = num_options
    model.decision.loss_weight = 1.0
    model.decision.confidence_bins = 10

    train = cfg.train
    train.batch_size = 64
    train.grad_accum_steps = 4
    train.data.seq_len = 128
    train.data.eos_token_id = 1
    train.data.pad_token_id = 0
    train.data.cross_doc_attention = True
    train.max_steps = max_steps
    train.schedule.warmup_steps = min(20, max(0, max_steps - 1))
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
    train.ce_keep_mode = "stride"
    train.learning_rate = 0.003
    train.max_grad_norm = 1.0
    train.adam.body_lr_scale = 1.0
    train.adam.weight_decay = 0.01
    train.adapt.freeze_base = freeze_base
    train.z_loss_weight = 0.0
    train.logging.exact_ce_interval = 0
    train.tokenizer = EXPECTED_TOKENIZER
    validate_config(cfg)
    return cfg


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _evaluate_batches(
    model: HAGI,
    batches: list[dict[str, Any]],
    *,
    device: torch.device,
    confidence_bins: int,
) -> tuple[dict[str, object], float]:
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    ce_sum = 0.0
    token_count = 0
    model.eval()
    with torch.no_grad():
        for cpu_batch in batches:
            batch = _to_device(cpu_batch, device)
            output = model(
                batch["input_ids"],
                batch["targets"],
                decision_targets=batch["decision_targets"],
            )
            assert output.decision_logits is not None
            assert output.ce is not None
            logits.append(output.decision_logits.detach().float().cpu())
            labels.append(batch["decision_targets"].detach().cpu())
            rows = int(batch["targets"].numel())
            ce_sum += float(output.ce) * rows
            token_count += rows
    if token_count < 1:
        raise ValueError("evaluation produced no scored LM tokens")
    return decision_metrics(
        torch.cat(logits),
        torch.cat(labels),
        confidence_bins=confidence_bins,
    ), ce_sum / token_count


def _common_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _metrics_finite(metrics: dict[str, object]) -> bool:
    for key in ("nll", "accuracy", "macro_f1", "brier", "ece"):
        if not math.isfinite(float(metrics[key])):
            return False
    return True


def _train_lane(
    name: str,
    cfg: Config,
    common: dict[str, torch.Tensor],
    accumulation_batches: list[list[dict[str, Any]]],
    holdout_batches: list[dict[str, Any]],
    *,
    device: torch.device,
    include_lm: bool,
    step_log_dir: Path,
) -> dict[str, Any]:
    torch.manual_seed(int(cfg.model.init_seed))
    model = HAGI(cfg).to(device)
    load_common_state(model, common)
    initial, initial_lm_ce = _evaluate_batches(
        model,
        holdout_batches,
        device=device,
        confidence_bins=cfg.model.decision.confidence_bins,
    )
    assert model.decision_head is not None
    head_before = model.decision_head.weight.detach().float().cpu().clone()
    base_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("decision_head.")
    }
    common_before = _common_state_hash(base_before)
    trainer = Trainer(model, cfg)
    base_trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("decision_head.")
    }
    all_base_names = {
        name
        for name, _ in model.named_parameters()
        if not name.startswith("decision_head.")
    }
    expected_base_trainable_names = set() if cfg.train.adapt.freeze_base else set(all_base_names)
    base_ownership_matches = base_trainable_names == expected_base_trainable_names
    required_base_groups = (
        {"encoder", "blocks", "out_norm", "head"} if include_lm else {"encoder", "blocks", "out_norm"}
    ) if expected_base_trainable_names else set()
    step_times: list[float] = []
    update_flags: list[bool] = []
    step_log_path = step_log_dir / f"seed-{cfg.model.init_seed}-{name}.jsonl"
    step_log_path.parent.mkdir(parents=True, exist_ok=True)
    step_log_digest = hashlib.sha256()
    step_records = 0
    all_steps_finite = True
    finiteness_counts = {
        "update_applied": 0,
        "metrics_finite": 0,
        "gradients_finite": 0,
        "model_state_finite": 0,
        "optimizer_state_finite": 0,
        "all_finite": 0,
    }
    with step_log_path.open("wb") as step_log:
        for step_index, groups in enumerate(accumulation_batches):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            if include_lm:
                metrics = trainer.train_step([_to_device(batch, device) for batch in groups])
            else:
                metrics = trainer.train_step(
                    [
                        {
                            "input_ids": _to_device(batch, device)["input_ids"],
                            "decision_targets": _to_device(batch, device)["decision_targets"],
                        }
                        for batch in groups
                    ]
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append((time.perf_counter() - started) * 1000.0)
            update_flags.append(bool(metrics.get("update_applied", False)))
            step_finiteness = _record_step_finiteness(
                model=model,
                trainer=trainer,
                metrics=metrics,
                use_muon=bool(cfg.train.use_muon),
            )
            for key in finiteness_counts:
                finiteness_counts[key] += int(bool(step_finiteness[key]))
            all_steps_finite = all_steps_finite and bool(step_finiteness["all_finite"])
            record = {
                "schema_version": 1,
                "seed": int(cfg.model.init_seed),
                "lane": name,
                "step_index": step_index,
                "trainer_step": int(metrics["step"]),
                "include_lm": include_lm,
                "metrics": _json_safe(metrics),
                "finiteness": step_finiteness,
            }
            payload = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            step_log.write(payload)
            step_log_digest.update(payload)
            step_records += 1
        step_log.flush()
        os.fsync(step_log.fileno())
    step_log_summary = {
        "path": str(step_log_path.resolve()),
        "sha256": step_log_digest.hexdigest(),
        "record_count": step_records,
        "all_steps_finite": all_steps_finite,
        "counts": finiteness_counts,
    }
    head_update = float(
        (model.decision_head.weight.detach().float().cpu() - head_before).abs().max()
    )
    base_after = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("decision_head.")
    }
    common_after = _common_state_hash(base_after)
    base_changed = 0
    base_changed_max = 0.0
    base_changed_names: list[str] = []
    for parameter_name in sorted(base_trainable_names):
        old = base_before[parameter_name]
        new = base_after[parameter_name]
        if not torch.equal(old, new):
            base_changed += 1
            base_changed_names.append(parameter_name)
            base_changed_max = max(
                base_changed_max,
                float((new.float() - old.float()).abs().max()),
            )
    base_changed_groups = {
        group: sum(name == group or name.startswith(f"{group}.") for name in base_changed_names)
        for group in sorted(required_base_groups)
    }
    required_group_updates_observed = bool(
        base_ownership_matches
        and (
            not expected_base_trainable_names
            or (
                bool(base_trainable_names)
                and all(count > 0 for count in base_changed_groups.values())
            )
        )
    )
    final, final_lm_ce = _evaluate_batches(
        model,
        holdout_batches,
        device=device,
        confidence_bins=cfg.model.decision.confidence_bins,
    )
    finite = (
        _metrics_finite(initial)
        and _metrics_finite(final)
        and math.isfinite(initial_lm_ce)
        and math.isfinite(final_lm_ce)
        and all_steps_finite
    )
    configuration = _config_provenance(cfg)
    return {
        "name": name,
        "configuration": configuration,
        "initial": initial,
        "final": final,
        "initial_lm_ce": initial_lm_ce,
        "final_lm_ce": final_lm_ce,
        "steps_completed": len(accumulation_batches),
        "updates_applied": sum(update_flags),
        "rejected_updates": len(update_flags) - sum(update_flags),
        "mean_step_ms": statistics.fmean(step_times) if step_times else 0.0,
        "step_finiteness": step_log_summary,
        "decision_head_max_abs_update": head_update,
        "common_state_sha256_before": common_before,
        "common_state_sha256_after": common_after,
        "common_state_unchanged": common_before == common_after,
        "base_trainable_parameter_count": len(base_trainable_names),
        "base_expected_trainable_parameter_count": len(expected_base_trainable_names),
        "base_ownership_matches": base_ownership_matches,
        "base_changed_parameter_count": base_changed,
        "base_changed_max_abs_update": base_changed_max,
        "base_trainable_update_observed": required_group_updates_observed,
        "base_required_groups": sorted(required_base_groups),
        "base_changed_groups": base_changed_groups,
        "parameters": model.param_summary()["total"],
        "finite": finite,
    }


def _seed_quality_checks(report_lanes: dict[str, dict[str, Any]]) -> dict[str, bool]:
    end = report_lanes["end_to_end"]["final"]
    majority = report_lanes["majority_or_uniform"]["initial"]
    frozen = report_lanes["frozen_probe"]["final"]
    assert end is not None and frozen is not None
    accuracy_reference = max(float(majority["accuracy"]), float(frozen["accuracy"]))
    return {
        "execution_complete": all(bool(lane.get("finite", False)) for lane in report_lanes.values()),
        "beats_majority_nll": float(end["nll"]) < float(majority["nll"]),
        "beats_frozen_probe_nll": float(end["nll"]) < float(frozen["nll"]),
        "no_accuracy_regression": float(end["accuracy"]) + MAX_ACCURACY_REGRESSION >= accuracy_reference,
        "ece_within_limit": float(end["ece"]) <= MAX_ECE,
    }


def run_seed(
    data: DecisionData,
    *,
    seed: int,
    counts_path: Path,
    epochs: int,
    max_batch_size: int,
    grad_accum_steps: int,
    device: str,
) -> dict[str, Any]:
    """Run the four matched lanes for one Banking77 seed."""
    if not data.train or not data.test:
        raise ValueError("real DecisionPlane data must contain train and test rows")
    if data.num_options != EXPECTED_CATEGORIES and data.quality_eligible:
        raise ValueError("quality-eligible Banking77 data requires 77 categories")
    if data.quality_eligible and not data.pinned_token_streams_verified:
        raise ValueError("quality-eligible Banking77 data requires verified pinned token streams")
    device_obj = torch.device(device)
    train_batches = repeat_epoch_batches(
        data.train,
        epochs=epochs,
        max_batch_size=max_batch_size,
        base_seed=seed,
        num_options=data.num_options,
    )
    holdout_batches = make_exact_length_batches(
        data.test,
        max_batch_size=max_batch_size,
        seed=seed + 2_000_003,
        num_options=data.num_options,
    )
    accumulation_batches = _accumulation_batches(train_batches, grad_accum_steps)
    max_steps = len(accumulation_batches)
    base_cfg = make_real_config(
        seed=seed,
        max_steps=max_steps,
        num_options=data.num_options,
        vocab_size=3 + data.train_native_unique,
        counts_path=counts_path,
        freeze_base=False,
    )
    # The compact vocabulary contains the 77 labels as decision options, not as
    # token IDs. The model vocabulary is exactly the train-vocabulary size.
    torch.manual_seed(seed)
    base = HAGI(base_cfg)
    common = _common_state(base)
    del base

    train_labels = torch.tensor([document.label for document in data.train], dtype=torch.long)
    prior_logits = smoothed_label_prior_logits(
        train_labels,
        num_options=data.num_options,
        rows=len(data.train),
    )
    holdout_logits = torch.stack(
        [prior_logits.clone() for _ in data.test]
    )
    holdout_targets = torch.tensor([document.label for document in data.test], dtype=torch.long)
    majority_metrics = decision_metrics(
        holdout_logits,
        holdout_targets,
        confidence_bins=base_cfg.model.decision.confidence_bins,
    )
    majority = {
        "name": "majority_or_uniform",
        "initial": majority_metrics,
        "final": None,
        "initial_lm_ce": None,
        "final_lm_ce": None,
        "steps_completed": 0,
        "updates_applied": 0,
        "rejected_updates": 0,
        "mean_step_ms": 0.0,
        "step_finiteness": {
            "path": None,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "record_count": 0,
            "all_steps_finite": True,
            "counts": {
                "update_applied": 0,
                "metrics_finite": 0,
                "gradients_finite": 0,
                "model_state_finite": 0,
                "optimizer_state_finite": 0,
                "all_finite": 0,
            },
        },
        "decision_head_max_abs_update": 0.0,
        "common_state_sha256_before": None,
        "common_state_sha256_after": None,
        "common_state_unchanged": True,
        "base_trainable_parameter_count": 0,
        "base_expected_trainable_parameter_count": 0,
        "base_ownership_matches": True,
        "base_changed_parameter_count": 0,
        "base_changed_max_abs_update": 0.0,
        "base_trainable_update_observed": False,
        "base_required_groups": [],
        "base_changed_groups": {},
        "parameters": 0,
        "finite": _metrics_finite(majority_metrics),
    }
    lanes: list[dict[str, Any]] = [majority]
    for lane_name, freeze_base, include_lm in (
        ("frozen_probe", True, False),
        ("decision_only", False, False),
        ("end_to_end", False, True),
    ):
        cfg = make_real_config(
            seed=seed,
            max_steps=max_steps,
            num_options=data.num_options,
            vocab_size=base_cfg.model.vocab_size,
            counts_path=counts_path,
            freeze_base=freeze_base,
        )
        lanes.append(
            _train_lane(
                lane_name,
                cfg,
                common,
                accumulation_batches,
                holdout_batches,
                device=device_obj,
                include_lm=include_lm,
                step_log_dir=Path(counts_path).parent / "step_logs",
            )
        )
    by_name = {str(lane["name"]): lane for lane in lanes}
    initial_metrics_agree = all(
        by_name[name]["initial"] == by_name["frozen_probe"]["initial"]
        for name in ("frozen_probe", "decision_only", "end_to_end")
    )
    initial_lm_agree = all(
        math.isclose(
            float(by_name[name]["initial_lm_ce"]),
            float(by_name["frozen_probe"]["initial_lm_ce"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for name in ("frozen_probe", "decision_only", "end_to_end")
    )
    updates_ok = all(
        int(lane["steps_completed"]) == max_steps
        and int(lane["updates_applied"]) == max_steps
        and int(lane["rejected_updates"]) == 0
        and float(lane["decision_head_max_abs_update"]) > 0.0
        for lane in lanes[1:]
    )
    runtime_checks = {
        "finite_metrics": all(bool(lane["finite"]) for lane in lanes),
        "all_steps_and_updates_finite": all(
            bool(lane["step_finiteness"]["all_steps_finite"]) for lane in lanes
        ),
        "step_ledgers_hash_bound": all(
            lane["name"] == "majority_or_uniform"
            or (
                isinstance(lane["step_finiteness"]["path"], str)
                and _validate_sha256_file(
                    Path(lane["step_finiteness"]["path"]),
                    str(lane["step_finiteness"]["sha256"]),
                    f"{lane['name']} step ledger",
                )
                == str(lane["step_finiteness"]["sha256"])
            )
            for lane in lanes
        ),
        "all_updates_applied": updates_ok,
        "common_initial_decision_metrics_exact": initial_metrics_agree,
        "common_initial_lm_ce_exact": initial_lm_agree,
        "frozen_base_unchanged": bool(by_name["frozen_probe"]["common_state_unchanged"]),
        "base_ownership_verified": all(
            bool(by_name[lane_name]["base_ownership_matches"])
            for lane_name in ("frozen_probe", "decision_only", "end_to_end")
        ),
        "body_trainable_updates_observed": all(
            bool(by_name[name]["base_trainable_update_observed"])
            for name in ("decision_only", "end_to_end")
        ),
    }
    mechanism_supported = all(runtime_checks.values())
    quality_checks = _seed_quality_checks(by_name)
    preregistered_budget = bool(
        data.quality_eligible
        and epochs == 3
        and max_batch_size == 64
        and grad_accum_steps == 4
    )
    return {
        "schema_version": 2,
        "seed": seed,
        "device": str(device_obj),
        "epochs": epochs,
        "max_batch_size": max_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "microbatches": len(train_batches),
        "optimizer_steps": max_steps,
        "train_rows": len(data.train),
        "holdout_rows": len(data.test),
        "num_options": data.num_options,
        "train_native_unique": data.train_native_unique,
        "test_token_count": data.test_token_count,
        "test_oov_token_count": data.test_oov_token_count,
        "manifest_sha256": data.manifest_sha256,
        "train_text_sha256": data.train_text_sha256,
        "test_text_sha256": data.test_text_sha256,
        "tokenizer_name": EXPECTED_TOKENIZER,
        "tokenizer_version": data.tokenizer_version,
        "tokenizer_fingerprint": data.tokenizer_fingerprint,
        "artifact_validated": bool(data.quality_eligible),
        "pinned_token_streams_verified": bool(data.pinned_token_streams_verified),
        "pinned_token_stream_metadata": {
            split: dict(metadata) for split, metadata in data.pinned_token_stream_metadata
        },
        "preregistered_budget": preregistered_budget,
        "mechanism_supported": mechanism_supported,
        "quality_supported": False,
        "quality_checks": quality_checks,
        "runtime_checks": runtime_checks,
        "lanes": lanes,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("label JSONL record is not an object")
                records.append(value)
    return records


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_artifact_metadata(manifest: dict[str, Any], root: Path) -> None:
    expected = {
        "artifact_id": EXPECTED_ARTIFACT_ID,
        "revision": EXPECTED_REVISION,
        "dataset": EXPECTED_DATASET,
        "license": EXPECTED_LICENSE,
        "tokenizer_name": EXPECTED_TOKENIZER,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "eos_token_id": 1,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "test_rows": EXPECTED_TEST_ROWS,
        "category_count": EXPECTED_CATEGORIES,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key} mismatch: {manifest.get(key)!r} != {value!r}")
    _validate_sha256_file(
        root / "text/train.txt",
        EXPECTED_TRAIN_TEXT_SHA256,
        "train text",
    )
    _validate_sha256_file(
        root / "text/test.txt",
        EXPECTED_TEST_TEXT_SHA256,
        "test text",
    )


def load_banking77_data(
    artifact_dir: str | Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[DecisionData, TrainVocabulary]:
    """Validate the immutable artifact and tokenize its exact official rows."""
    root = Path(artifact_dir)
    tokenizer_version = version("gigatoken")
    if tokenizer_version != EXPECTED_GIGATOKEN_VERSION:
        raise ValueError(
            f"this pinned runner requires gigatoken {EXPECTED_GIGATOKEN_VERSION}, got {tokenizer_version}"
        )
    manifest = load_published_artifact(root)
    actual_manifest = _validate_sha256_file(
        root / "manifest.json",
        expected_manifest_sha256,
        "manifest",
    )
    _validate_artifact_metadata(manifest, root)
    categories = json.loads((root / "metadata/categories.json").read_text(encoding="utf-8"))
    if not isinstance(categories, list) or len(categories) != EXPECTED_CATEGORIES:
        raise ValueError("Banking77 categories are not the pinned 77-category list")
    label_by_intent = {intent: index for index, intent in enumerate(categories)}
    native_train: list[list[int]] = []
    native_test: list[list[int]] = []
    train_labels: list[int] = []
    test_labels: list[int] = []
    for split, native_target, labels_target in (
        ("train", native_train, train_labels),
        ("test", native_test, test_labels),
    ):
        rows = _read_csv(root / "provenance" / f"{split}.csv")
        label_records = _read_jsonl(root / "labels" / f"{split}.jsonl")
        if len(rows) != len(label_records):
            raise ValueError(f"{split} CSV/label row count mismatch")
        texts: list[str] = []
        for index, (row, label_record) in enumerate(zip(rows, label_records, strict=True), start=2):
            if set(row) != {"text", "category"}:
                raise ValueError(f"{split} CSV has unexpected columns")
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{split} row {index} has empty text")
            if label_record.get("source_row") != index:
                raise ValueError(f"{split} label source row mismatch at {index}")
            if label_record.get("intent") != row["category"]:
                raise ValueError(f"{split} label intent mismatch at {index}")
            if label_record.get("text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
                raise ValueError(f"{split} label text hash mismatch at {index}")
            labels_target.append(label_by_intent[row["category"]])
            texts.append(text)
        import gigatoken

        tokenizer = gigatoken.Tokenizer(EXPECTED_TOKENIZER)
        tokenizer_fingerprint = _tokenizer_fingerprint(tokenizer)
        if int(tokenizer.vocab_size) != EXPECTED_VOCAB_SIZE:
            raise ValueError("installed Gigatoken vocabulary size mismatch")
        encoded: list[list[int]] = []
        for start in range(0, len(texts), 256):
            batch = texts[start : start + 256]
            rows_encoded = tokenizer.encode_batch_list(batch)
            if len(rows_encoded) != len(batch):
                raise ValueError("tokenizer returned a different number of rows")
            for original, ids in zip(batch, rows_encoded, strict=True):
                native = [int(value) for value in ids]
                decoded = tokenizer.decode(native)
                if isinstance(decoded, bytes):
                    decoded = decoded.decode("utf-8", errors="strict")
                if decoded != original:
                    raise ValueError("native tokenizer decode is not exact")
                encoded.append(native)
        native_target.extend(encoded)
    train_vocab = build_train_vocabulary(native_train, native_vocab_size=EXPECTED_VOCAB_SIZE)
    token_stream_metadata = verify_pinned_token_streams(
        root,
        manifest,
        native_rows={"train": native_train, "test": native_test},
        eos_id=int(manifest["eos_token_id"]),
    )
    train_native_set = set(int(value) for value in train_vocab.new_to_old[3:])
    test_tokens = sum(len(row) for row in native_test)
    test_oov = sum(
        int(token) not in train_native_set
        for row in native_test
        for token in row
    )
    data = DecisionData(
        train=encode_decision_documents(
            native_train,
            train_labels,
            train_vocab,
            num_options=EXPECTED_CATEGORIES,
        ),
        test=encode_decision_documents(
            native_test,
            test_labels,
            train_vocab,
            num_options=EXPECTED_CATEGORIES,
        ),
        num_options=EXPECTED_CATEGORIES,
        native_vocab_size=EXPECTED_VOCAB_SIZE,
        train_native_unique=len(train_vocab.new_to_old) - 3,
        test_token_count=test_tokens,
        test_oov_token_count=test_oov,
        manifest_sha256=actual_manifest,
        train_text_sha256=EXPECTED_TRAIN_TEXT_SHA256,
        test_text_sha256=EXPECTED_TEST_TEXT_SHA256,
        tokenizer_version=tokenizer_version,
        tokenizer_fingerprint=tokenizer_fingerprint,
        quality_eligible=True,
        pinned_token_streams_verified=all(
            bool(item.get("verified")) for item in token_stream_metadata.values()
        ),
        pinned_token_stream_metadata=tuple(sorted(token_stream_metadata.items())),
    )
    return data, train_vocab


def _write_vocabulary(vocab: TrainVocabulary, run_dir: Path) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "old_to_new": vocab.old_to_new,
        "new_to_old": vocab.new_to_old,
        "counts": vocab.counts,
    }
    result: dict[str, str] = {}
    for name, value in artifacts.items():
        path = run_dir / f"{name}.npy"
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        result[name] = _sha256_file(path)
    return result


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Run the pre-registered multi-seed gate or a fail-closed smoke."""
    data, vocab = load_banking77_data(
        args.artifact,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    run_dir = (Path(args.run_dir) if args.run_dir is not None else Path(args.output).parent).resolve()
    vocab_files = _write_vocabulary(vocab, run_dir)
    source_provenance = _source_provenance(run_dir / "source_snapshot")
    environment = _environment_provenance(torch.device(args.device))
    counts_path = (run_dir / "counts.npy").resolve()
    seeds = [SEEDS[offset] for offset in range(args.num_seeds)]
    seed_reports = [
        run_seed(
            data,
            seed=seed,
            counts_path=counts_path,
            epochs=args.epochs,
            max_batch_size=args.max_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            device=args.device,
        )
        for seed in seeds
    ]
    source_provenance_after = _source_provenance()
    source_provenance["tree_sha256_after_run"] = source_provenance_after["tree_sha256"]
    source_provenance["files_match_after_run"] = (
        source_provenance_after["files"] == source_provenance["files"]
    )
    source_provenance["stable_during_run"] = bool(
        source_provenance_after["tree_sha256"] == source_provenance["tree_sha256"]
        and source_provenance["files_match_after_run"]
    )
    required_wins = 2
    majority_wins = frozen_wins = accuracy_ok = ece_ok = 0
    for report in seed_reports:
        lanes = {str(lane["name"]): lane for lane in report["lanes"]}
        end = lanes["end_to_end"]["final"]
        majority = lanes["majority_or_uniform"]["initial"]
        frozen = lanes["frozen_probe"]["final"]
        assert end is not None and frozen is not None
        majority_wins += int(float(end["nll"]) < float(majority["nll"]))
        frozen_wins += int(float(end["nll"]) < float(frozen["nll"]))
        accuracy_ok += int(bool(report["quality_checks"]["no_accuracy_regression"]))
        ece_ok += int(bool(report["quality_checks"]["ece_within_limit"]))
    execution_completed = all(bool(report["mechanism_supported"]) for report in seed_reports)
    overall_checks = {
        "at_least_three_seeds": args.num_seeds == 3,
        "exact_fixed_seed_set": (
            list(seeds) == list(SEEDS)
            and args.seed == SEEDS[0]
            and [int(report["seed"]) for report in seed_reports] == list(SEEDS)
        ),
        "all_mechanisms_completed": execution_completed,
        "strict_majority_wins_vs_majority": majority_wins >= required_wins,
        "strict_majority_wins_vs_frozen_probe": frozen_wins >= required_wins,
        "all_accuracy_checks_passed": accuracy_ok == args.num_seeds,
        "all_ece_checks_passed": ece_ok == args.num_seeds,
        "exact_preregistered_budget": all(bool(report["preregistered_budget"]) for report in seed_reports),
        "source_provenance_stable_during_run": bool(source_provenance["stable_during_run"]),
    }
    observed_gate_passed = all(overall_checks.values())
    return {
        "experiment": "decision_plane_banking77_real_matched_multi_seed",
        "schema_version": 2,
        "artifact": str(Path(args.artifact).resolve()),
        "manifest_sha256": data.manifest_sha256,
        "train_text_sha256": data.train_text_sha256,
        "test_text_sha256": data.test_text_sha256,
        "tokenizer_name": EXPECTED_TOKENIZER,
        "tokenizer_version": data.tokenizer_version,
        "tokenizer_fingerprint": data.tokenizer_fingerprint,
        "train_rows": data.train.__len__(),
        "test_rows": data.test.__len__(),
        "num_options": data.num_options,
        "train_native_unique": data.train_native_unique,
        "test_token_count": data.test_token_count,
        "test_oov_token_count": data.test_oov_token_count,
        "test_oov_rate": data.test_oov_token_count / max(data.test_token_count, 1),
        "vocab_sha256": vocab.sha256,
        "vocab_files": vocab_files,
        "source_provenance": source_provenance,
        "environment": environment,
        "config_fingerprints": {
            str(seed_report["seed"]): {
                str(lane["name"]): lane["configuration"]
                for lane in seed_report["lanes"]
                if lane["name"] in {"frozen_probe", "decision_only", "end_to_end"}
            }
            for seed_report in seed_reports
        },
        "base_seed": args.seed,
        "seed_ids": list(seeds),
        "num_seeds": args.num_seeds,
        "epochs": args.epochs,
        "max_batch_size": args.max_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "device": args.device,
        "execution_completed": execution_completed,
        "mechanism_supported": execution_completed,
        "quality_supported": observed_gate_passed,
        "promotion_status": "promoted-real-slice" if observed_gate_passed else "research-only",
        "observed_quality_gate_passed": observed_gate_passed,
        "pre_registered_gates": {
            "minimum_seeds": 3,
            "required_strict_majority_wins": 2,
            "max_ece": MAX_ECE,
            "max_accuracy_regression": MAX_ACCURACY_REGRESSION,
            "exact_preregistered_budget": True,
        },
        "quality_evidence": {
            "wins_vs_majority": majority_wins,
            "wins_vs_frozen_probe": frozen_wins,
            "accuracy_checks_passed": accuracy_ok,
            "ece_checks_passed": ece_ok,
        },
        "overall_checks": overall_checks,
        "seed_reports": seed_reports,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("reports/decision_banking77"))
    parser.add_argument("--output", type=Path, default=Path("reports/decision_plane_banking77.json"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if _validate_sha256(args.expected_manifest_sha256, "expected manifest hash") != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"this pinned runner requires manifest {EXPECTED_MANIFEST_SHA256}")
    if args.num_seeds < 1 or args.num_seeds > len(SEEDS):
        raise ValueError(f"num-seeds must be in [1, {len(SEEDS)}]")
    if args.num_seeds == len(SEEDS) and args.seed != SEEDS[0]:
        raise ValueError(f"three-seed quality protocol requires --seed {SEEDS[0]}")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.max_batch_size < 1 or args.grad_accum_steps < 1:
        raise ValueError("batch settings must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_experiment(args)
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["execution_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
