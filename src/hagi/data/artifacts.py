"""Reproducible, fail-closed artifacts for packed training data.

The production loader deliberately keeps its historical ``*.bin`` contract.
This module is the opt-in publication boundary: a directory becomes a
usable data artifact only when its manifest and every listed file validate.
No network or model imports belong here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from tempfile import mkdtemp
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
UINT32_MAX = (1 << 32) - 1
_HEX_DIGITS = frozenset("0123456789abcdef")
_FILE_KINDS = frozenset({"tokens", "text", "jsonl"})
_REQUIRED_SOURCE_FIELDS = frozenset(
    {
        "name",
        "ratio",
        "dataset",
        "revision",
        "license",
        "source_url",
        "retrieval_timestamp",
        "tokenizer_version",
        "filter_policy_version",
        "dedup_policy_version",
        "byte_count",
        "token_count",
        "input_sha256",
        "output_sha256",
    }
)


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of *value*."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file in bounded blocks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value.lower()) <= _HEX_DIGITS


def _is_link(path: Path) -> bool:
    """Return whether a path is a symlink or Windows junction/reparse point."""
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _reject_link(path: Path, label: str) -> None:
    if _is_link(path):
        raise ValueError(f"{label} must not be a symlink or junction: {path}")


def _reject_link_chain(path: Path, label: str) -> None:
    """Reject symlink/junction components from an existing path to its root."""
    current = Path(os.path.abspath(path))
    while True:
        if current.exists() or _is_link(current):
            _reject_link(current, label)
        parent = current.parent
        if parent == current:
            return
        current = parent


def _artifact_root(root: str | Path) -> Path:
    raw = Path(root)
    _reject_link_chain(raw, "artifact root")
    if not raw.is_dir():
        raise ValueError(f"artifact root is not a directory: {raw}")
    resolved = raw.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"artifact root is not a directory: {raw}")
    return resolved


def _safe_artifact_file(root: str | Path, relative_path: str) -> Path:
    """Resolve one manifest path while rejecting links and root escape."""
    base = _artifact_root(root)
    current = base
    for part in PurePosixPath(validate_relative_path(relative_path)).parts:
        current = current / part
        if current.exists() or _is_link(current):
            _reject_link(current, "artifact path")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"artifact file cannot be resolved: {relative_path}") from exc
    if not resolved.is_relative_to(base):
        raise ValueError(f"artifact path escapes root: {relative_path}")
    if not resolved.is_file():
        raise ValueError(f"manifest file is not a regular file: {relative_path}")
    return resolved


def validate_relative_path(value: str) -> str:
    """Validate a portable relative artifact path."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("path must be a non-empty relative string")
    if "\\" in value:
        raise ValueError("path must use '/' separators")
    components = value.split("/")
    if any(part in {"", ".", ".."} for part in components):
        raise ValueError("path traversal or empty component is not allowed")
    if PurePosixPath(value).is_absolute():
        raise ValueError("path must be relative")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def normalize_mix(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize source ratios in stable name order."""
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    names: set[str] = set()
    total = 0.0
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each source must be a mapping")
        name = source.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("source names must be non-empty and unique")
        names.add(name)
        total += _positive_number(source.get("ratio"), f"ratio for {name!r}")

    result: list[dict[str, Any]] = []
    for source in sorted(sources, key=lambda item: item["name"]):
        item = dict(source)
        item["ratio"] = float(source["ratio"]) / total
        result.append(item)
    return result


def _validate_count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_file_entry(entry: object) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("file entries must be mappings")
    path = validate_relative_path(entry.get("path"))
    digest = entry.get("sha256")
    if not _is_sha256(digest):
        raise ValueError(f"file {path!r} has an invalid sha256")
    kind = entry.get("kind", "tokens")
    if kind not in _FILE_KINDS:
        raise ValueError(f"file {path!r} has unsupported kind {kind!r}")
    byte_count = _validate_count(entry.get("byte_count"), f"file {path!r}.byte_count")
    token_count = _validate_count(entry.get("token_count"), f"file {path!r}.token_count")
    return {
        "path": path,
        "kind": kind,
        "sha256": digest.lower(),
        "byte_count": byte_count,
        "token_count": token_count,
    }


def validate_manifest(manifest: dict[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    """Validate and normalize a dataset manifest.

    With ``root=None`` this validates schema and source metadata. With a
    root, every listed file must also match its size/hash, and token files
    are checked against the declared vocabulary size.
    """
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a mapping")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema version")
    if manifest.get("artifact_type", "dataset") != "dataset":
        raise ValueError("manifest artifact_type must be 'dataset'")

    normalized_sources = normalize_mix(manifest.get("sources"))
    for source in normalized_sources:
        missing = _REQUIRED_SOURCE_FIELDS.difference(source)
        if missing:
            raise ValueError(f"source is missing fields: {sorted(missing)}")
        for field in ("byte_count", "token_count"):
            _validate_count(source[field], f"source {source['name']!r}.{field}")
        for field in ("input_sha256", "output_sha256"):
            if not _is_sha256(source[field]):
                raise ValueError(f"source {source['name']!r}.{field} is not a SHA-256 digest")
        for field in (
            "dataset",
            "revision",
            "license",
            "source_url",
            "retrieval_timestamp",
            "tokenizer_version",
            "filter_policy_version",
            "dedup_policy_version",
        ):
            if not isinstance(source[field], str) or not source[field]:
                raise ValueError(f"source {source['name']!r}.{field} must be a non-empty string")

    files = manifest.get("files", [])
    if not isinstance(files, list):
        raise ValueError("files must be a list")
    normalized_files = [_validate_file_entry(entry) for entry in files]
    paths = [entry["path"] for entry in normalized_files]
    if len(paths) != len(set(paths)):
        raise ValueError("file paths must be unique")

    vocab_size = manifest.get("vocab_size")
    if vocab_size is not None and (type(vocab_size) is not int or vocab_size < 1):
        raise ValueError("manifest.vocab_size must be a positive integer")

    result = dict(manifest)
    result["sources"] = normalized_sources
    result["files"] = normalized_files
    if root is not None:
        if not normalized_files:
            raise ValueError("published artifact manifest must list files")
        base = _artifact_root(root)
        for entry in normalized_files:
            path = _safe_artifact_file(base, str(entry["path"]))
            if path.stat().st_size != entry["byte_count"]:
                raise ValueError(f"manifest byte count mismatch: {entry['path']}")
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"manifest hash mismatch: {entry['path']}")
            if entry["kind"] == "tokens":
                actual_token_count = validate_uint32_stream_file(path, vocab_size)
                if actual_token_count != entry["token_count"]:
                    raise ValueError(f"manifest token count mismatch: {entry['path']}")
            elif entry["kind"] == "jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip() and not isinstance(json.loads(line), dict):
                            raise ValueError(f"invalid JSONL quarantine record: {entry['path']}")
            else:
                path.read_text(encoding="utf-8")
    if "total_token_count" in result:
        total = sum(entry["token_count"] for entry in normalized_files if entry["kind"] == "tokens")
        if _validate_count(result["total_token_count"], "total_token_count") != total:
            raise ValueError("manifest total_token_count mismatch")
    if "total_byte_count" in result:
        total = sum(entry["byte_count"] for entry in normalized_files)
        if _validate_count(result["total_byte_count"], "total_byte_count") != total:
            raise ValueError("manifest total_byte_count mismatch")
    return result


def validate_uint32_stream_bytes(data: bytes, vocab_size: int | None = None) -> list[int]:
    """Decode a little-endian uint32 stream and reject truncated/out-of-range data."""
    if len(data) % 4:
        raise ValueError("truncated uint32 stream")
    if vocab_size is not None and (type(vocab_size) is not int or vocab_size < 1):
        raise ValueError("vocab_size must be a positive integer")
    values: list[int] = []
    for offset in range(0, len(data), 4):
        value = int.from_bytes(data[offset : offset + 4], "little", signed=False)
        if vocab_size is not None and value >= vocab_size:
            raise ValueError(f"token id {value} outside vocabulary [0, {vocab_size})")
        values.append(value)
    return values


def validate_uint32_stream_file(path: str | Path, vocab_size: int | None = None) -> int:
    """Validate a packed token file in bounded blocks and return token count."""
    if vocab_size is not None and (type(vocab_size) is not int or vocab_size < 1):
        raise ValueError("vocab_size must be a positive integer")
    carry = b""
    count = 0
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            data = carry + block
            usable = len(data) - (len(data) % 4)
            for offset in range(0, usable, 4):
                value = int.from_bytes(data[offset : offset + 4], "little", signed=False)
                if vocab_size is not None and value >= vocab_size:
                    raise ValueError(f"token id {value} outside vocabulary [0, {vocab_size})")
                count += 1
            carry = data[usable:]
    if carry:
        raise ValueError("truncated uint32 stream")
    return count


def deduplicate_normalized_lines(
    lines: Iterable[str],
    *,
    source_name: str | None = None,
) -> tuple[list[str], list[dict[str, object]]]:
    """Keep first-seen non-empty lines and quarantine duplicates/empty records."""
    accepted: list[str] = []
    quarantine: list[dict[str, object]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(lines, start=1):
        normalized = raw.strip()
        record: dict[str, object] = {
            "line_number": line_number,
            "reason": "",
            "content_sha256": "",
        }
        if source_name is not None:
            record["source_name"] = source_name
        if not normalized:
            record["reason"] = "empty"
            record["content_sha256"] = sha256_bytes(b"")
            quarantine.append(record)
            continue
        digest = sha256_bytes(normalized.encode("utf-8"))
        record["content_sha256"] = digest
        if digest in seen:
            record["reason"] = "duplicate"
            quarantine.append(record)
            continue
        seen.add(digest)
        accepted.append(normalized)
    return accepted, quarantine


def read_utf8_records(path: str | Path) -> tuple[list[str], list[dict[str, object]]]:
    """Read strict UTF-8 newline records with stable source line numbers."""
    raw = Path(path).read_bytes()
    accepted: list[str] = []
    quarantine: list[dict[str, object]] = []
    seen: set[str] = set()
    for line_number, payload in enumerate(raw.splitlines(), start=1):
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError:
            quarantine.append(
                {
                    "line_number": line_number,
                    "reason": "malformed_utf8",
                    "content_sha256": sha256_bytes(payload),
                }
            )
            continue
        normalized = decoded.strip()
        digest = sha256_bytes(normalized.encode("utf-8")) if normalized else sha256_bytes(b"")
        if not normalized:
            reason = "empty"
        elif digest in seen:
            reason = "duplicate"
        else:
            seen.add(digest)
            accepted.append(normalized)
            continue
        quarantine.append(
            {
                "line_number": line_number,
                "reason": reason,
                "content_sha256": digest,
            }
        )
    return accepted, quarantine


def quarantine_jsonl(records: Iterable[dict[str, object]]) -> bytes:
    """Serialize quarantine records deterministically as UTF-8 JSONL."""
    lines = [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON to a sibling temporary file and replace the target atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def file_entry(path: str, payload: bytes, *, kind: str = "tokens", token_count: int | None = None) -> dict[str, Any]:
    """Build a manifest file entry for an in-memory artifact payload."""
    if kind not in _FILE_KINDS:
        raise ValueError(f"unsupported file kind {kind!r}")
    if kind == "tokens":
        if len(payload) % 4:
            raise ValueError("truncated uint32 token payload")
        computed_count = len(payload) // 4
        if token_count is not None and token_count != computed_count:
            raise ValueError("token_count does not match uint32 payload")
        count = computed_count
    else:
        # Non-token files (quarantine JSONL, text manifests) carry byte counts
        # separately; their token_count is zero unless a caller explicitly
        # supplies a count for a custom artifact kind.
        count = 0 if token_count is None else token_count
    _validate_count(count, "file token_count")
    return {
        "path": validate_relative_path(path),
        "kind": kind,
        "sha256": sha256_bytes(payload),
        "byte_count": len(payload),
        "token_count": count,
    }


def split_packed_tokens(
    source_dir: str | Path,
    destination: str | Path,
    *,
    holdout_tokens: int,
) -> Path:
    """Publish a deterministic disjoint train/holdout split of packed tokens.

    The split is a contiguous suffix of the sorted ``*.bin`` streams, so it
    can be reproduced from the manifest and cannot silently overlap train.
    """
    if type(holdout_tokens) is not int or holdout_tokens < 1:
        raise ValueError("holdout_tokens must be a positive integer")
    base = _artifact_root(source_dir)
    sources = sorted(path for path in base.glob("*.bin") if path.is_file())
    if not sources:
        raise ValueError("source directory contains no packed .bin files")
    decoded: list[tuple[str, list[int], bytes]] = []
    for path in sources:
        _reject_link(path, "packed input file")
        payload = path.read_bytes()
        values = validate_uint32_stream_bytes(payload)
        decoded.append((path.name, values, payload))
    total = sum(len(values) for _, values, _ in decoded)
    if holdout_tokens >= total:
        raise ValueError("holdout_tokens must be smaller than the total token count")
    train_count = total - holdout_tokens
    train: list[int] = []
    holdout: list[int] = []
    for _, values, _ in decoded:
        for value in values:
            (holdout if len(train) >= train_count else train).append(value)
    train_bytes = b"".join(value.to_bytes(4, "little") for value in train)
    holdout_bytes = b"".join(value.to_bytes(4, "little") for value in holdout)
    files = {"train/data.bin": train_bytes, "holdout/data.bin": holdout_bytes}
    entries = [file_entry(path, payload) for path, payload in files.items()]
    input_digest = sha256_bytes(b"".join(payload for _, _, payload in decoded))
    manifest_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "dataset",
        "split_policy": "contiguous_suffix",
        "input_sha256": input_digest,
        "input_files": [name for name, _, _ in decoded],
        "holdout_tokens": holdout_tokens,
        "train_tokens": len(train),
        "sources": [{
            "name": "packed-input",
            "ratio": 1.0,
            "dataset": "local-packed",
            "revision": "v1",
            "license": "unknown",
            "source_url": "local",
            "retrieval_timestamp": "local",
            "tokenizer_version": "unknown",
            "filter_policy_version": "none",
            "dedup_policy_version": "none",
            "byte_count": sum(len(payload) for _, _, payload in decoded),
            "token_count": total,
            "input_sha256": input_digest,
            "output_sha256": input_digest,
        }],
        "files": entries,
    }
    return atomic_publish_directory(destination, files, manifest_payload)


def atomic_publish_directory(
    destination: str | Path,
    files: dict[str, bytes],
    manifest: dict[str, Any],
) -> Path:
    """Publish files and a manifest, making the directory visible last."""
    target = Path(destination)
    requested_parent = target.parent
    _reject_link_chain(requested_parent, "destination parent")
    requested_parent.mkdir(parents=True, exist_ok=True)
    _reject_link_chain(requested_parent, "destination parent")
    parent = requested_parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError(f"destination parent is not a directory: {requested_parent}")
    target = parent / target.name
    if target.exists() or _is_link(target):
        raise FileExistsError(target)
    staging = Path(mkdtemp(prefix=f".{target.name}.", dir=str(parent)))
    try:
        for relative_path, payload in files.items():
            path = staging / validate_relative_path(relative_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        normalized = validate_manifest(manifest)
        if not normalized["files"]:
            raise ValueError("published artifact manifest must list files")
        file_paths = {validate_relative_path(path) for path in files}
        expected_paths = {str(entry["path"]) for entry in normalized["files"]}
        if file_paths != expected_paths:
            missing = sorted(expected_paths - file_paths)
            extra = sorted(file_paths - expected_paths)
            raise ValueError(f"manifest/file map mismatch: missing={missing}, extra={extra}")
        for entry in normalized["files"]:
            payload = files[str(entry["path"])]
            actual = file_entry(
                str(entry["path"]),
                payload,
                kind=str(entry["kind"]),
                token_count=int(entry["token_count"]),
            )
            if actual != entry:
                raise ValueError(f"manifest entry mismatch: {entry['path']}")
        manifest_path = staging / "manifest.json"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(normalized, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Validate the complete staging tree before making it visible. This
        # catches truncated uint32 payloads, invalid JSONL/text, range errors
        # and count/hash drift at the publication boundary, not only on load.
        validate_manifest(normalized, root=staging)
        _reject_link_chain(parent, "destination parent")
        if target.exists() or _is_link(target):
            raise FileExistsError(target)
        os.replace(staging, target)
        return target
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_published_artifact(root: str | Path) -> dict[str, Any]:
    """Load and fully validate a published artifact directory."""
    base = _artifact_root(root)
    raw_manifest = base / "manifest.json"
    _reject_link(raw_manifest, "artifact manifest")
    if not raw_manifest.is_file():
        raise ValueError("artifact manifest is missing")
    manifest_path = _safe_artifact_file(base, "manifest.json")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("artifact manifest is unreadable") from exc
    return validate_manifest(payload, root=base)


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "split_packed_tokens",
    "atomic_publish_directory",
    "deduplicate_normalized_lines",
    "file_entry",
    "load_published_artifact",
    "normalize_mix",
    "quarantine_jsonl",
    "read_utf8_records",
    "sha256_bytes",
    "sha256_file",
    "validate_manifest",
    "validate_relative_path",
    "validate_uint32_stream_bytes",
    "validate_uint32_stream_file",
    "write_json_atomic",
]
