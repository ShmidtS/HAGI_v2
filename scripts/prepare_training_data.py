#!/usr/bin/env python3
"""Bounded, opt-in preparation of versioned packed training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np

from hagi.data.artifacts import (
    MANIFEST_SCHEMA_VERSION,
    atomic_publish_directory,
    file_entry,
    load_published_artifact,
    quarantine_jsonl,
    read_utf8_records,
    sha256_file,
)

DEFAULT_EOS_ID = 1
DEFAULT_VOCAB_SIZE = 0


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_limit(max_bytes: int, timeout: int) -> None:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if type(timeout) is not int or timeout < 1:
        raise ValueError("timeout must be a positive integer")


class _NoRedirect(HTTPRedirectHandler):
    """Fail closed instead of following a server-controlled redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise HTTPError(newurl, code, "HTTP redirects are not allowed", headers, fp)


def _open_no_redirect(request: Request, timeout: int):
    """Open one HTTP(S) request without redirect or cookie side effects."""
    opener = build_opener(_NoRedirect())
    return opener.open(request, timeout=timeout)


def _expected_digest(expected_sha256: str | None) -> str | None:
    """Validate an optional expected SHA-256 digest."""
    if expected_sha256 is None:
        return None
    value = expected_sha256.lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    return value


def acquire_source(
    source: str,
    output: Path,
    max_bytes: int,
    expected_sha256: str | None,
    timeout: int,
) -> None:
    """Copy/stream a bounded source; remote sources require an expected hash."""
    _validate_limit(max_bytes, timeout)
    expected = _expected_digest(expected_sha256)
    source_path = Path(source.removeprefix("file://")) if source.startswith("file://") else Path(source)
    parsed = urlparse(source)
    remote = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    windows_drive = len(source) >= 2 and source[1] == ":"

    if remote:
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("credentials are not allowed in source URLs")
        if expected is None:
            raise ValueError("remote acquisition requires expected_sha256")
        request = Request(source, headers={"Accept-Encoding": "identity"})
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with _open_no_redirect(request, timeout) as response:
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except (TypeError, ValueError) as exc:
                        raise ValueError("invalid Content-Length") from exc
                    if declared_size < 0:
                        raise ValueError("invalid Content-Length")
                    if declared_size > max_bytes:
                        raise ValueError("content-length exceeds max_bytes")
                handle = tempfile.NamedTemporaryFile(
                    dir=output.parent,
                    prefix=f".{output.name}.",
                    delete=False,
                )
                temporary = Path(handle.name)
                digest = hashlib.sha256()
                total = 0
                with handle:
                    while True:
                        block = response.read(min(64 * 1024, max_bytes - total + 1))
                        if not block:
                            break
                        total += len(block)
                        if total > max_bytes:
                            raise ValueError("download exceeds max_bytes")
                        digest.update(block)
                        handle.write(block)
                if digest.hexdigest() != expected:
                    raise ValueError("sha256 mismatch")
            os.replace(temporary, output)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return

    if not remote and not source.startswith("file://") and parsed.scheme and not windows_drive:
        raise ValueError("only http, https, file, and local paths are supported")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.stat().st_size > max_bytes:
        raise ValueError("source exceeds max_bytes")
    if expected is not None and sha256_file(source_path) != expected:
        raise ValueError("sha256 mismatch")
    _copy_atomic(source_path, output)


def _official_gigatoken_batch(name: str):
    """Return the official native Gigatoken batch encoder.

    Reference: https://pypi.org/project/gigatoken and
    https://github.com/marcelroed/gigatoken. Gigatoken 0.10 exposes
    ``encode_batch_list`` directly on ``Tokenizer(model_name)``; that native
    seam avoids materializing an AwkwardArray and adds no special tokens.
    ``as_hf().encode_batch`` is not present on this installed compatibility
    wrapper, so it must not be guessed here.
    """
    try:
        import gigatoken
    except ImportError as exc:
        raise RuntimeError("gigatoken is required for prepare") from exc
    tokenizer = gigatoken.Tokenizer(name)
    if not hasattr(tokenizer, "encode_batch_list"):
        raise RuntimeError("installed gigatoken lacks the documented encode_batch_list API")
    return tokenizer.encode_batch_list


def default_tokenizer(name: str):
    return _official_gigatoken_batch(name)


def _encode_documents(
    documents: list[str],
    tokenizer_name: str,
    tokenizer_callable,
    batch_size: int,
) -> list[list[int]]:
    encoder = tokenizer_callable or default_tokenizer(tokenizer_name)
    encoded: list[list[int]] = []
    for start in range(0, len(documents), batch_size):
        rows = encoder(documents[start : start + batch_size])
        if len(rows) != len(documents[start : start + batch_size]):
            raise ValueError("tokenizer returned a different number of rows")
        encoded.extend(rows)
    return encoded


def prepare_data(
    input_file: Path,
    output_dir: Path,
    tokenizer_name: str,
    tokenizer_callable=None,
    *,
    source_name: str = "local",
    source_url: str | None = None,
    dataset: str = "local",
    revision: str = "local",
    license_name: str = "unspecified",
    retrieval_timestamp: str = "unspecified",
    eos_token_id: int = DEFAULT_EOS_ID,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    shard_tokens: int = 1_000_000,
    batch_size: int = 256,
    max_input_bytes: int = 64 * 1024 * 1024,
) -> Path:
    """Prepare one versioned packed artifact atomically under a new directory."""
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    if type(max_input_bytes) is not int or max_input_bytes < 1:
        raise ValueError("max_input_bytes must be a positive integer")
    if input_file.stat().st_size > max_input_bytes:
        raise ValueError("input exceeds max_input_bytes")
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("eos_token_id must be a non-negative integer")
    if type(vocab_size) is not int or vocab_size < 0:
        raise ValueError("vocab_size must be a non-negative integer")
    if type(shard_tokens) is not int or shard_tokens < 2:
        raise ValueError("shard_tokens must be an integer >= 2")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if output_dir.exists():
        raise FileExistsError(output_dir)

    input_payload = input_file.read_bytes()
    documents, quarantine = read_utf8_records(input_file)
    if not documents:
        raise ValueError("no valid non-empty documents remain after validation")

    encoded = _encode_documents(documents, tokenizer_name, tokenizer_callable, batch_size)
    if len(encoded) != len(documents):
        raise ValueError("tokenizer/document count mismatch")

    normalized_rows: list[list[int]] = []
    for row in encoded:
        ids = [int(value) for value in row]
        if not ids:
            raise ValueError("tokenizer returned an empty document")
        if vocab_size and (min(ids) < 0 or max(ids) >= vocab_size):
            raise ValueError("token id outside declared vocabulary")
        if eos_token_id >= (1 << 32):
            raise ValueError("eos_token_id does not fit uint32")
        normalized_rows.append(ids)

    files: dict[str, bytes] = {}
    file_entries: list[dict[str, object]] = []
    shard_index = 0
    shard_tokens_written = 0
    current = bytearray()
    for row in normalized_rows:
        required = len(row) + 1
        if shard_tokens_written and shard_tokens_written + required > shard_tokens:
            payload = bytes(current)
            relative = f"shards/{shard_index:06d}.bin"
            files[relative] = payload
            file_entries.append(file_entry(relative, payload))
            shard_index += 1
            current = bytearray()
            shard_tokens_written = 0
        if required > shard_tokens:
            raise ValueError("one document exceeds shard_tokens")
        current.extend(np.asarray([*row, eos_token_id], dtype=np.uint32).tobytes())
        shard_tokens_written += required
    if current:
        payload = bytes(current)
        relative = f"shards/{shard_index:06d}.bin"
        files[relative] = payload
        file_entries.append(file_entry(relative, payload))

    quarantine_payload = quarantine_jsonl(quarantine)
    if quarantine_payload:
        files["quarantine/records.jsonl"] = quarantine_payload
        file_entries.append(file_entry("quarantine/records.jsonl", quarantine_payload, kind="jsonl"))

    source_output_hash = hashlib.sha256()
    for entry in file_entries:
        if entry["kind"] == "tokens":
            source_output_hash.update(files[str(entry["path"])])
    total_token_count = sum(int(entry["token_count"]) for entry in file_entries if entry["kind"] == "tokens")
    total_byte_count = sum(int(entry["byte_count"]) for entry in file_entries)
    origin = source_url or str(input_file)
    parsed_origin = urlparse(origin)
    if parsed_origin.username is not None or parsed_origin.password is not None:
        raise ValueError("credentials are not allowed in source_url")
    source = {
        "name": source_name,
        "ratio": 1.0,
        "dataset": dataset,
        "revision": revision,
        "license": license_name,
        "source_url": origin,
        "retrieval_timestamp": retrieval_timestamp,
        "tokenizer_version": tokenizer_name,
        "filter_policy_version": "utf8-strip-v1",
        "dedup_policy_version": "exact-sha256-strip-v1",
        "byte_count": len(input_payload),
        "token_count": total_token_count,
        "input_sha256": hashlib.sha256(input_payload).hexdigest(),
        "output_sha256": source_output_hash.hexdigest(),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "dataset",
        "vocab_size": vocab_size or None,
        "shard_token_limit": shard_tokens,
        "quarantine_count": len(quarantine),
        "total_byte_count": total_byte_count,
        "total_token_count": total_token_count,
        "sources": [source],
        "files": file_entries,
    }
    atomic_publish_directory(output_dir, files, manifest)
    return output_dir / f"shards/{shard_index:06d}.bin"


def validate_output(output_dir: Path) -> dict[str, object]:
    """Fully validate a published artifact and return its manifest."""
    return load_published_artifact(output_dir)


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input")
    parser.add_argument("output_dir")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--source-name", default="local")
    parser.add_argument("--source-url")
    parser.add_argument("--dataset", default="local")
    parser.add_argument("--revision", default="local")
    parser.add_argument("--license", default="unspecified")
    parser.add_argument("--retrieval-timestamp", default="unspecified")
    parser.add_argument("--eos-token-id", type=int, default=DEFAULT_EOS_ID)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--shard-tokens", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-input-bytes", type=int, default=64 * 1024 * 1024)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    acquire = sub.add_parser("acquire")
    acquire.add_argument("source")
    acquire.add_argument("output")
    acquire.add_argument("--max-bytes", type=int, required=True)
    acquire.add_argument("--expected-sha256")
    acquire.add_argument("--timeout", type=int, default=30)

    prepare = sub.add_parser("prepare")
    _add_prepare_arguments(prepare)

    validate = sub.add_parser("validate")
    validate.add_argument("output_dir")

    args = parser.parse_args(argv)
    if args.cmd == "acquire":
        acquire_source(args.source, Path(args.output), args.max_bytes, args.expected_sha256, args.timeout)
    elif args.cmd == "prepare":
        prepare_data(
            Path(args.input),
            Path(args.output_dir),
            args.tokenizer,
            source_name=args.source_name,
            source_url=args.source_url,
            dataset=args.dataset,
            revision=args.revision,
            license_name=args.license,
            retrieval_timestamp=args.retrieval_timestamp,
            eos_token_id=args.eos_token_id,
            vocab_size=args.vocab_size,
            shard_tokens=args.shard_tokens,
            batch_size=args.batch_size,
            max_input_bytes=args.max_input_bytes,
        )
    else:
        print(json.dumps(validate_output(Path(args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
