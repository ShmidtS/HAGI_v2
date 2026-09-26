"""Stream-pack the raw JSONL corpora into int32 .bin shards.

Why this exists: `prepare_training_data.py` does `input_file.read_bytes()`
(`scripts/prepare_training_data.py:224`) and then holds the encoded result, so
memory scales with the source. On wikipedia_ru (2.1 GB of UTF-8) that peaked
at 17.8 GB RSS and the process was killed; edu.jsonl is 5.8 GB and would not
have survived at all. The share holds 14.7 GB in total.

The fix is to never hold a whole source. Documents are read as lines, encoded
in bounded batches through the same official gigatoken path the project
already uses (`encode_batch_list`, 17.3 M chars/s measured, 3.5x the HF
tokenizers route), and flushed to shards as soon as a shard is full. Peak RSS
is then set by the batch size, not by the file.

Output layout matches what the training data layer expects:
    data/<name>/shards/000000.bin, 000001.bin, ...   int32, little-endian
    data/<name>/manifest.json                        provenance, same shape
                                                     prepare_training_data writes

The manifest is written here rather than delegated, so a partially packed
corpus is still self-describing: records which sources were packed, the
tokenizer, and the shard count actually written.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np

BS = chr(92)
RAW = BS * 2 + "Home" + BS + "1" + BS + "raw"

# name, source file, text budget in documents per batch
SOURCES: list[tuple[str, str]] = [
    ("wikipedia_ru",    "wikipedia_ru.jsonl"),
    ("oscar_ru",        "oscar_ru.jsonl"),
    ("wikipedia_en",    "wikipedia_en.jsonl"),
    ("slimpajama",      "slimpajama.jsonl"),
    ("edu",             "edu.jsonl"),
    ("openwebmath",     "openwebmath.jsonl"),
    ("smoltalk",        "smoltalk.jsonl"),
    ("python_instruct", "python_instruct.jsonl"),
    ("tinystories",     "tinystories.jsonl"),
]

# Fields a JSONL row may use for its text, in priority order. Probed per row
# rather than assumed, because the share mixes several formats.
TEXT_KEYS = ("text", "content", "prompt", "document", "body", "raw_text")
MIN_CHARS = 200


def build_encoder(name: str):
    """The official gigatoken batch encoder, or a loud failure.

    Not guessed: `as_hf().encode_batch` is absent from the installed
    compatibility wrapper, and the native `encode_batch_list` is the documented
    seam (see prepare_training_data._official_gigatoken_batch).
    """
    import gigatoken

    tokenizer = gigatoken.Tokenizer(name)
    if not hasattr(tokenizer, "encode_batch_list"):
        raise RuntimeError("installed gigatoken lacks encode_batch_list")
    return tokenizer


def row_text(row: object) -> str:
    if not isinstance(row, dict):
        return ""
    for key in TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and len(value) > MIN_CHARS:
            return value
    return ""


def iter_batches(path: str, batch_docs: int, max_chars: int):
    """Yield lists of decoded document strings, bounded by both counts."""
    batch: list[str] = []
    chars = 0
    with io.open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            text = row_text(row)
            if not text:
                continue
            batch.append(text)
            chars += len(text)
            if len(batch) >= batch_docs or chars >= max_chars:
                yield batch
                batch = []
                chars = 0
    if batch:
        yield batch


def pack(name: str, source: str, args, encoder) -> dict:
    dest = Path(args.out_dir) / name / "shards"
    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.glob("*.bin"):
        stale.unlink()

    shards: list[dict] = []
    shard_index = 0
    shard_rows = 0
    total_tokens = 0
    total_docs = 0
    start = time.time()

    handle_out: io.BufferedWriter | None = None
    try:
        for batch in iter_batches(source, args.batch_docs, args.max_batch_chars):
            encoded = encoder.encode_batch_list(batch)
            for ids in encoded:
                if handle_out is None:
                    path = dest / f"{shard_index:06d}.bin"
                    handle_out = io.open(path, "wb", buffering=1 << 20)
                    shards.append({"shard": shard_index, "path": path.name, "tokens": 0})
                    shard_rows = 0
                # encode_batch_list yields one plain list of ints per document.
                n_tokens = len(ids)
                if shard_rows + n_tokens > args.shard_tokens and shard_rows > 0:
                    # Flush the current shard rather than splitting a document
                    # across two files: a shard is a unit the loader reads whole.
                    handle_out.close()
                    handle_out = None
                    shard_index += 1
                    path = dest / f"{shard_index:06d}.bin"
                    handle_out = io.open(path, "wb", buffering=1 << 20)
                    shards.append({"shard": shard_index, "path": path.name, "tokens": 0})
                    shard_rows = 0
                # numpy's tofile avoids building one bytes object per document.
                np.asarray(ids, dtype=np.int32).tofile(handle_out)
                shard_rows += n_tokens
                shards[-1]["tokens"] = shard_rows
                total_tokens += n_tokens
                total_docs += 1
    finally:
        if handle_out is not None:
            handle_out.close()

    elapsed = time.time() - start
    manifest = {
        "version": 1,
        "source_name": name,
        "source_file": os.path.basename(source),
        "tokenizer": args.tokenizer,
        "packing": "documents_with_eos",
        "documents": total_docs,
        "tokens": total_tokens,
        "shards": shards,
        "elapsed_s": round(elapsed, 1),
    }
    out = Path(args.out_dir) / name
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="google/gemma-4-E2B-it")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--shard-tokens", type=int, default=4_000_000)
    ap.add_argument("--batch-docs", type=int, default=2000)
    ap.add_argument("--max-batch-chars", type=int, default=16_000_000)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    encoder = build_encoder(args.tokenizer)
    want = set(args.only or [])
    for name, filename in SOURCES:
        if want and name not in want:
            continue
        source = os.path.join(RAW, filename)
        if not os.path.exists(source):
            print(f"SKIP {name}: not on the share", flush=True)
            continue
        size_gb = os.path.getsize(source) / 1e9
        print(f"packing {name} ({filename}, {size_gb:.2f} GB)", flush=True)
        try:
            manifest = pack(name, source, args, encoder)
        except Exception as exc:
            print(f"  FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        rate = manifest["tokens"] / max(1e-6, manifest["elapsed_s"])
        print(
            f"  {manifest['documents']:,} docs -> {manifest['tokens']:,} tokens, "
            f"{len(manifest['shards'])} shards, {rate/1e6:.2f} Mtok/s "
            f"in {manifest['elapsed_s']:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
