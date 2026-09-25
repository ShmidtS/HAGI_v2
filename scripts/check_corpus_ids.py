"""Fail-closed corpus id validation.

The raw ``data/<name>.bin`` streams are token ids from a 248320-entry
vocabulary; the model vocabulary is 32768 after compaction. A run that falls
back to a raw stream therefore feeds out-of-range ids into the embedding
lookup, which is either an illegal-memory access or silent corruption
depending on the device. Several corpora are small enough that "part of the
corpus is already past the vocabulary" reads as "the model is at chance".

This module is the guard the project needed and did not have: it resolves
every corpus a config will actually use through the same
``hagi.data.dataset.dataset_path`` the trainer uses, scans the maximum token
id, and raises instead of training on a stream that cannot be trusted.

Usage:
    python scripts/check_corpus_ids.py --data-dir data
    python scripts/check_corpus_ids.py --data-dir data --config configs/m2_merged_joint.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.config import load_config  # noqa: E402
from hagi.data.dataset import dataset_path  # noqa: E402

SCAN_CHUNK = 64 * 1024 * 1024  # tokens per read; keeps peak RSS bounded


def scan_max_id(path: Path) -> int:
    """Return the maximum token id in a uint32 stream, or -1 when empty."""
    total = path.stat().st_size // 4
    if total == 0:
        return -1
    arr = np.memmap(path, dtype=np.uint32, mode="r")
    best = 0
    for start in range(0, total, SCAN_CHUNK):
        best = max(best, int(arr[start : min(start + SCAN_CHUNK, total)].max()))
    return best


def check(
    data_dir: Path, vocab_size: int, sources: list[str] | None = None
) -> tuple[list[dict], list[str]]:
    """Validate every requested corpus; return (rows, failures)."""
    if sources is None:
        # Strip the compaction suffix so the resolved name matches the config
        # vocabulary (``wikipedia_ru``, not ``wikipedia_ru.compact``).
        sources = sorted(
            {p.name[: -len(".bin")].split(".compact")[0] for p in data_dir.glob("*.bin")}
        )

    rows: list[dict] = []
    failures: list[str] = []
    for name in sources:
        path = dataset_path(data_dir, name)
        if not path.exists():
            rows.append({"source": name, "file": None, "tokens": 0, "max_id": None,
                         "status": "missing"})
            failures.append(f"{name}: no stream found under {data_dir}")
            continue
        tokens = path.stat().st_size // 4
        max_id = scan_max_id(path)
        ok = 0 <= max_id < vocab_size
        rows.append({
            "source": name,
            "file": path.name,
            "tokens": tokens,
            "max_id": max_id,
            "status": "ok" if ok else "OUT_OF_RANGE",
        })
        if not ok:
            failures.append(
                f"{name} ({path.name}): max_id={max_id} >= vocab_size={vocab_size}"
            )
    return rows, failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--config", default=None,
                    help="validate exactly the sources this config trains on")
    ap.add_argument("--json", default=None, help="write the report here")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    vocab = args.vocab_size
    sources = None
    if args.config:
        cfg = load_config(args.config)
        vocab = vocab or cfg.model.vocab_size
        sources = sorted(cfg.train.data.weights)
    if vocab is None:
        ap.error("--vocab-size is required without --config")

    rows, failures = check(data_dir, vocab, sources)
    report = {
        "data_dir": str(data_dir),
        "vocab_size": vocab,
        "corpora": rows,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if failures:
        print(f"\nFAIL: {len(failures)} corpus/corpora unusable for training", file=sys.stderr)
        return 1
    print(f"\nOK: {len(rows)} corpora within vocabulary {vocab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
