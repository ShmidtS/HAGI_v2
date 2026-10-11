"""Pack + compact a NEW domain corpus into the training format.

Reuses the exact production path (stream_pack_corpus.pack → gigatoken
encode_batch_list, 262144-space int32 shards) and then compacts through
the EXISTING data/vocab_map.npz (262144 → 32768), producing
``data/<name>.compact2.bin`` — directly consumable by PackedStream /
PackedMixDataset (dataset_path prefers .compact2.bin).

Usage:
    python scripts/data/pack_new_domain.py --name books \
        --source data/newdomains/books.jsonl

The unigram table is NOT rebuilt: the frozen compact vocabulary is part
of the incumbent model's identity (compact↔gemma mapping); a new corpus
must map INTO it, never change it (fallback id absorbs unseen tail —
measured unk rate is reported so a pathological corpus is visible).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))

spec = importlib.util.spec_from_file_location(
    "spc", ROOT / "scripts" / "data" / "stream_pack_corpus.py")
spc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--tokenizer", default="google/gemma-4-E2B-it")
    ap.add_argument("--shard-tokens", type=int, default=4_000_000)
    args = ap.parse_args()

    encoder = spc.build_encoder(args.tokenizer)
    unigram = np.zeros(262144, dtype=np.int64)

    class _A:  # the attrs pack() reads
        out_dir = str(ROOT / "data")
        batch_docs = 500
        max_batch_chars = 32_000_000  # books docs are ~500KB each
        tokenizer = args.tokenizer
    _A.shard_tokens = args.shard_tokens

    manifest = spc.pack(args.name, args.source, _A, encoder, unigram)
    print(json.dumps({k: manifest[k] for k in ("tokens", "documents")}))

    # compact through the FROZEN map: all shards -> ONE .compact2.bin
    from compact_vocab import rewrite_stream
    vm = np.load(ROOT / "data" / "vocab_map.npz")
    old_to_new = vm["old_to_new"]
    fb = int(old_to_new[3])  # compact id of the gemma unk token
    assert fb >= 0, "unk not retained in the frozen map"
    dest = ROOT / "data" / f"{args.name}.compact2.bin"
    tmp = dest.with_suffix(".tmp.bin")
    total_written = 0
    total_repl = 0
    with open(dest, "wb"):
        pass  # truncate
    for shard in sorted((ROOT / "data" / args.name / "shards").glob("*.bin")):
        w, r = rewrite_stream(shard, tmp, old_to_new, fb)
        total_written += w
        total_repl += r
        with open(tmp, "rb") as src, open(dest, "ab") as out:
            while True:
                chunk = src.read(1 << 24)
                if not chunk:
                    break
                out.write(chunk)
    tmp.unlink(missing_ok=True)
    unk_rate = total_repl / max(total_written, 1)
    print(f"compact: {total_written/1e6:.1f}M tokens, unk={unk_rate:.6f}, "
          f"-> {dest.name}")
    if unk_rate > 0.02:
        print("WARNING: unk rate > 2% — corpus vocabulary poorly covered",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
