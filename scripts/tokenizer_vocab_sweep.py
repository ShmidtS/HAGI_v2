"""Sweep our own BPE vocabulary size against the pruned off-the-shelf baseline.

The hypothesis being tested is not "our tokenizer is better" -- the 32k run
already refuted that. It is narrower and more useful: there is a vocab size
at which a self-trained BPE on our own mix stops losing to gemma-262k pruned
to the same budget, and below it any self-trained tokenizer is dead weight.

For each size the script trains on the same ru-upweighted corpus, serializes
through `tokenizers` (the byte-level representation is fussy and
`tokenizers` knows how to write its own model), then measures tokens per 1000
characters on real ru/en/code text off the network share. Pruning gemma to
the same vocab is what the project's compact_vocab.py does in production, so
comparing against gemma's full 262k would flatter us unfairly.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import time

import gigatoken
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

BS = chr(92)
RAW = BS * 2 + "Home" + BS + "1" + BS + "raw"
CORPUS = "data/bpe_corpus.txt"
SIZES = (32768, 65536, 131072, 262144)


def bytes_to_unicode() -> dict[int, str]:
    codes = list(range(ord("!"), ord("~") + 1)) + [0x20]
    codes += list(range(ord("\xa1"), ord("\xac") + 1))
    codes += list(range(ord("\xae"), ord("\xff") + 1))
    chars = list(codes)
    spare = 0
    for byte in range(256):
        if byte not in codes:
            codes.append(byte)
            chars.append(256 + spare)
            spare += 1
    return dict(zip(codes, map(chr, chars)))


def build(vocab: dict, merges: list, path: str) -> Tokenizer:
    b2u = bytes_to_unicode()
    token_to_id = {b2u[raw[0]]: int(idx) for idx, raw in vocab.items() if len(raw) == 1}
    for idx, raw in vocab.items():
        if len(raw) > 1:
            token_to_id["".join(b2u[b] for b in raw)] = int(idx)
    model = models.BPE(
        vocab=token_to_id,
        merges=[
            ("".join(b2u[b] for b in left), "".join(b2u[b] for b in right))
            for left, right in merges
        ],
        unk_token=None,
        fuse_unk=False,
        byte_fallback=False,
    )
    tk = Tokenizer(model)
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tk.decoder = decoders.ByteLevel()
    tk.save(path)
    return Tokenizer.from_file(path)


def sample(fname: str, n: int = 900) -> str:
    out: list[str] = []
    with io.open(os.path.join(RAW, fname), "r", encoding="utf-8", errors="replace") as h:
        for line in h:
            if len(out) >= n:
                break
            try:
                row = json.loads(line)
            except Exception:
                continue
            text = row.get("text") or row.get("content") or row.get("prompt") or ""
            if isinstance(text, str) and len(text) > 200:
                out.append(text[:4000])
    return "\n".join(out)


def rate(tk: Tokenizer, blob: str) -> float:
    return len(tk.encode(blob).ids) / (len(blob) / 1000)


def main() -> None:
    blobs = {
        "ru": sample("wikipedia_ru.jsonl"),
        "en": sample("slimpajama.jsonl"),
        "code": sample("python_instruct.jsonl"),
    }
    print("baseline gemma-262k (full vocabulary):")
    gemma = gigatoken.Tokenizer("google/gemma-4-E2B-it")
    base = {d: len(gemma.encode(b)) / (len(b) / 1000) for d, b in blobs.items()}
    print("  " + "  ".join(f"{d}={v:.0f}" for d, v in base.items()))

    print("\nour own BPE, ru-upweighted corpus:")
    for size in SIZES:
        t0 = time.time()
        vocab, merges = gigatoken.train_bpe(CORPUS, size, [], "huggingface")
        with open(f"data/sweep_{size}.pkl", "wb") as handle:
            pickle.dump((vocab, merges), handle)
        tk = build(vocab, merges, f"data/sweep_{size}.json")
        cyr = sum(1 for raw in vocab.values() if any(0xC0 <= c <= 0xFF for c in raw))
        row = "  ".join(f"{d}={rate(tk, blobs[d]):.0f}" for d in blobs)
        print(
            f"  V={size:7d}  {row}   "
            f"cyr={cyr/len(vocab)*100:4.1f}%  ({time.time()-t0:.0f}s)"
        )


if __name__ == "__main__":
    main()
