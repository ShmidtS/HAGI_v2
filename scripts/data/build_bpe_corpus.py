"""Build a ru-upweighted training text file for gigatoken.train_bpe.

Why train our own: every off-the-shelf tokenizer we measured either wastes
most of its vocabulary on languages we do not train (gemma 262k and Qwen 248k
collapse to 97.9% / 97.3% mass after pruning to 32k, and both spend the head
on English), or is small and blind to Cyrillic (nomic-30k: 6.21 tokens/word
on Russian against gemma's 2.23). A 32k vocabulary trained on our own mix has
no dead rows and gets Russian by construction rather than by luck.

Russian is upweighted by repeating ru documents: ru is one third of our
evaluation domains but the corpora where merges are learned are dominated by
English web text by volume. GigaChat used the same lever (upweighting target
languages when training) and reached 2.2-2.5 tokens/word on Slavic text at
128k, which our own 32k has to beat at 2.23.

Input: the JSONL corpora on the network share. Output: one utf-8 text file
with documents separated by a form feed, which is what BPE training wants and
what keeps document boundaries from merging into each other.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

BS = chr(92)
RAW = BS * 2 + "Home" + BS + "1" + BS + "raw"

# ru sources are repeated `repeat` times; the rest once. Mix ratios follow
# data/mix.json so the tokenizer optimizes for the distribution we train on.
SOURCES: list[tuple[str, int]] = [
    ("wikipedia_ru.jsonl",    3),
    ("oscar_ru.jsonl",        2),
    ("wikipedia_en.jsonl",    1),
    ("slimpajama.jsonl",      1),
    ("edu.jsonl",             1),
    ("openwebmath.jsonl",     1),
    ("smoltalk.jsonl",        1),
    ("python_instruct.jsonl", 2),
    ("tinystories.jsonl",     1),
]


def iter_documents(path: Path, budget: int):
    """Yield decoded text fields until `budget` characters are collected."""
    with io.open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            text = row.get("text") or row.get("content") or row.get("prompt") or ""
            if not isinstance(text, str) or len(text) < 200:
                continue
            yield text
            budget -= len(text)
            if budget <= 0:
                return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bpe_corpus.txt")
    ap.add_argument("--per-source-chars", type=int, default=60_000_000,
                    help="characters per source pass, before ru upweighting")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    t0 = time.time()
    with io.open(out_path, "w", encoding="utf-8") as out:
        for name, repeat in SOURCES:
            src = os.path.join(RAW, name)
            if not os.path.exists(src):
                print(f"skip missing {name}")
                continue
            for _ in range(repeat):
                for text in iter_documents(Path(src), args.per_source_chars):
                    # form feed: a token boundary that never crosses documents
                    out.write(text)
                    out.write("\f")
                    written += len(text) + 1
            print(f"{name:24s} x{repeat}  total={written/1e6:.0f}M chars")
    print(f"wrote {out_path} {written/1e6:.1f}M chars in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
