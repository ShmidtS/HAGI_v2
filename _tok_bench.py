"""Measure tokenizer compression on OUR real corpus (network share).

Territory, not map: tokens-per-character on actual ru/en/math/code text.
The previous attempt fed gemma-ids back through gemma, which measures nothing.
This reads the original JSONL directly, so the numbers are ground truth.

The UNC path is built from chr(92) because "\\Home\\1\\raw" in a Python
string literal turns \\1 into a control character.
"""
import io
import json
import os
import time

from transformers import AutoTokenizer

BS = chr(92)
RAW = BS * 2 + "Home" + BS + "1" + BS + "raw"

SAMPLES = {
    "ru":     ("wikipedia_ru.jsonl",    4000),
    "en":     ("slimpajama.jsonl",      4000),
    "math":   ("openwebmath.jsonl",     4000),
    "code":   ("python_instruct.jsonl", 4000),
    "ru_web": ("oscar_ru.jsonl",        4000),
}

CANDS = {
    "gemma-262k (cur)": "google/gemma-4-E2B-it",
    "nomic-30k":        "nomic-ai/nomic-embed-text-v1.5",
    "Qwen3.8-248k":     "models/qwen3_8-27b-tokenizer",
}


def load(fname, n_lines, min_chars=200):
    out, got = [], 0
    path = os.path.join(RAW, fname)
    with io.open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if got >= n_lines:
                break
            try:
                row = json.loads(line)
            except Exception:
                continue
            text = row.get("text") or row.get("content") or row.get("prompt") or ""
            if isinstance(text, str) and len(text) > min_chars:
                out.append(text[:4000])
                got += 1
    return out


def main() -> None:
    texts = {}
    for dom, (fname, n) in SAMPLES.items():
        if not os.path.exists(os.path.join(RAW, fname)):
            print(f"skip {dom}: {fname} missing")
            continue
        texts[dom] = load(fname, n)
        chars = sum(len(t) for t in texts[dom])
        print(f"{dom:7s} {fname:24s} lines={len(texts[dom]):5d} chars={chars:9d}")

    tokens = {}
    for label, repo in CANDS.items():
        try:
            tokens[label] = AutoTokenizer.from_pretrained(repo, local_files_only=True)
            print(f"loaded {label}")
        except Exception as exc:
            print(f"MISS  {label}: {type(exc).__name__} {str(exc)[:70]}")

    doms = list(texts)
    print()
    print("=== tokens per 1000 chars (lower = more compact) ===")
    print("tokenizer".ljust(20) + "".join(d.ljust(10) for d in doms) + "vocab")
    for label, tk in tokens.items():
        row = label.ljust(20)
        for dom in doms:
            blob = "\n".join(texts[dom])[:2_000_000]
            ids = tk.encode(blob, add_special_tokens=False)
            row += f"{len(ids) / (len(blob) / 1000):.1f}".ljust(10)
        vocab = getattr(tk, "vocab_size", len(tk))
        print(row + str(vocab))

    print()
    print("=== Russian: tokens per word, and encode speed ===")
    ru_blob = "\n".join(texts.get("ru", []))[:400_000]
    words = max(1, len(ru_blob.split()))
    for label, tk in tokens.items():
        start = time.time()
        ids = tk.encode(ru_blob, add_special_tokens=False)
        dt = max(1e-6, time.time() - start)
        print(
            f"{label.ljust(20)} {len(ids) / words:.2f} tok/word  "
            f"{len(ru_blob) / 1000 / dt:.0f} kchar/s"
        )


if __name__ == "__main__":
    main()
