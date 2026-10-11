"""Download FRESH independent corpora for the frontier-injection lever.

Theory: ALGORITHMS.md §4 — at consensus/near-exhaustion the ONLY growth
source is fresh independent data (liveness_data_axis R80: inj > ξ ⟹
D_T > 0 each generation). Two new DOMAINS, independent of the existing
8 corpora:
  books — sedthh/gutenberg_english (long coherent prose; distinct from
          wiki/web/edu)
  sci   — omnibioai/pubmed-abstracts-36M (biomedical abstracts; distinct
          register + vocabulary)
Output: data/newdomains/{books,sci}.jsonl with {"text": ...} rows
(the stream-pack TEXT_KEYS/row_text contract, MIN_CHARS=200).
"""
import gzip
import io
import json
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

OUT = Path("data/newdomains")
OUT.mkdir(parents=True, exist_ok=True)


def append_jsonl(dest: Path, texts):
    with dest.open("a", encoding="utf-8") as fh:
        for t in texts:
            t = t.strip()
            if len(t) >= 200:
                fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def do_books():
    dest = OUT / "books.jsonl"
    if dest.exists() and dest.stat().st_size > 200_000_000:
        print("books.jsonl already present", flush=True)
        return
    import pyarrow.parquet as pq
    names = [
        "data/train-00000-of-00037-f5fce855b93d2d02.parquet",
        "data/train-00001-of-00037-9f227d74fc154ce9.parquet",
        "data/train-00002-of-00037-bea1b37be317e1da.parquet",
    ]
    for n in names:
        p = hf_hub_download("sedthh/gutenberg_english", n,
                            repo_type="dataset")
        tbl = pq.read_table(p, columns=["TEXT"])
        col = tbl.column("TEXT").to_pylist()
        append_jsonl(dest, (t for t in col if isinstance(t, str)))
        print(f"books: packed {p}", flush=True)
        os.remove(p)  # free cache; jsonl is the keep-form


def do_sci():
    dest = OUT / "sci.jsonl"
    if dest.exists() and dest.stat().st_size > 200_000_000:
        print("sci.jsonl already present", flush=True)
        return
    names = ["AI_Drug_Discovery.jsonl.gz", "ALS_MND.jsonl.gz",
             "Addiction_Substance.jsonl.gz", "Aging_Longevity.jsonl.gz"]
    for n in names:
        p = hf_hub_download("omnibioai/pubmed-abstracts-36M", n,
                            repo_type="dataset")
        texts = []
        with gzip.open(p, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                # probe the same TEXT_KEYS priority as the packer
                for k in ("text", "content", "abstract", "document", "body"):
                    v = row.get(k) if isinstance(row, dict) else None
                    if isinstance(v, str):
                        texts.append(v)
                        break
        append_jsonl(dest, texts)
        print(f"sci: packed {p} ({len(texts)} rows)", flush=True)
        os.remove(p)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("books", "all"):
        do_books()
    if which in ("sci", "all"):
        do_sci()
    for f in sorted(OUT.glob("*.jsonl")):
        print(f.name, f.stat().st_size // 1_000_000, "MB", flush=True)
    print("DONE", flush=True)
