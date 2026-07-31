"""Chunked streaming preprocess: raw .jsonl -> flat uint32 .bin with EOS(1) doc delimiters.

For each <name>.jsonl in data/raw/:
  read line batches -> json.loads -> text ->
  gigatoken encode_batch_list (list[list[int]], no BOS/EOS auto) ->
  append EOS(1) per document -> flatten -> write to data/<name>.bin as uint32 LE.

Format matches remote v4_3b_gemma/*.bin (flat uint32 LE, EOS=1 doc delimiter,
no BOS). vocab 262144 > 65535 -> uint32 (MemmapDataset auto dtype).

Memory-bounded: flush encode batch when >= BATCH lines OR >= CHAR_BUDGET chars
accumulated; write bytes when the bytearray buffer >= FLUSH_BYTES. bytearray
extend is amortized O(1), so this is O(n) overall (no np.append O(n^2)).

Usage: python scripts/preprocess_gemma.py [name1 name2 ...]
  (no args -> process all *.jsonl present in data/raw/)
"""
import glob
import json
import os
import sys
import time
from array import array

import gigatoken as gt

RAW_DIR = r"C:\HAGI_v2\data\raw"
OUT_DIR = r"C:\HAGI_v2\data"
TOKENIZER = "google/gemma-4-E2B-it"
EOS_ID = 1
BATCH = 512
CHAR_BUDGET = 8_000_000  # flush encode batch early if text chars exceed this
FLUSH_BYTES = 64 * 1024 * 1024  # flush output file when buffer >= 64 MiB
LOG = r"C:\HAGI_v2\_preprocess.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def process_file(tok, name):
    rp = os.path.join(RAW_DIR, f"{name}.jsonl")
    op = os.path.join(OUT_DIR, f"{name}.bin")
    if not os.path.exists(rp):
        log(f"SKIP {name}: no raw file at {rp}")
        return
    log(f"START {name}.jsonl -> {name}.bin ({os.path.getsize(rp)/1e9:.3f} GB raw)")
    n_tok = 0
    n_doc = 0
    n_bad = 0
    buf = bytearray()  # amortized O(1) extend; flush to file periodically
    batch_texts = []
    batch_chars = 0

    def encode_to_buf(texts):
        nonlocal n_tok, n_doc
        encs = tok.encode_batch_list(texts)  # list[list[int]], no BOS/EOS auto
        for toks in encs:
            a = array("I", toks)            # host uint32; tobytes() = LE on x86
            a.append(EOS_ID)                # EOS doc delimiter
            buf.extend(a.tobytes())
            n_tok += len(a)
            n_doc += 1

    def flush_buf(fout):
        if len(buf) >= FLUSH_BYTES:
            fout.write(buf)
            buf.clear()
            log(f"  {name}: {n_doc} docs, {n_tok} tokens so far")

    open(op, "wb").close()
    t0 = time.time()
    with open(rp, encoding="utf-8", errors="replace") as fin, \
         open(op, "ab") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                txt = obj.get("text", "")
            except Exception as e:
                n_bad += 1
                if n_bad <= 5:
                    log(f"  {name}: json err skip #{n_bad}: {e}")
                continue
            if not isinstance(txt, str):
                txt = str(txt) if txt is not None else ""
            batch_texts.append(txt)
            batch_chars += len(txt)
            if len(batch_texts) >= BATCH or batch_chars >= CHAR_BUDGET:
                encode_to_buf(batch_texts)
                batch_texts.clear()
                batch_chars = 0
                flush_buf(fout)
        if batch_texts:
            encode_to_buf(batch_texts)
        if buf:
            fout.write(buf)
            buf.clear()
    sz = os.path.getsize(op)
    log(f"DONE {name}: {n_doc} docs, {n_tok} tokens, {n_bad} bad lines, "
        f"{sz/1e9:.3f} GB bin, {time.time()-t0:.0f}s")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isdir(RAW_DIR):
        log(f"NO raw dir {RAW_DIR} -- nothing to do")
        return
    log(f"loading tokenizer {TOKENIZER}")
    tok = gt.Tokenizer(TOKENIZER)
    log(f"tokenizer loaded; vocab={tok.vocab_size}; eos_id={EOS_ID}")

    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(RAW_DIR, "*.jsonl"))
        )
    log(f"will process {len(names)} files: {names}")

    for name in names:
        process_file(tok, name)
    log("ALL DONE")


if __name__ == "__main__":
    main()
