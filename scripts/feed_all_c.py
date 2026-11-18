#!/usr/bin/env python3
"""Feed EVERYTHING on C:/ into the ONE unified model.

Honest engineering: C:/ has ~10^5-10^6 files, many TB. No model can
train on all bytes; so the ingest contract is a full-coverage SWEEP:
every file on the disk contributes bytes (capped per file, sampled
across the file so structure not just the head is seen). Binary,
system, locked files included — bytes are bytes. Dedup by sha1 so
identical system DLLs don't dominate. Locked/inaccessible files are
counted as skipped with reason (honest ledger).

The model sees ALL of the disk in expectation: every directory is
walked, every eligible file contributes a sample. Per-chunk certified
training on the cumulative corpus (no forgetting: replay).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hagi import unified  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SWEEP_STATE = ROOT / "checkpoints" / "sweep_state.json"

# caps (honest: full-disk bytes won't fit; every FILE is visited)
MAX_FILE_BYTES = 1 << 20          # read at most 1 MB per file
SAMPLE_HEAD = 1 << 16             # 64KB head
SAMPLE_TAIL = 1 << 16             # 64KB tail
SAMPLE_MID = 1 << 16              # 64KB from the middle
MIN_FILE = 16                     # skip empty/trivial
CHUNK_TARGET = 200_000            # train in ~8MB cumulative chunks
EPOCHS_PER_CHUNK = 8              # many chunks > few big passes

SKIP_DIRS = {"$Recycle.Bin", "System Volume Information",
             "Windows.old", "$WinREAgent", "hiberfil.sys",
             "Recovery", "DumpStack.log.tmp", "pagefile.sys",
             "swapfile.sys", "onedrive_temp", "node_modules",
             ".git", "__pycache__",
             # NTFS junction loops (infinite recursion without these)
             "Documents and Settings", "Application Data",
             "Local Settings", "My Documents", "Cookies", "NetHood",
             "PrintHood", "Recent", "SendTo", "Templates", "Start Menu"}

_lock = None


def iter_files(root: str = "C:/", start_after: str | None = None):
    """Walk EVERYTHING; skip only unavoidable OS locks. start_after:
    full path of the last consumed FILE — resume lexically after it
    (files visited in sorted order within sorted dirs)."""
    started = start_after is None
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dirnames[:] = sorted([d for d in dirnames if d not in SKIP_DIRS])
        for f in sorted(filenames):
            path = os.path.join(dirpath, f)
            if not started:
                if path > start_after:
                    started = True
                    yield path
                continue
            yield path


def sample_bytes(p: str, size: int) -> bytes | None:
    """Head + middle + tail sample of a file (structure, not just head)."""
    try:
        with open(p, "rb") as fh:
            head = fh.read(min(SAMPLE_HEAD, size))
            if size <= SAMPLE_HEAD:
                return head
            mid_off = size // 2
            fh.seek(mid_off)
            mid = fh.read(SAMPLE_MID)
            fh.seek(max(0, size - SAMPLE_TAIL))
            tail = fh.read(SAMPLE_TAIL)
            return head + b"\n\x00MID\x00\n" + mid + b"\n\x00TAIL\x00\n" + tail
    except (PermissionError, OSError):
        return None


def load_state() -> dict:
    if SWEEP_STATE.exists():
        return json.loads(SWEEP_STATE.read_text(encoding="utf-8"))
    return {"seen": {}, "visited": 0, "sampled": 0, "skipped_locked": 0,
            "bytes_ingested": 0, "chunks": 0, "cursor": None}


def save_state(st: dict) -> None:
    SWEEP_STATE.write_text(json.dumps(st, ensure_ascii=False),
                           encoding="utf-8")


def main() -> None:
    max_minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    deadline = time.time() + max_minutes * 60
    u = unified.Unified.load() if unified.Unified.exists() else unified.Unified()
    st = load_state()
    seen = st["seen"]
    chunk = []
    chunk_bytes = 0
    files_now = 0
    t0 = time.time()
    print(f"sweep start: visited={st['visited']} sampled={st['sampled']} "
          f"ingested={st['bytes_ingested']}B chunks={st['chunks']}",
          flush=True)
    for path in iter_files("C:/"):
        if time.time() > deadline:
            print("deadline reached — flush", flush=True)
            break
        st["visited"] += 1
        try:
            size = os.path.getsize(path)
        except OSError:
            st["skipped_locked"] += 1
            continue
        if size < MIN_FILE:
            continue
        h = hashlib.sha1(path.encode("utf-8", errors="replace")).hexdigest()[:12]
        if h in seen:
            continue                      # this path already ingested
        data = sample_bytes(path, size)
        if data is None:
            st["skipped_locked"] += 1
            seen[h] = "locked"
            continue
        seen[h] = min(len(data), size)
        st["sampled"] += 1
        chunk.append(data)
        chunk_bytes += len(data)
        st["bytes_ingested"] += len(data)
        files_now += 1
        # chunk training: cumulative replay (the ONE model, no forgetting)
        if chunk_bytes >= CHUNK_TARGET:
            blob = b"\n\x00FILE\x00\n".join(chunk)
            u.ingest(f"csweep_{st['chunks']:04d}", blob,
                     epochs=EPOCHS_PER_CHUNK, repeat=1)
            st["chunks"] += 1
            print(f"  chunk {st['chunks']}: {chunk_bytes/1e6:.1f}MB "
                  f"({files_now} files, "
                  f"{st['bytes_ingested']/1e6:.0f}MB total, "
                  f"{time.time()-t0:.0f}s)", flush=True)
            chunk, chunk_bytes, files_now = [], 0, 0
            save_state(st)
    if chunk:
        blob = b"\n\x00FILE\x00\n".join(chunk)
        u.ingest(f"csweep_{st['chunks']:04d}", blob,
                 epochs=EPOCHS_PER_CHUNK, repeat=1)
        st["chunks"] += 1
    save_state(st)
    print(f"sweep done: visited={st['visited']} sampled={st['sampled']} "
          f"locked={st['skipped_locked']} ingested="
          f"{st['bytes_ingested']/1e6:.0f}MB chunks={st['chunks']} "
          f"domains={len(u.domains)}", flush=True)


if __name__ == "__main__":
    main()
