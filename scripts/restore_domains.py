"""Restore unified model by replaying ALL saved domain chunks
(412 ns_*.bin) through ingest — same path as the original cycle.
Run with 1-thread OpenBLAS env (see watchdog pattern)."""
import sys, time, os
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hagi.unified import Unified


def main() -> None:
    dom_dir = ROOT / "checkpoints" / "domains"
    chunks = sorted(dom_dir.glob("ns_*.bin"))
    print(f"restore: {len(chunks)} domain chunks", flush=True)
    u = Unified.load() if Unified.exists() else Unified()
    # WAVE 216 (user: учи сразу тернарную): QAT on — forward sees
    # {-s,0,+s} throughout training; payload export round-trips
    # without post-training surprise.
    u.model._qat = True
    print("QAT: on (ternary straight-through)", flush=True)
    t0 = time.perf_counter()
    # WAVE 178: resume — a chunk whose sha is already in the ledger
    # with a trained CE is already learned; skip it (idempotent
    # restart: the model in memory is the accumulator, not the log).
    from hagi.unified import _sha
    skipped = 0
    for i, p in enumerate(chunks, 1):
        blob = p.read_bytes()
        if len(blob) < 16:
            continue
        rec = u.domains.get(p.stem)
        if rec and rec.get("sha") == _sha(blob) and rec.get("ce") is not None:
            skipped += 1
            print(f"[restore {i}/{len(chunks)}] {p.stem} -> SKIP "
                  "(already learned, sha match)", flush=True)
            continue
        r = u.ingest(p.stem, blob, epochs=2, repeat=1, replay_cap=None)
        # WAVE 218 (user: модель наследует семантику embeddinggemma):
        # после ingest — semantic distill на фрагментах домена.
        if os.environ.get("HAGI_SEMANTIC") == "1":
            try:
                from hagi.lm import SemanticDistillHead
                bank = getattr(u, "sem_head", None)
                if bank is None:
                    bank = u.sem_head = SemanticDistillHead(dim=0)
                import random as _rd
                blob_u = blob.decode("utf-8", errors="ignore")
                frags = [blob_u[i:i+200] for i in
                         range(0, min(len(blob_u), 2000), 200)][:8]
                if frags:
                    idx = torch.tensor(
                        [[(ord(c) % 256) for c in f[:u.model.k]]
                         for f in frags], dtype=torch.long)
                    if idx.shape[1] < u.model.k:
                        pad = torch.full(
                            (idx.shape[0], u.model.k - idx.shape[1]), 32,
                            dtype=torch.long)
                        idx = torch.cat([pad, idx], dim=1)
                    u.model.domain_adapter = None
                    l = bank.align_step(u.model, idx, frags)
                    print(f"    semantic loss {l:.3f}", flush=True)
            except Exception as _e:
                print(f"    semantic skipped: {_e!r}", flush=True)
        ce = r.get("holdout_ce", -1)
        # WAVE 205 (VRAM hygiene): the replay corpus grows with every
        # domain (full-book replay) and allocator caching fragments
        # VRAM over a long run — measured as chunk time 700s -> 980s
        # and HIP launch failures after ~1h on the iGPU. Free the
        # cached blocks and log VRAM so growth is visible.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                mem = torch.cuda.memory_allocated() / 2**20
                resv = torch.cuda.memory_reserved() / 2**20
                print(f"    vram: alloc={mem:.0f}MB reserved={resv:.0f}MB",
                      flush=True)
        except Exception:
            pass
        import gc
        gc.collect()
        print(f"[restore {i}/{len(chunks)}] {p.stem} -> CE {ce} "
              f"({r.get('seconds', '?')}s)", flush=True)
        if i % 8 == 0:
            u.save()  # periodic checkpoint: resume boundary advances
    print(f"restore done in {(time.perf_counter()-t0)/3600:.1f}h "
          f"(skipped {skipped})", flush=True)


if __name__ == "__main__":
    main()
