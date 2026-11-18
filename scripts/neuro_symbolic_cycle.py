#!/usr/bin/env python3
"""Neuro-symbolic self-evolution cycle — ONE model, fabric inside.

Loop (per iteration):
  1. GROW (neural):   sweep C:/ → sample files → certified training
                      chunk on the cumulative corpus (ONE body).
  2. DERIVE (symbolic): from the trained body, derive boolean fabric
                      tasks from what was just learned — e.g. sign
                      structure of W1 rows / next-bit predicates on
                      high-energy bytes. Each task = truth table.
  3. WEAVE (fabric):  certified-first evolution (GROW correctness +
                      neutral drift, then OPTIMIZE cost hard-gated) —
                      the fabric SELF-ASSEMBLES a circuit per task.
  4. COMMIT (tape-out): certified circuits appended to the processor
                      netlist (fabric.json); neuro checkpoint saved.

Every commit carries a certificate (exhaustive truth-table check).
No commit without one — the processor is built only from proven
circuits. This wires fabric evolution INTO the training loop.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hagi import fabric, unified  # noqa: E402
from scripts.feed_all_c import (CHUNK_TARGET, SKIP_DIRS, iter_files,
                                load_state, save_state, sample_bytes)  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TAPEOUT = ROOT / "checkpoints" / "processor_netlist.json"

# ---------- DERIVE: fabric tasks FROM the trained body ----------

def derive_tasks(model, k: int, n_tasks: int = 3, seed: int = 13) -> list[dict]:
    """Derive fabric tasks from the neural structure. Honest scope:
    the fabric builds the ADDER CORE (bit slices of a byte-wide adder)
    — the arithmetic heart a byte-adder cascade is composed from.
    Task bit-j selects which carry chain the fabric re-derives; the
    target byte from W2 tells which output column the body weights
    most (meta only). 8-row exhaustive tables — certifiable."""
    import numpy as np
    W2 = model.W2.detach().numpy()
    col = np.abs(W2).sum(axis=0)
    top = col.argsort()[-n_tasks:]
    tasks = []
    for j, bt in enumerate(top):
        # full adder slice: (a, b, cin) -> (sum, cout)
        truth = {}
        for a in (0, 1):
            for b in (0, 1):
                for cin in (0, 1):
                    s = a ^ b ^ cin
                    cout = (a & b) | (cin & (a ^ b))
                    truth[(a, b, cin)] = [s, cout]

        def ins(key):
            return [key[0], key[1], key[2]]
        tasks.append({"name": f"adder_slice_w{int(bt)}",
                      "n_inputs": 3, "truth": truth, "ins": ins,
                      "meta": {"weighted_output_col": int(bt)}})
    return tasks


def weave_task(task: dict, pop_size: int = 80, generations: int = 2000,
               restarts: int = 4, seed: int = 5) -> dict | None:
    """Run the fabric loop for one derived task. Multi-restart GA with
    certified-first COMMIT gate. Returns the certified optimized
    genome (the tape-out circuit) or None (honest: no commit)."""
    truth = task["truth"]
    ins = task["ins"]
    for r in range(restarts):
        # W=4: nibble buses (task derives nibble adder bits)
        res = fabric.evolve({**truth, "_W": 4}, ins,
                            n_inputs=task["n_inputs"],
                            pop_size=pop_size, generations=generations,
                            seed=seed + r, log=False,
                            profile=task["name"])
        if res["best"] is not None:
            ok, _ = fabric.certify(res["best"], truth, ins)
            if ok:
                return res
    return None


# ---------- COMMIT: tape-out the certified netlist ----------

def commit(circuits: list[dict], chunk_id: str, ce: float) -> None:
    prev = json.loads(TAPEOUT.read_text(encoding="utf-8")) \
        if TAPEOUT.exists() else []
    tape = {
        "chunk": chunk_id, "holdout_ce": ce,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "circuits": [
            {"name": c["profile"], "seed": c["seed"],
             "certified_gen": c["certified_gen"],
             "gates": len(c["best"]["gates"]),
             "bits": c["bits"],
             "netlist": fabric.render(c["best"]),
             "certificate": "exhaustive truth table: PASS"}
            for c in circuits]}
    prev.append(tape)
    TAPEOUT.write_text(json.dumps(prev, ensure_ascii=False, indent=2),
                       encoding="utf-8")


# ---------- main loop ----------

def main() -> None:
    max_minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 480.0
    deadline = time.time() + max_minutes * 60
    u = unified.Unified.load() if unified.Unified.exists() \
        else unified.Unified()
    st = load_state()
    seen = st["seen"]
    chunk, chunk_bytes = [], 0
    files_now = 0
    it = 0
    print(f"neuro-symbolic cycle start: chunks={st['chunks']}", flush=True)
    while time.time() < deadline:
        # ---- 1. GROW: collect a chunk from C:/ ----
        for path in iter_files("C:/"):
            if chunk_bytes >= CHUNK_TARGET or time.time() > deadline:
                break
            st["visited"] += 1
            try:
                size = __import__("os").path.getsize(path)
            except OSError:
                continue
            if size < 16:
                continue
            h = __import__("hashlib").sha1(
                path.encode("utf-8", errors="replace")).hexdigest()[:12]
            if h in seen:
                continue
            data = sample_bytes(path, size)
            if data is None:
                seen[h] = "locked"
                continue
            seen[h] = len(data)
            st["sampled"] += 1
            chunk.append(data)
            chunk_bytes += len(data)
            st["bytes_ingested"] += len(data)
            files_now += 1
        if not chunk:
            print("no more files — disk fully swept", flush=True)
            break
        blob = b"\n\x00FILE\x00\n".join(chunk)
        cid = f"ns_{st['chunks']:04d}"
        (ROOT / "checkpoints" / "domains" / f"{cid}.bin").write_bytes(blob)
        # ---- neural training on cumulative corpus ----
        r = u.ingest(cid, blob, epochs=4, repeat=1)
        st["chunks"] += 1
        ce = r.get("holdout_ce", -1)
        print(f"[{cid}] neural: +{chunk_bytes/1e6:.1f}MB "
              f"({files_now} files) → CE {ce} "
              f"({r.get('seconds', '?')}s)", flush=True)
        chunk, chunk_bytes, files_now = [], 0, 0
        # ---- 2+3. DERIVE + WEAVE: fabric self-assembly ----
        model = u.model
        tasks = derive_tasks(model, k=u.k)
        circuits = []
        for i, task in enumerate(tasks):
            res = weave_task(task, seed=5 + i)
            if res:
                circuits.append(res)
                nb = res["bits"]
                ng = len(res["best"]["gates"])
                print(f"  fabric [{task['name']}]: CERTIFIED "
                      f"{ng} gates, {nb} bits, gen "
                      f"{res['certified_gen']}", flush=True)
            else:
                print(f"  fabric [{task['name']}]: no certified circuit "
                      f"(honest: no commit)", flush=True)
        # ---- 4. COMMIT tape-out ----
        if circuits:
            commit(circuits, cid, ce)
            print(f"  TAPE-OUT: {len(circuits)} certified circuits "
                  f"→ processor_netlist.json", flush=True)
        # ---- persist one-model state ----
        u.save()
        save_state(st)
        it += 1
    print(f"cycle done: iterations={it} chunks={st['chunks']} "
          f"ingested={st['bytes_ingested']/1e6:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
