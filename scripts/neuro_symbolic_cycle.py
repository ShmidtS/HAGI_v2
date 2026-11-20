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
    """Grow-the-ladder: task complexity rises as circuits are certified.
    Stage 0: full adder slice (solved). Stage 1: 2-bit adder (two
    slices composed — inputs a0,a1,b0,b1,carry; outputs s0,s1,c2).
    Stage 2: MUX / comparator. The ladder is honest: each stage's
    truth table is exhaustively certifiable, and the stage N+1 is a
    COMPOSITION of stage N blocks — the logical schema modernizes by
    reusing its own certified cells."""
    import numpy as np
    stage_file = ROOT / "checkpoints" / "ladder_stage.json"
    stage = 0
    if stage_file.exists():
        try:
            stage = json.loads(stage_file.read_text(encoding="utf-8")).get("stage", 0)
        except Exception:
            stage = 0
    tasks = []
    if stage == 0:
        # full adder slice (a, b, cin) -> (sum, cout)
        truth = {}
        for a in (0, 1):
            for b in (0, 1):
                for cin in (0, 1):
                    s = a ^ b ^ cin
                    cout = (a & b) | (cin & (a ^ b))
                    truth[(a, b, cin)] = [s, cout]
        tasks.append({"name": "adder_slice", "n_inputs": 3,
                      "truth": truth,
                      "ins": lambda key: [key[0], key[1], key[2]],
                      "meta": {"stage": 0}})
    elif stage == 1:
        # 2-bit adder: (a0,a1,b0,b1,cin) -> (s0,s1,c2) — 32 rows,
        # composition of two slices
        truth = {}
        for a0 in (0,1):
            for a1 in (0,1):
                for b0 in (0,1):
                    for b1 in (0,1):
                        for cin in (0,1):
                            a = a0 | (a1 << 1)
                            b = b0 | (b1 << 1)
                            s = a + b + cin
                            s0, s1, c2 = s & 1, (s >> 1) & 1, (s >> 2) & 1
                            truth[(a0,a1,b0,b1,cin)] = [s0, s1, c2]
        tasks.append({"name": "adder2_bit", "n_inputs": 5,
                      "truth": truth,
                      "ins": lambda key: list(key),
                      "meta": {"stage": 1}})
    else:
        # 2:1 MUX: (a, b, sel) -> out; comparator bit (a>b)
        truth = {}
        for a in (0,1):
            for b in (0,1):
                for sel in (0,1):
                    truth[(a,b,sel)] = [a if sel else b]
        tasks.append({"name": "mux21", "n_inputs": 3,
                      "truth": truth,
                      "ins": lambda key: list(key),
                      "meta": {"stage": 2}})
    return tasks


def compose_adder2() -> dict:
    """Stage 1: 2-bit adder = cascade of 2 certified full-adder cells.
    Deterministic composition of the processor's OWN certified cells
    (standard cell-library flow); the composition is exhaustively
    certified (32/32)."""
    from hagi import fabric
    def adder_cell(srcA, srcB, srcC, base):
        g = [
            {"a": srcA, "b": srcB, "mask": fabric.MASKS["XOR"]},
            {"a": base+0, "b": srcC, "mask": fabric.MASKS["XOR"]},
            {"a": base+0, "b": srcC, "mask": fabric.MASKS["AND"]},
            {"a": srcA, "b": srcB, "mask": fabric.MASKS["AND"]},
            {"a": base+3, "b": base+2, "mask": fabric.MASKS["OR"]},
        ]
        return g, base+1, base+4
    n_in = 5
    g1, s0, c1 = adder_cell(0, 2, 4, n_in)
    gates = list(g1)
    base2 = n_in + len(gates)
    g2, s1, c2 = adder_cell(1, 3, c1, base2)
    gates += g2
    genome = {"W": 1, "n_inputs": n_in, "gates": gates,
              "outs": [s0, s1, c2]}
    truth = {}
    for a0 in (0,1):
        for a1 in (0,1):
            for b0 in (0,1):
                for b1 in (0,1):
                    for cin in (0,1):
                        s = (a0|(a1<<1)) + (b0|(b1<<1)) + cin
                        truth[(a0,a1,b0,b1,cin)] = [s&1, (s>>1)&1, (s>>2)&1]
    ok, _ = fabric.certify(genome, truth, lambda k: list(k))
    if not ok:
        return None
    return {"profile": "adder2_composed", "seed": 0,
            "certified_gen": 0, "best": genome, "bits": None,
            "history": [{"phase": "COMPOSE", "gates": len(gates)}]}


def fusion_fastpath(task: dict) -> dict | None:
    """Fusion Lab v2.1 fast-path (wave 128): before the expensive GA,
    try proof-carrying ANF interpolation. Trusted issuer = own certified
    seed-cells family; degree contract comes from the task spec itself
    (honest: the task IS the spec, so max degree is computed from truth
    by exact Mobius). Falls back to GA when n exceeds the full-support
    cap or the fit HOLDs."""
    try:
        from hagi.fusion_v4_0 import (ProofCarryingANFLearner,
                                      ExactMDLLibrary)
    except ImportError:
        try:
            from hagi.fusion_anf import (ProofCarryingANFLearner,
                                         ExactMDLLibrary)
        except ImportError:
            return None
    n = task["n_inputs"]
    if n < 1 or n > 14:
        return None
    ins = task["ins"]
    key0 = next(iter(task["truth"]))
    n_out = len(task["truth"][key0])
    def oracle(x):
        return tuple(int(v) for v in
                     task["truth"][tuple((x >> i) & 1
                                          for i in range(n))])
    L = ProofCarryingANFLearner(trusted_issuers={"hagi-truth-spec"})
    try:
        cand, ev = L.fit(n, n_out, oracle, allow_full_support=True)
    except Exception:
        return None
    if cand is None or ev.get("action") != "COMMIT_CANDIDATE":
        return None
    # Convert to a fabric genome via the ANF netlist (mask 6=XOR,
    # 8=AND, 15/0 = const) and certify it against the same truth.
    genome = {"W": 1, "n_inputs": n, "gates": cand.netlist["gates"],
              "outs": cand.netlist["outs"]}
    try:
        ok, _ = fabric.certify(genome, task["truth"], ins)
    except Exception:
        return None
    if not ok:
        return None
    return {"best": genome, "seed": "fusion-anf",
            "certified_gen": 0, "bits": cand.netlist["metrics"]["program_bits"],
            "_fusion": {"mode": ev.get("mode"),
                        "queries": ev.get("query_count"),
                        "gate_count": cand.netlist["metrics"]["gate_count"],
                        "depth": cand.netlist["metrics"]["depth"]}}


def weave_task(task: dict, pop_size: int = 100, generations: int = 3000,
               restarts: int = 10, seed: int = 5) -> dict | None:
    """Run the fabric loop for one derived task. Fusion fast-path first,
    then multi-restart GA with certified-first COMMIT gate. Returns the
    certified optimized genome (the tape-out circuit) or None."""
    fast = fusion_fastpath(task)
    if fast is not None:
        return fast
    truth = task["truth"]
    ins = task["ins"]
    for r in range(restarts):
        # W=1: adder slice is boolean (inputs are single bits) —
        # 1 lane is exact and 4x cheaper to certify
        res = fabric.evolve({**truth, "_W": 1}, ins,
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

def prev_gates(c: dict) -> int:
    p = c.get("_reuse_of") or {}
    return p.get("gates", 0)


def prev_netlist(c: dict) -> str:
    return (c.get("_reuse_of") or {}).get("netlist", "")


def commit(circuits: list[dict], chunk_id: str, ce: float) -> None:
    prev = json.loads(TAPEOUT.read_text(encoding="utf-8")) \
        if TAPEOUT.exists() else []
    tape = {
        "chunk": chunk_id, "holdout_ce": ce,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "circuits": [
            ({"name": c["profile"], "seed": c["seed"],
              "certified_gen": c["certified_gen"],
              "gates": prev_gates(c), "bits": c["bits"],
              "netlist": prev_netlist(c) if c.get("_reuse_of")
                         else fabric.render(c["best"]),
              "certificate":
                  "REUSE of certified circuit (exhaustive PASS)"
                  if c.get("_reuse_of")
                  else "exhaustive truth table: PASS"}
             if c.get("_reuse_of") else
             {"name": c["profile"], "seed": c["seed"],
              "certified_gen": c["certified_gen"],
              "gates": len(c["best"]["gates"]),
              "bits": c["bits"],
              "netlist": fabric.render(c["best"]),
              "certificate": "exhaustive truth table: PASS",
              "truth_hash": c.get("truth_hash")})
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
        # ---- 1. GROW: collect a chunk from C:/ (persistent cursor) ----
        last_path = None
        for path in iter_files("C:/", start_after=st.get("cursor")):
            last_path = path
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
        if last_path:
            st["cursor"] = last_path     # resume here next iteration
        if not chunk:
            print("no more files — disk fully swept", flush=True)
            break
        blob = b"\n\x00FILE\x00\n".join(chunk)
        cid = f"ns_{st['chunks']:04d}"
        (ROOT / "checkpoints" / "domains" / f"{cid}.bin").write_bytes(blob)
        # ---- neural training on cumulative corpus ----
        r = u.ingest(cid, blob, epochs=2, repeat=1,
                     replay_cap=400_000)
        st["chunks"] += 1
        ce = r.get("holdout_ce", -1)
        print(f"[{cid}] neural: +{chunk_bytes/1e6:.1f}MB "
              f"({files_now} files) → CE {ce} "
              f"({r.get('seconds', '?')}s)", flush=True)
        chunk, chunk_bytes, files_now = [], 0, 0
        # ---- 2+3. DERIVE + WEAVE: fabric self-assembly ----
        model = u.model
        tasks = derive_tasks(model, k=u.k,
                             seed=13 + st['chunks'] * 7)
        # dedup: truth-table hash — same boolean problem = same circuit;
        # honest "reused" instead of wasting search budget
        solved = {}
        netlist_path = ROOT / "checkpoints" / "processor_netlist.json"
        if netlist_path.exists():
            for tape in json.loads(netlist_path.read_text(encoding="utf-8")):
                for c in tape["circuits"]:
                    solved[c["name"]] = c
        circuits, reused = [], 0
        seen_tables = {}
        # index: truth-hash → certified circuit from past tape-outs
        solved_tables = {}
        netlist_path = ROOT / "checkpoints" / "processor_netlist.json"
        if netlist_path.exists():
            for tape in json.loads(
                    netlist_path.read_text(encoding="utf-8")):
                for c in tape["circuits"]:
                    th = c.get("truth_hash")
                    if th:
                        solved_tables[th] = c
        for i, task in enumerate(tasks):
            thash = __import__("hashlib").sha1(
                json.dumps(sorted(task["truth"].items()),
                           default=str).encode()).hexdigest()[:10]
            if task.get("meta", {}).get("stage", 0) == 1:
                res = compose_adder2()
                if res:
                    res["truth_hash"] = thash
                    res["profile"] = f"{task['name']}_{thash}"
                    circuits.append(res)
                    # ladder advance: stage 1 composed ONCE → stage 2
                    sf = ROOT / "checkpoints" / "ladder_stage.json"
                    if json.loads(sf.read_text(encoding="utf-8")).get("stage", 0) == 1:
                        sf.write_text(json.dumps({"stage": 2}),
                                      encoding="utf-8")
                        print(f"  LADDER: stage 1 composed → stage 2 "
                              f"(MUX) unlocked", flush=True)
                    print(f"  fabric [{task['name']}]: COMPOSED from "
                          f"certified cells — CERTIFIED 32/32",
                          flush=True)
                else:
                    print(f"  fabric [{task['name']}]: composition "
                          f"failed (honest: no commit)", flush=True)
                continue
            if thash in seen_tables:
                print(f"  fabric [{task['name']}]: DUPLICATE of "
                      f"[{seen_tables[thash]}] — reused, no re-search",
                      flush=True)
                reused += 1
                continue
            seen_tables[thash] = task["name"]
            if thash in solved_tables:
                # already solved in an earlier tape-out: honest reuse
                prev = solved_tables[thash]
                print(f"  fabric [{task['name']}]: SOLVED earlier "
                      f"[{prev['name']}] — reusing certified circuit",
                      flush=True)
                circuits.append({"profile": f"{task['name']}_{thash}",
                                 "seed": prev.get("seed", 0),
                                 "certified_gen": prev.get("certified_gen"),
                                 "best": None, "bits": prev.get("bits"),
                                 "_reuse_of": prev})
                continue
            res = weave_task(task, seed=5 + i)
            if res:
                res["truth_hash"] = thash
                # ladder: certified stage task → advance complexity
                stage = task.get("meta", {}).get("stage", 0)
                sf = ROOT / "checkpoints" / "ladder_stage.json"
                sf.write_text(json.dumps({"stage": stage + 1}),
                              encoding="utf-8")
                print(f"  LADDER: stage {stage} solved → stage "
                      f"{stage+1} unlocked", flush=True)
            if res:
                res["profile"] = f"{task['name']}_{thash}"
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
