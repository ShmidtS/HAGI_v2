#!/usr/bin/env python3
"""Bridge: v4.4 env-growth learned world models -> v4.0 gate netlists.

A learned 5-bit temporal world model (32 states, 4 actions) compiles to
a compact proof-carrying GF(2) netlist (33 gates for a bank model),
behaviorally exact over all 128 transitions. Input encoding:
bits 0-4 = state s, bits 5-6 = action a; outputs 0-4 = next state.
"""
from __future__ import annotations
import random
import hagi_fusion_v4_0 as v40
from hagi import env_growth_v4_4 as env


def env_model_to_netlist(model_id: int, clock_action: int | None = None,
                         seed: int = 4242, context: str = "core",
                         goal: int = 5) -> dict:
    """Learn the world actively, then compile its table to a v4.0 netlist."""
    bank = env.generate_bank()
    w = env.make_world(seed, model_id, context, goal)
    agent = env.TemporalGrowthAgent()
    res = agent.run_task(w, random.Random(1))
    cand = env.candidate_records(res["clock_action"])[res["model_id"]]
    table = env.build_candidate_table(cand, res["clock_action"], bank)
    n = 7  # 5 state bits + 2 action bits
    rows: list = [None] * 128
    for s in range(32):
        for a in range(4):
            rows[s | (a << 5)] = tuple(((table[s][a] >> i) & 1)
                                       for i in range(5))
    anf = v40.mobius(rows, n)
    net = v40.ProofCarryingANFLearner.compile_optimal(n, anf)

    def ev(s, a):
        wv = [(s >> i) & 1 for i in range(5)] + [(a >> i) & 1
                                                 for i in range(2)]
        for g in net["gates"]:
            x, y, m = wv[g["a"]], wv[g["b"]], g["mask"]
            wv.append(0 if m == 0 else (1 if m == 15 else
                     (x ^ y if m == 6 else x & y)))
        return sum(wv[net["outs"][o]] << o for o in range(5))

    ok = all(ev(s, a) == table[s][a] for s in range(32) for a in range(4))
    if not ok:
        raise RuntimeError("bridge netlist behavioral mismatch")
    return {"net": net, "table": table, "model_id": res["model_id"],
            "learn_mode": res["mode"], "anf": anf,
            "gates": len(net["gates"]), "depth": net["metrics"]["depth"]}
