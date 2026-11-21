#!/usr/bin/env python3
"""Shared env-world -> GF(2) netlist compiler (used by env bridges).

Compiles a transition table (n_bits state, n_act_bits action) to a
compact proof-carrying v4.0 netlist, behaviorally exact over all
transitions. Rule of Three refactor: the table-compiler was duplicated
in both env bridges.
"""
from __future__ import annotations
from hagi import skill_engine as v40


def compile_transition_table(table, n_bits: int, n_act_bits: int) -> dict:
    """table[s][a] -> next state (integer). Returns dict with net/anf/
    gates/depth and behavioral_exact flag (checked over ALL pairs)."""
    n = n_bits + n_act_bits
    n_states, n_actions = 1 << n_bits, 1 << n_act_bits
    rows: list = [None] * (n_states * n_actions)
    for s in range(n_states):
        for a in range(n_actions):
            ns = table[s][a]
            rows[s | (a << n_bits)] = tuple(((ns >> i) & 1)
                                            for i in range(n_bits))
    anf = v40.mobius(rows, n)
    net = v40.ProofCarryingANFLearner.compile_optimal(n, anf)

    def ev(s, a):
        wv = [(s >> i) & 1 for i in range(n_bits)] + \
             [(a >> i) & 1 for i in range(n_act_bits)]
        for g in net["gates"]:
            x, y, m = wv[g["a"]], wv[g["b"]], g["mask"]
            wv.append(0 if m == 0 else (1 if m == 15 else
                     (x ^ y if m == 6 else x & y)))
        return sum(wv[net["outs"][o]] << o for o in range(n_bits))

    ok = all(ev(s, a) == table[s][a]
             for s in range(n_states) for a in range(n_actions))
    if not ok:
        raise RuntimeError("bridge netlist behavioral mismatch")
    return {"net": net, "table": table, "anf": anf,
            "gates": len(net["gates"]), "depth": net["metrics"]["depth"],
            "behavioral_exact": True}
