#!/usr/bin/env python3
"""Bridge: v4.6 env-growth world models -> v4.0 gate netlists.

4-bit world (16 states, 4 actions). PROVEN_EXACT worlds compile to
compact proof-carrying GF(2) netlists, behaviorally exact over all 64
transitions. OOD worlds (GRAMMAR_MISMATCH) go through full-table
fallback; the LUT3 rule is inferred after compilation.
Encoding: bits 0-3 = state s, bits 4-5 = action a; outputs 0-3 = next
state.
"""
from __future__ import annotations
import hagi_fusion_v4_0 as v40
from hagi import env_growth_v4_6 as env


def _compile_table(table: dict, n_bits: int = 4, n_act_bits: int = 2) -> dict:
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
            "behavioral_exact": ok}


def env_model_to_netlist(seed: int = 777, ood: bool = False) -> dict:
    """Actively learn a v4.6 world, audit/fallback it, compile to v4.0."""
    w = env.make_world(seed, ood=ood)
    res = env.learn_active(w)
    out = {"learn_status": res["status"], "probes": res["queries"],
           "clock_action": res["clock_action"], "ood": ood}
    if res["status"] == "PROVISIONAL_SAMPLED":
        audit = env.audit_candidate_full(w, res["clock_action"],
                                         res["model"])
        table, audit_inter, _ = env.table_from_full_audit(
            audit, res["clock_action"])
        out["audit_interactions"] = audit_inter
        out["proven"] = bool(audit.get("complete"))
    else:  # GRAMMAR_MISMATCH / out-of-family
        table, inter2, _ = env.full_table_fallback(w, res["clock_action"])
        out["audit_interactions"] = inter2
        out["proven"] = False
        rule = env.infer_deep_ood_rule(table, res["clock_action"])
        out["inferred_ood_rule"] = rule
    out.update(_compile_table(table))
    return out
