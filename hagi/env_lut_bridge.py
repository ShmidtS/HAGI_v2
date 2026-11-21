#!/usr/bin/env python3
"""Bridge: env-growth world models -> gate netlists.

4-bit world (16 states, 4 actions). PROVEN_EXACT worlds compile to
compact proof-carrying GF(2) netlists, behaviorally exact over all 64
transitions. OOD worlds (GRAMMAR_MISMATCH) go through full-table
fallback; the LUT3 rule is inferred after compilation.
Encoding: bits 0-3 = state s, bits 4-5 = action a; outputs 0-3 = next
state.
"""
from __future__ import annotations
from hagi import env_lut as env
from hagi.env_table_compiler import compile_transition_table


def env_model_to_netlist(seed: int = 777, ood: bool = False) -> dict:
    """Actively learn a world, audit/fallback it, compile to netlist."""
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
    out.update(compile_transition_table(table, n_bits=4, n_act_bits=2))
    return out
