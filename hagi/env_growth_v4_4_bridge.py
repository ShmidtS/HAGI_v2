#!/usr/bin/env python3
"""Bridge: v4.4 env-growth learned world models -> v4.0 gate netlists.

A learned 5-bit temporal world model (32 states, 4 actions) compiles to
a compact proof-carrying GF(2) netlist (33 gates for a bank model),
behaviorally exact over all 128 transitions. Input encoding:
bits 0-4 = state s, bits 5-6 = action a; outputs 0-4 = next state.
"""
from __future__ import annotations
import random
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
    from hagi.env_table_compiler import compile_transition_table
    compiled = compile_transition_table(table, n_bits=5, n_act_bits=2)
    net = compiled["net"]
    return {"net": net, "table": table, "model_id": res["model_id"],
            "learn_mode": res["mode"], "anf": compiled["anf"],
            "gates": compiled["gates"], "depth": compiled["depth"],
            "behavioral_exact": compiled["behavioral_exact"]}
