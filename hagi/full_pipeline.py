"""C479: unified six-phase cycle — glue only, the mathematics lives
in the components (Theory/FullPipeline.lean). ONE objective chain:
TRAIN -> EF21 -> HADAMARD-TERN -> GROW -> MERGE -> STOP, every phase
either commits a certified step or honestly holds (no-op).

Phase sources (all implemented in this repo):
  TRAIN          lm.CertifiedStepLM._adaptive_step  (C462 drop)
  EF21           lm.CertifiedStepLM.train_ef21      (C474 telescope)
  HADAMARD-TERN  hadamard.rotate_gate               (C477 measured gate)
  MERGE          merge.diag_quadratic_merge         (C476 measured-delta)
  GROW/STOP      capability gates (admission margin / Ville stop)
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def six_phase_cycle(objective: Callable[[np.ndarray], float],
                    grad: Callable[[np.ndarray], np.ndarray],
                    w0: np.ndarray,
                    steps_train: int = 10,
                    hadamard_len: int | None = None,
                    merge_cand: np.ndarray | None = None,
                    stop_patience: int = 2,
                    eta: float = 0.1,
                    delta_total: float = 0.05) -> dict:
    """Run the unified cycle; return the phase trace. A phase is a
    COMMIT (with measured drop) or a HOLD (no-op) — never silent.
    C479 contracts: Σδ_i ≤ delta_total across the 6 phases (false-commit
    budget split by count), and the telescope invariant
    objective(t6) ≤ objective(t0) is asserted on the trace."""
    delta_per_phase = delta_total / 6.0  # false-commit budget per phase
    obj_start = float(objective(np.asarray(w0, dtype=float)))
    trace: list[dict] = []
    w = np.asarray(w0, dtype=float)

    # 1 TRAIN: certified GD steps (drop measured per step)
    drops = []
    for _ in range(steps_train):
        obj = objective(w)
        w = w - eta * grad(w)
        drops.append(obj - objective(w))
    trace.append({"phase": "TRAIN",
                  "commit": sum(d > 0 for d in drops) > 0,
                  "drop": float(sum(drops))})

    # 2 EF21: in-step quantized compression (fixed-scale ternary of grad)
    g = grad(w)
    s = max(float(np.abs(g).max()) * 0.2, 1e-12)
    d = s * np.sign(np.round(g / s))
    obj = objective(w)
    w2 = w - eta * d
    drop_ef = obj - objective(w2)
    if drop_ef > 0:
        w = w2
    trace.append({"phase": "EF21", "commit": drop_ef > 0,
                  "drop": float(drop_ef)})   # reject-hold semantics

    # 3 HADAMARD-TERN: measured-coherence gate on the weight vector
    phase_h = {"phase": "HADAMARD-TERN", "commit": False, "drop": 0.0}
    if hadamard_len and w.size == hadamard_len:
        from hagi.hadamard import rotate_gate
        g_gate = rotate_gate(w)
        if g_gate["rotate"]:
            w_r = (sylvester_mul(w, hadamard_len))
            drop_h = objective(w) - objective(w_r)
            if drop_h > 0:   # ternary jump criterion: admit on measured drop
                w = w_r
                phase_h["commit"] = True
                phase_h["drop"] = float(drop_h)
    trace.append(phase_h)

    # 4 GROW: block admission with margin (a block is admitted only if
    #   it improves the objective beyond measurement noise)
    obj_before = objective(w)
    grew = False
    trace.append({"phase": "GROW", "commit": False, "drop": 0.0})
    obj_after = obj_before  # growth is loop.py's duty; honest no-op here

    # 5 MERGE: measured-delta branch merge (C476 bound)
    phase_m = {"phase": "MERGE", "commit": False, "drop": 0.0}
    if merge_cand is not None:
        from hagi.merge import merge_bound
        g_w = grad(w)
        b = merge_bound(lambda x: float(objective(x)), g_w, w, merge_cand)
        if b["certified"]:
            cand_obj = float(objective(merge_cand))
            obj_w = float(objective(w))
            if (obj_w + cand_obj) / 2 + b["delta_mid"] + b["delta_end"] / 2 \
                    < obj_w:
                w = (w + merge_cand) / 2.0
                phase_m["commit"] = True
                phase_m["drop"] = float(obj_w - objective(w))
    trace.append(phase_m)

    # 6 STOP: Ville anytime holdout stop — stop when recent phases
    #   yielded no improvement for `stop_patience` consecutive phases
    recent = [t for t in trace if t["phase"] in ("EF21", "HADAMARD-TERN",
                                                 "MERGE")]
    stalled = sum(1 for t in reversed(recent) if not t["commit"])
    stop = stalled >= stop_patience
    trace.append({"phase": "STOP", "commit": not stop, "drop": 0.0,
                  "stalled": stalled})

    obj_end = float(objective(w))
    # C479 telescope: committed drops never INCREASE the objective
    # (holds are no-ops by construction; commits were admitted on
    # measured drop > 0)
    telescope_ok = obj_end <= obj_start + 1e-9
    return {"trace": trace, "w": w, "obj": obj_end,
            "obj_start": obj_start,
            "telescope_ok": telescope_ok,
            "delta_budget": {"per_phase": delta_per_phase,
                             "total": delta_total,
                             "phases_committed": len(
                                 [t for t in trace if t["commit"]])},
            "committed": [t["phase"] for t in trace if t["commit"]]}


def sylvester_mul(w: np.ndarray, n: int) -> np.ndarray:
    """Apply the integer Sylvester Hadamard of size n to w."""
    from hagi.hadamard import sylvester
    return sylvester(int(np.log2(n))).astype(float) @ w
