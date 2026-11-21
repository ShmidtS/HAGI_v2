"""HAGI — hierarchical growing model.

Consolidated layout (2026-11-17 refactor):
  hagi.core      — canonical kernel
  hagi.legacy_kernel — frozen V5-V8 body (star-imported by core)
  hagi.genesis   — Architecture Genesis: loop-config evolution (part 1)
                   and EC-processor genome evolution (part 2)
  hagi.logic     — logic basis: Venn head + ANF/Zhegalkin
  hagi.lm        — byte-LM neural body (ByteLM, CertifiedStepLM, ...)
  hagi.lessons   — Nadya's school (cumulative curriculum, from-scratch)
  hagi.school    — lessons + curriculum + mind
  hagi.loop      — certified self-development loop
  hagi.selfedit  — certified knob self-editing
  hagi.rewrite   — forever driver (+ watchdog in watchdog.pyw)
  hagi.mdl       — MDL-prune of the expert library
  hagi.selfmodel / hagi.experiment / hagi.controller — self-model,
                   experiments, v10 controller
  hagi.ingest    — learn from arbitrary data files
  hagi.execute   — run the programs the model wrote
  
`from hagi import *` exposes the core namespace (same as before the
refactor).
"""
from .core import *  # noqa: F401,F403
from . import genesis, logic  # noqa: F401
