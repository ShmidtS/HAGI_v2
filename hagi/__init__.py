"""HAGI — self-developing neuro-symbolic processor.

Layout (WAVE 195 refactor: 55 -> 32 files, merged by meaning):
  Entry points
    hagi.cli        — unified CLI: python -m hagi.cli <cmd>
    hagi.pipeline   — THE pipeline: ingest -> LM -> diagnose ->
                      invent -> gate -> reify (HAGI_SYMBOLIC=1)
  Kernel & body
    hagi.core       — canonical kernel (legacy body star-imported)
    hagi.legacy_kernel — frozen legacy body
    hagi.lm         — byte-LM (CertifiedStepLM, RopeAttentionByteLM)
    hagi.unified    — ONE model: ingest + ledger
    hagi.gpu        — CUDA/ROCm backend
  Construct discovery
    hagi.invention  — blueprints + CEGIS + prior + while-loops + RDL
    hagi.representations — representation registry + reifier
    hagi.fabric     — Venn-cloth genome evolution
    hagi.skill_engine / anf_frontend — proof-carrying ANF synthesis
  Measurement
    hagi.measurement — transfer meter, capability, budget, oracle, QBC
    hagi.diagnosis  — residual profiles, hold classification
  Trunk ops
    hagi.trunk_ops  — six-phase cycle, merge, Hadamard, growth
  Theory bridges
    hagi.sandwich   — certified softmax argmax (C489-C502)
  Environments
    hagi.env_lut / env_temporal — worlds + bridges + verifiers
  Self development
    hagi.selfmodel  — self-model + controller + experiment + selfedit
    hagi.loop / rewrite — growth loops
    hagi.genesis    — architecture genesis + EC library
    hagi.school     — lessons, curriculum, mind
  Misc
    hagi.certificates — portable certificate suite
    hagi.cli        — see Entry points

`from hagi import *` exposes the core namespace.
"""
from .core import *  # noqa: F401,F403
from . import genesis, logic  # noqa: F401
