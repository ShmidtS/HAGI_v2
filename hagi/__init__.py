"""HAGI — hierarchical growing model (package split of the monolith).

Import order preserves the historical module-level override semantics
of the single-file build: legacy V5–V8 first, then the V9 canonical
core (the pipeline the self-model derives its algorithm from).
"""
from ._legacy import *  # noqa: F401,F403
from ._v9 import *  # noqa: F401,F403
