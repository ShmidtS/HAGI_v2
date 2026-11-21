"""Unified pipeline (WAVE 185): neural + symbolic in one flow."""
import os
import sys
import tempfile

import numpy as np
import pytest



def _fresh_unified():
    os.environ["HAGI_CKPT_DIR"] = tempfile.mkdtemp()
    import importlib
    import hagi.util as _util
    importlib.reload(_util)   # SSOT first: CKPT_DIR from the env
    import hagi.unified as U
    importlib.reload(U)       # then consumers pick the sandbox up
    import hagi.pipeline as _pipe
    importlib.reload(_pipe)
    return U.Unified(k=4, dim=32, hidden=64)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.setenv("HAGI_SYMBOLIC", "1")


def test_counter_domain_full_flow():
    from hagi.diagnosis import HoldReason
    from hagi.pipeline import unified_ingest
    u = _fresh_unified()
    seq = bytes([(3 * i) % 16 for i in range(6000)])
    r = unified_ingest(u, "counter", seq, epochs=1, repeat=1)
    assert r["holdout_ce"] < 2.0
    sym = r["symbolic"]
    assert sym is not None
    # WAVE 196: the symbolic phase now works on the LM's OWN
    # residuals. On a well-learned counter the residual set is tiny
    # or empty — the honest outcomes are either an invention with
    # the exact gate, or an honest skip; both are correct.
    if sym["invented"]:
        assert sym["invented"]["status"] in ("INVENTED", "PROVEN_EXACT",
                                             "HOLD_CEGIS_EXHAUSTED")
    else:
        assert any("residual" in n or "no blueprint" in n
                   for n in sym["notes"])


def test_noise_domain_honest_skip():
    from hagi.pipeline import unified_ingest
    u = _fresh_unified()
    noise = bytes(np.random.default_rng(1).integers(0, 256, 6000,
                                                   dtype=np.uint8))
    r = unified_ingest(u, "noise", noise, epochs=1, repeat=1)
    assert r["symbolic"]["hold_reason"] == "RANDOM_INCOMPRESSIBLE"
    assert r["symbolic"]["reified_op"] is None


def test_symbolic_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HAGI_SYMBOLIC", raising=False)
    from hagi.pipeline import unified_ingest
    u = _fresh_unified()
    seq = bytes([(3 * i) % 16 for i in range(6000)])
    r = unified_ingest(u, "x", seq, epochs=1, repeat=1)
    assert r["symbolic"] is None
