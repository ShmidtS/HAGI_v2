"""Unified pipeline (WAVE 185): neural + symbolic in one flow."""
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    assert sym["invented"]["status"] == "INVENTED"
    assert sym["exact_points"] == 16
    assert sym["reified_op"]


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
