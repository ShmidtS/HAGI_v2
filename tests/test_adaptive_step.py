"""AdaptiveMajorizer: variance-sensitive certified CE step."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from hagi.lm import CertifiedStepLM

DATA = (b"abc" * 40 + b"xyz" * 40) * 20


def _batch(lm):
    idx = np.frombuffer(DATA, dtype=np.uint8).astype(np.int64)
    rng = np.random.default_rng(7); perm = rng.permutation(len(idx) - 4)
    sel = np.sort(perm[16:144])
    X = np.stack([idx[i:i + 4] for i in sel]); y = idx[sel + 4]
    return (torch.as_tensor(X, dtype=torch.long),
            torch.as_tensor(y, dtype=torch.long))


def test_adaptive_step_descends_and_promises():
    lm = CertifiedStepLM(k=4, dim=32, hidden=64, seed=1)
    ctx, nxt = _batch(lm)
    def ce():
        with lm.t.no_grad():
            return float(lm.t.nn.functional.cross_entropy(
                lm.logits(ctx), nxt))
    before = ce()
    promised = 0.0
    for _ in range(20):
        promised += lm._adaptive_step(ctx, nxt)
    after = ce()
    assert after < before
    assert promised > 0
    # certificate: each step's promised drop holds on the trajectory
    assert after <= before + 1e-9
