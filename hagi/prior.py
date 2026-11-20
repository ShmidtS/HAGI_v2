"""NeuralProposalPrior — structural-feature proposal router (softmax
over blueprint families), trained online on solved experiences. Feeds
the diagnosis→search wire (invention._ranked_quotients hints) with a
LEARNED prior instead of hand rules. Simulates the gpu.py/selfmodel
role: intuition + attention, never the certified answer.
"""
from __future__ import annotations

import numpy as np

from hagi.diagnosis import ResidualProfile

FAMILIES = ("mod", "xor", "reflect", "mask", "shift")


def profile_features(p: ResidualProfile) -> np.ndarray:
    """10-dim structural feature vector from a ResidualProfile."""
    return np.array([
        1.0,                                    # bias
        p.autocorr_shift,                       # periodicity signal
        p.autocorr_xor,                         # xor-invariance
        p.autocorr_reflect,                     # reflection
        float(p.mn_classes is not None),        # sequential task
        p.sparsity_gaps_entropy,
        p.n_runs / max(1.0, p.n),
        p.k_density,
        p.uniq_outputs / max(1.0, p.n),
        p.residual_frac_builtin,
    ], dtype=np.float64)


class NeuralProposalPrior:
    def __init__(self, lr: float = 0.8):
        self.lr = lr
        self.W = np.zeros((len(FAMILIES), 10), dtype=np.float64)

    def predict_probs(self, feat: np.ndarray) -> dict[str, float]:
        logits = self.W @ feat
        logits -= logits.max()
        exps = np.exp(logits)
        probs = exps / exps.sum()
        return {f: float(p) for f, p in zip(FAMILIES, probs)}

    def update(self, feat: np.ndarray, winner: str, epochs: int = 12) -> None:
        if winner not in FAMILIES:
            return
        tgt = FAMILIES.index(winner)
        for _ in range(epochs):
            logits = self.W @ feat
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            grad = probs.copy()
            grad[tgt] -= 1.0
            self.W -= self.lr * np.outer(grad, feat)

    def hints_from_profile(self, p: ResidualProfile) -> dict:
        """Bridge to invention._ranked_quotients: top-1 family becomes
        the hint flag (order wire, never the certified choice)."""
        probs = self.predict_probs(profile_features(p))
        best = max(probs, key=probs.get)
        flag = {fam: False for fam in FAMILIES}
        flag[best] = True
        return flag


# default shared prior (weights persist in-process)
DEFAULT_PRIOR = NeuralProposalPrior()
