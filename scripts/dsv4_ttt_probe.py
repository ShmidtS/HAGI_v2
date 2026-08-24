"""TTT probe: plastic readout on frozen int4x features, streaming on REAL activations.

Question: can weights update during inference, stably, with bounded memory?
Setup (expert k, layer L):
  - features h = f_int4x(z) frozen (unpack packed signs/biases from the saved checkpoint)
  - readout W2: start at the int4 values, then adapt ONLINE on a row stream:
      RLS (sufficient stats G=Sum lam^t h h^T, C=Sum lam^t h y^T, exact refit) vs
      naive per-token SGD (outer-product updates, the classic fast-weight rule)
  - holdout tail never trained on: measures real generalization of the update
  - consolidation: snap RLS W2 back to int4 + per-channel LS scale (bounded memory)
Metrics: static / online / consolidated holdout resid, prequential curve, max spike.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from dsv4_experts import unpack_binary, unpack_int4
from dsv4_refit_experts import soft_lim

L, K = 0, int(os.environ.get("TTT_K", "0"))
RED = os.path.join("dsv4_reduced", f"layer_{L}")
POD = os.path.join("checkpoints_dsv4", "pod_all_tokens")
DEV = "cuda"
N_STREAM = 1024
REFIT = 64


def load_stuff(k):
    P = torch.load(os.path.join(RED, "P.pt"), map_location=DEV).float()
    mu = torch.load(os.path.join(RED, "mu.pt"), map_location=DEV).float()
    acts = torch.load(os.path.join(POD, f"acts_layer{L}.pt"), map_location="cpu", weights_only=False)
    x, y = acts[str(k)]
    del acts
    z = (x.float().to(DEV) - mu) @ P  # [n, K]
    e = torch.load(os.path.join(RED, f"expert_{k}.pt"), map_location="cpu", weights_only=False)
    assert e["mode"] == "int4x"
    q1 = unpack_binary(e["w1a"]).float().to(DEV)
    s1 = e["w1a_scale"].float().to(DEV)
    q3 = unpack_binary(e["w3a"]).float().to(DEV)
    s3 = e["w3a_scale"].float().to(DEV)
    q2 = unpack_int4(e["w2a"]).float().to(DEV)
    s2 = e["w2a_scale"].float().to(DEV)
    b1 = e["bias1a"].float().to(DEV)
    b3 = e["bias3a"].float().to(DEV)
    return z, y.float().to(DEV), q1 * s1[:, None], q3 * s3[:, None], q2, s2, b1, b3


def features(z, w1, w3, b1, b3):
    g = soft_lim(z @ w1.T + b1)
    u = soft_lim(z @ w3.T + b3)
    return F.silu(g) * u  # [n, F]


def resid(w2, h, y):
    yh = h @ w2.T
    return (((yh - y) ** 2).sum() / (y**2).sum()).item()


def snap_int4(w2):
    """continuous [D, F] -> int4 pattern + per-channel LS scale (consolidation)."""
    sg = w2.abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 7.0
    q = (w2 / sg).round().clamp(-7, 7)
    num = (q * w2).sum(dim=1)
    den = (q * q).sum(dim=1).clamp_min(1e-9)
    a = (num / den)
    a = a.sign() * a.abs().clamp_min(1e-6)
    return q, a


def main():
    lam = float(os.environ.get("TTT_LAM", "0.999"))
    lr_sgd = float(os.environ.get("TTT_SGD_LR", "3e-3"))
    z, y, w1, w3, q2, s2, b1, b3 = load_stuff(K)
    n = z.shape[0]
    h = features(z, w1, w3, b1, b3).float()  # frozen features [n, F]
    Fd = h.shape[1]
    hs, hh = h[:N_STREAM], y[:N_STREAM]
    ht, yt = h[N_STREAM:], y[N_STREAM:]  # holdout
    w2_int4 = q2 * s2[:, None]
    print(f"expert L{L} k{K}: n={n} F={Fd} stream={N_STREAM} holdout={n - N_STREAM}")
    print(f"  static int4 holdout resid: {resid(w2_int4, ht, yt) * 100:.4f}%")

    # --- honest baselines on the SAME first-half data as the online stream ---
    gam = (hs.T @ hs).diagonal().mean() * float(os.environ.get("TTT_BREG", "1.0"))
    Kk = hs.T @ hs + gam * torch.eye(Fd, device=DEV)
    w2_half = torch.linalg.solve(Kk, hs.T @ hh + gam * w2_int4.T).T.contiguous()  # ridge TOWARD the int4 init
    print(f"  [baseline] continuous ridge (first half): holdout {resid(w2_half, ht, yt) * 100:.4f}%")
    qb, ab_ = snap_int4(w2_half)
    print(f"  [baseline] int4 snap of it:               holdout {resid(qb * ab_[:, None], ht, yt) * 100:.4f}%")

    # ---- RLS: sufficient statistics + periodic exact refit ----
    # prior toward the static int4 solution: prevents rank-deficient early
    # refits (F=2048 > stream tokens; measured 57% holdout without it)
    alpha = (hs.T @ hs).diagonal().mean() / N_STREAM * float(os.environ.get('TTT_ALPHA', '10'))
    G = torch.eye(Fd, device=DEV) * alpha
    C = alpha * w2_int4.T.clone()
    w2 = w2_int4.clone()
    preq = []
    for t in range(N_STREAM):
        hv, yv = hs[t : t + 1], hh[t : t + 1]
        preq.append(resid(w2, hv, yv))
        G.mul_(lam).add_(hv.T @ hv)  # rank-1 (forgetting)
        C.mul_(lam).add_(hv.T @ yv)
        if (t + 1) % REFIT == 0:
            reg = G.diagonal().mean() * 1e-3
            w2 = torch.linalg.solve(G + reg * torch.eye(Fd, device=DEV), C).T.contiguous()
    preq = torch.tensor(preq)
    r_rls = resid(w2, ht, yt)
    print(
        f"  RLS (lam={lam}, refit {REFIT}): holdout {r_rls * 100:.4f}% | "
        f"preq first100 {(preq[:100].mean()) * 100:.3f}% last100 {(preq[-100:].mean()) * 100:.3f}% | "
        f"max spike {preq.max() * 100:.2f}%"
    )
    # consolidation: requantize the adapted readout back to int4 (bounded memory)
    qc, ac = snap_int4(w2)
    print(f"  consolidated int4 holdout resid: {resid(qc * ac[:, None], ht, yt) * 100:.4f}%")

    # ---- naive per-token SGD (outer-product fast weights) ----
    w2s = w2_int4.clone()
    preq_s = []
    for t in range(N_STREAM):
        hv, yv = hs[t : t + 1], hh[t : t + 1]
        preq_s.append(resid(w2s, hv, yv))
        e = w2s @ hv.T - yv.T  # [D, 1]
        w2s -= lr_sgd * (e @ hv)  # outer product
    preq_s = torch.tensor(preq_s)
    print(
        f"  SGD (lr={lr_sgd}): holdout {resid(w2s, ht, yt) * 100:.4f}% | "
        f"preq first100 {(preq_s[:100].mean()) * 100:.3f}% last100 {(preq_s[-100:].mean()) * 100:.3f}% | "
        f"max spike {preq_s.max() * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
