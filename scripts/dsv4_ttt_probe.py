"""TTT probe v2: plastic readout + hybrid memory (int4 base + atom-residual bank).

Stream = REAL activations in token order (natural drift). Per expert:
  1. static int4 (offline fit) - baseline; its prequential mean = "no thinking"
  2. anchored RLS online (lambda sweep) - continuous plastic readout
  3. consolidation to int4 (bounded memory, parity expected)
  4. HYBRID: consolidated int4 + m recent (h, e) atoms solved by small dual
     ridge - the "thinking memory": KV-like residual at bounded MB cost
Aggregates over experts of layers 0..3 (whenever checkpoints exist).
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from dsv4_experts import unpack_binary, unpack_int4
from dsv4_refit_experts import soft_lim

RED = "dsv4_reduced"
POD = os.path.join("checkpoints_dsv4", "pod_all_tokens")
DEV = "cuda"
N_STREAM = 1024
REFIT = 64
ATOMS = int(os.environ.get("TTT_ATOMS", "128"))


def load_expert(L, k):
    red = os.path.join(RED, f"layer_{L}")
    if not os.path.exists(os.path.join(red, f"expert_{k}.pt")):
        return None
    P = torch.load(os.path.join(red, "P.pt"), map_location=DEV).float()
    mu = torch.load(os.path.join(red, "mu.pt"), map_location=DEV).float()
    acts = torch.load(os.path.join(POD, f"acts_layer{L}.pt"), map_location="cpu", weights_only=False)
    x, y = acts[str(k)]
    del acts
    z = (x.float().to(DEV) - mu) @ P
    e = torch.load(os.path.join(red, f"expert_{k}.pt"), map_location="cpu", weights_only=False)
    if e.get("mode") != "int4x":
        return None
    q1 = unpack_binary(e["w1a"]).float().to(DEV)
    q3 = unpack_binary(e["w3a"]).float().to(DEV)
    q2 = unpack_int4(e["w2a"]).float().to(DEV)
    w1 = q1 * e["w1a_scale"].float().to(DEV)[:, None]
    w3 = q3 * e["w3a_scale"].float().to(DEV)[:, None]
    w2_int4 = q2 * e["w2a_scale"].float().to(DEV)[:, None]
    g = soft_lim(z @ w1.T + e["bias1a"].float().to(DEV))
    u = soft_lim(z @ w3.T + e["bias3a"].float().to(DEV))
    h = (F.silu(g) * u).float()
    return h, y.float().to(DEV), w2_int4


def resid(w2, h, y, extra=None):
    yh = h @ w2.T
    if extra is not None:
        yh = yh + extra
    return (((yh - y) ** 2).sum() / (y**2).sum()).item()


def snap_int4(w2):
    sg = w2.abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 7.0
    q = (w2 / sg).round().clamp(-7, 7)
    num = (q * w2).sum(dim=1)
    den = (q * q).sum(dim=1).clamp_min(1e-9)
    a = num / den
    a = a.sign() * a.abs().clamp_min(1e-6)
    return q * a[:, None]


def run_expert(L, k, lam, alpha_mult):
    r = load_expert(L, k)
    if r is None:
        return None
    h, y, w2_int4 = r
    n = h.shape[0]
    Fd = h.shape[1]
    hs, hh = h[:N_STREAM], y[:N_STREAM]
    ht, yt = h[N_STREAM:], y[N_STREAM:]
    out = {"static": resid(w2_int4, ht, yt)}
    out["static_preq"] = resid(w2_int4, hs, hh)

    alpha = (hs.T @ hs).diagonal().mean() / N_STREAM * alpha_mult
    G = torch.eye(Fd, device=DEV) * alpha
    C = alpha * w2_int4.T.clone()
    w2 = w2_int4.clone()
    preq = []
    for t in range(N_STREAM):
        hv, yv = hs[t : t + 1], hh[t : t + 1]
        preq.append(resid(w2, hv, yv))
        G.mul_(lam).add_(hv.T @ hv)
        C.mul_(lam).add_(hv.T @ yv)
        if (t + 1) % REFIT == 0:
            reg = G.diagonal().mean() * 1e-3
            w2 = torch.linalg.solve(G + reg * torch.eye(Fd, device=DEV), C).T.contiguous()
    out["rls"] = resid(w2, ht, yt)
    out["rls_preq"] = torch.tensor(preq[-256:]).mean().item()

    base = snap_int4(w2)
    out["cons"] = resid(base, ht, yt)

    # hybrid: bounded atom bank on the residual, dual ridge over m atoms
    Hr = hs[-ATOMS:]
    Er = hh[-ATOMS:] - Hr @ base.T
    Kk = Hr @ Hr.T
    Kk = Kk + Kk.diagonal().mean() * 1e-2 * torch.eye(ATOMS, device=DEV)
    A = torch.linalg.solve(Kk, Er)  # [m, D]
    out["hybrid"] = resid(base, ht, yt, extra=(ht @ Hr.T) @ A)
    return out


def main():
    lam = float(os.environ.get("TTT_LAM", "0.9995"))  # sweep: 0.995 worse (1.87), 0.9995 best
    alpha_mult = float(os.environ.get("TTT_ALPHA", "1000"))
    n_exp = int(os.environ.get("TTT_N", "12"))
    keys = ("static", "static_preq", "rls", "rls_preq", "cons", "hybrid")
    agg = {k: [] for k in keys}
    cnt = 0
    for L in range(4):
        for k in range(256):
            if cnt >= n_exp:
                break
            out = run_expert(L, k, lam, alpha_mult)
            if out is None:
                continue
            cnt += 1
            for kk in keys:
                agg[kk].append(out[kk])
            print(
                f"L{L} k{k:3d}: static {out['static']*100:.2f} | rls {out['rls']*100:.2f} | "
                f"cons {out['cons']*100:.2f} | hybrid {out['hybrid']*100:.2f}",
                flush=True,
            )
        if cnt >= n_exp:
            break
    print(f"\n=== AGGREGATE over {cnt} experts (lam={lam}, alpha={alpha_mult}, atoms={ATOMS}):")
    for kk in keys:
        v = torch.tensor(agg[kk])
        print(f"  {kk:12s}: mean {v.mean()*100:.3f}%  med {v.median()*100:.3f}%")


if __name__ == "__main__":
    main()
