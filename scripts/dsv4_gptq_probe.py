"""GPTQ-style Hessian-aware 2-bit probe for one DSV4 expert (vs binary4x champion 3.7113%).

Layerwise: quantize W1, W3 with H = Z^T Z (real activation buffer);
W2 with H = Hh^T Hh where Hh = silu(clamp(zW1q)) * clamp(zW3q) (quantized upstream).
Baselines: RTN vs GPTQ error-feedback, group sizes, bit widths.
Metric: resid = ||yhat - y||^2 / ||y||^2 on the SAME buffer the refit trainer uses.
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from dsv4_experts import load_selected_experts, ffn, SWIGLU_LIMIT

L, K = 0, 0
ACTS = os.path.join("checkpoints_dsv4", "pod_all_tokens", f"acts_layer{L}.pt")
DEV = "cuda"
GROUP = int(os.environ.get("GPTQ_GROUP", "64"))
BITS_W13 = int(os.environ.get("GPTQ_BITS_W13", "2"))
BITS_W2 = int(os.environ.get("GPTQ_BITS_W2", "2"))
NMAX = int(os.environ.get("GPTQ_NMAX", "300000"))
QS = {(2,): ((-1.5, -0.5, 0.5, 1.5), 1.5), (3,): (torch.arange(-3.5, 4.0).tolist(), 3.5),
      (4,): (torch.arange(-7, 8, 2).tolist()[:8], 7.0)}


def levels_for(bits: int):
    lev, qm = QS[(bits,)]
    return torch.tensor(lev, dtype=torch.float32, device=DEV), qm


def group_scales(w: torch.Tensor, group: int, qmax: float):
    """[out, in] -> per-(out, in/group) symmetric scales."""
    out, inn = w.shape
    wg = w.view(out, inn // group, group)
    return (wg.abs().amax(dim=2, keepdim=True) / qmax).clamp_min(1e-8)


def rtn(w: torch.Tensor, bits: int, group: int):
    lev, qm = levels_for(bits)
    out, inn = w.shape
    wg = w.view(out, inn // group, group)
    s = (wg.abs().amax(dim=2, keepdim=True) / qm).clamp_min(1e-8)
    mids = (lev[:-1] + lev[1:]) / 2
    idx = torch.bucketize(wg / s, mids)
    return (lev[idx] * s).view(out, inn).to(w.dtype), s.squeeze(2)


def gptq(w: torch.Tensor, hess: torch.Tensor, bits: int, group: int, damp: float = 0.01):
    """Classic per-column error feedback. w [out, in], hess [in, in]."""
    out, inn = w.shape
    d = hess.diagonal().mean()
    h = hess + d * damp * torch.eye(inn, device=w.device)
    hinv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    w = w.clone().float()
    wq = torch.zeros_like(w)
    lev, qm = levels_for(bits)
    mids = (lev[:-1] + lev[1:]) / 2
    for i in range(inn):
        col = w[:, i]
        # group scale from the ORIGINAL block (stable across feedback)
        gi = (i // group) * group
        ge = min(gi + group, inn)
        if i == gi or i == 0:
            s_blk = (w[:, gi:ge].abs().amax(dim=1, keepdim=True) / qm).clamp_min(1e-8)
        x = col / s_blk.squeeze(1)
        idx = torch.bucketize(x, mids)
        q = lev[idx] * s_blk.squeeze(1)
        wq[:, i] = q
        if i + 1 < inn:
            errcol = (col - q) / hinv[i, i]
            w[:, i + 1:] -= torch.outer(errcol, hinv[i, i + 1:] / hinv[i, i])
    return wq.to(torch.float32)


def hessian_from(x: torch.Tensor) -> torch.Tensor:
    """x [n, d] -> x^T x (fp32, chunked)."""
    d = x.shape[1]
    h = torch.zeros(d, d, device=DEV, dtype=torch.float32)
    for i in range(0, x.shape[0], 32768):
        c = x[i : i + 32768].float().to(DEV)
        h += c.T @ c
    return h


def resid(w1, w2, w3, z, y):
    se = torch.zeros((), device=DEV)
    tot = torch.zeros((), device=DEV)
    for i in range(0, z.shape[0], 16384):
        zc = z[i : i + 16384].to(DEV, non_blocking=True)
        yc = y[i : i + 16384].float().to(DEV)
        yhat = ffn(zc, w1.to(zc.dtype), w2.to(zc.dtype), w3.to(zc.dtype)).float()
        se += ((yhat - yc) ** 2).sum()
        tot += (yc**2).sum()
    return (se / tot).item()


def main():
    torch.manual_seed(0)
    acts = torch.load(ACTS, map_location="cpu", weights_only=False)
    x, y = acts[str(K)]
    if x.shape[0] > NMAX:
        x, y = x[:NMAX], y[:NMAX]
    print(f"expert L{L} k{K}: n={x.shape[0]} z={tuple(x.shape)} y={tuple(y.shape)}")
    ex = load_selected_experts(L, [K])[K]
    w1o, w2o, w3o = (t.float().to(DEV) for t in ex)
    I = w1o.shape[0]
    for nm, w in (("w1", w1o), ("w2", w2o), ("w3", w3o)):
        print(f"  orig {nm}: {tuple(w.shape)}")
    r_orig = resid(w1o, w2o, w3o, x, y)
    print(f"  resid original (dequant fp4): {r_orig * 100:.4f}%")

    hz = hessian_from(x)
    params = I * 4096 * 3
    budget_mb = params * (2 / 8) / 2**20  # all-2bit

    def name(tag, b13, b2, g):
        mb = (I * 4096 * 2 * b13 / 8 + 4096 * I * b2 / 8) / 2**20
        return f"{tag} w13={b13}b w2={b2}b g={g} [{mb:.2f} MB]"

    for tag, b13, b2, g in [
        ("gptq", BITS_W13, BITS_W2, GROUP),
        ("gptq", 3, 3, GROUP),
    ]:
        if tag == "rtn ":
            w1q, _ = rtn(w1o, b13, g)
            w3q, _ = rtn(w3o, b13, g)
        else:
            w1q = gptq(w1o, hz, b13, g, damp=float(os.environ.get("GPTQ_DAMP", "0.01")))
            w3q = gptq(w3o, hz, b13, g, damp=float(os.environ.get("GPTQ_DAMP", "0.01")))
        zc = x[:65536].to(DEV).float()
        hq = F.silu((zc @ w1q.T).clamp(max=SWIGLU_LIMIT)) * (zc @ w3q.T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        hh = hq.float().T @ hq.float()
        if tag == "rtn ":
            w2q, _ = rtn(w2o, b2, g)
        else:
            w2q = gptq(w2o, hh, b2, g, damp=float(os.environ.get("GPTQ_DAMP", "0.01")))
        r = resid(w1q, w2q, w3q, x, y)
        print(f"{name(tag, b13, b2, g)}: resid {r * 100:.4f}%", flush=True)
    print(f"budget: all-2bit = {budget_mb:.2f} MB (target 6.3 MB)")


if __name__ == "__main__":
    main()
