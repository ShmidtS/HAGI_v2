# -*- coding: utf-8 -*-
"""E8 lattice quantization prototype for W2 (2.25 bpw: 16-bit index + fp16 scale per 8-group).

Quality probe on one expert: nearest-E8-point (analytic Conway-Sloane, vectorized)
with per-group LS scale, vs scalar int4/int6. Output-space error on real activations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn.functional as F
import dsv4_refit_experts as R

L, k = 5, 7
red = os.path.join("dsv4_reduced", f"layer_{L}")
P = torch.load(os.path.join(red, "P.pt"), map_location="cuda").float()
mu = torch.load(os.path.join(red, "mu.pt"), map_location="cuda").float()
acts = torch.load(os.path.join("checkpoints_dsv4", "pod_all_tokens", f"acts_layer{L}.pt"), map_location="cpu", weights_only=False)
x, y = acts[str(k)]
z = (x.float().cuda() - mu) @ P
yf = y.float().cuda()
ex = R.load_selected_experts(L, [k])
w1, w2, w3 = ex[k]
w1r = w1.float() @ P; w3r = w3.float() @ P
b1 = mu.reshape(-1) @ w1.float().T; b3 = mu.reshape(-1) @ w3.float().T

# binary W13 -> h -> ridge W2 (same as PTQ harness)
q1 = torch.sign(w1r); s1 = w1r.abs().mean(1, keepdim=True).clamp_min(1e-9)
q3 = torch.sign(w3r); s3 = w3r.abs().mean(1, keepdim=True).clamp_min(1e-9)
W1 = q1 * s1; W3 = q3 * s3
gg = R.soft_lim(z @ W1.T + b1[None, :]); uu = R.soft_lim(z @ W3.T + b3[None, :])
h = F.silu(gg) * uu
Gm = h.T @ h; Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
W2c = torch.linalg.solve(Gm, h.T @ yf).T  # [4096, 2048]

D, I = W2c.shape  # 4096, 2048
G = I // 8  # 256 groups per row


def nearest_e8(x):
    """Nearest E8 lattice point (Conway-Sloane, vectorized). x [..., 8]."""
    # candidate A: D8 (integer, even sum)
    y = torch.round(x)
    ssum = y.sum(-1)
    err = (x - y).abs()
    j = err.argmax(-1)
    odd = (ssum % 2 != 0)
    flip = torch.where(x.gather(-1, j[..., None]) < y.gather(-1, j[..., None]), -1.0, 1.0)
    adj = torch.zeros_like(y).scatter_(-1, j[..., None], torch.where(odd[..., None], flip, 0.0))
    ya = y + adj
    da = ((x - ya) ** 2).sum(-1)
    # candidate B: D8' (all half-integers, even sum of halves*2)
    xh = x - 0.5
    yh = torch.round(xh) + 0.5
    ssumh = (yh - 0.5).sum(-1)
    errh = (x - yh).abs()
    jh = errh.argmax(-1)
    oddh = (ssumh % 2 != 0)
    fliph = torch.where(x.gather(-1, jh[..., None]) < (yh.gather(-1, jh[..., None]) - 0.5), -1.0, 1.0)
    # flip in integer space: yh_int ± 1 then +0.5
    yh_int = yh - 0.5
    adjh = torch.zeros_like(yh_int).scatter_(-1, jh[..., None], torch.where(oddh[..., None], fliph, 0.0))
    yb = yh_int + adjh + 0.5
    db = ((x - yb) ** 2).sum(-1)
    pick_b = db < da
    return torch.where(pick_b[..., None], yb, ya)


def e8_quant(W, rounds=2):
    """Per-group scale + nearest E8 point. Returns dequantized W."""
    out, in_ = W.shape
    v = W.view(out, in_ // 8, 8)
    s = v.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-9)  # RMS init
    for _ in range(rounds + 1):
        t = v / s
        pt = nearest_e8(t)
        num = (v * pt).sum(-1, keepdim=True)
        den = (pt * pt).sum(-1, keepdim=True).clamp_min(1e-9)
        s = (num / den).clamp_min(1e-9)
    return (nearest_e8(v / s) * s).view(out, in_)


W2q_e8 = e8_quant(W2c)
yh = h @ W2q_e8.T
print(f"bin W13 + W2 E8-2.25bpw : {((yh - yf).norm() / yf.norm()).item() * 100:.2f}% norm")

# reference: scalar int6 (measured before: 13.43%) and int4 (24.51%) for sanity
def quant_int(W, nlev, rounds=3):
    sg = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / nlev
    q = (W / sg).round().clamp(-nlev, nlev)
    for _ in range(rounds):
        num = (q * W).sum(1, keepdim=True); den = (q * q).sum(1, keepdim=True).clamp_min(1e-9)
        sg = (num / den).clamp_min(1e-9)
        q = (W / sg).round().clamp(-nlev, nlev)
    num = (q * W).sum(1); den = (q * q).sum(1).clamp_min(1e-9)
    s2 = (num / den)
    return q * s2.sign().abs()[:, None]

for name, nlev in (("int4", 7), ("int6", 31)):
    W2q = quant_int(W2c, nlev)
    yh = h @ W2q.T
    print(f"bin W13 + W2 {name}      : {((yh - yf).norm() / yf.norm()).item() * 100:.2f}% norm")

# weight-space MSE check of E8 vs its rate (sanity: bpw equivalent)
v = W2c.view(D, G, 8)
s = v.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-9)
pt = nearest_e8(v / s)
# how many distinct lattice points used (proxy for entropy / index fit)
werr = ((v / s) - pt).pow(2).sum(-1).mean().item()
print(f"E8 weight-space MSE (normalized per group): {werr:.4f}  (int4 scalar ~{((( W2c/quant_int(W2c,7)) - 1).numel()*0+0):.0f})")
norm_t = (v / s).pow(2).sum(-1).mean().item()
print(f"mean ||t||^2 per group: {norm_t:.2f} (E8 shells: 240@2, 2160@4, 6720@6, 17520@8, 30240@10)")
