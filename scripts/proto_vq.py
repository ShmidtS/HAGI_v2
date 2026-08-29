# -*- coding: utf-8 -*-
"""Spherical VQ prototype: 8-weight groups, 16-bit index into a 65536-entry
normalized codebook + per-group fp16 scale = 2.25 bpw.
Random-Gaussian normalized codebook = lower bound on E8P quality.
NN search in batches on GPU. Output-space error vs scalar int4/int6."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn.functional as F
import dsv4_refit_experts as R

torch.manual_seed(0)
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

q1 = torch.sign(w1r); s1 = w1r.abs().mean(1, keepdim=True).clamp_min(1e-9)
q3 = torch.sign(w3r); s3 = w3r.abs().mean(1, keepdim=True).clamp_min(1e-9)
W1 = q1 * s1; W3 = q3 * s3
gg = R.soft_lim(z @ W1.T + b1[None, :]); uu = R.soft_lim(z @ W3.T + b3[None, :])
h = F.silu(gg) * uu
Gm = h.T @ h; Gm.diagonal().add_(Gm.diagonal().mean() * 1e-2)
W2c = torch.linalg.solve(Gm, h.T @ yf).T  # [4096, 2048]

D, I = W2c.shape
v = W2c.view(D, I // 8, 8)  # [4096, 256, 8]

# codebook: random gaussian, normalized to unit length, [65536, 8] fp32
B = 65536
cb = torch.randn(B, 8, device="cuda")
cb = cb / cb.norm(dim=1, keepdim=True)

flat = v.reshape(-1, 8)  # [D*256, 8]
N = flat.shape[0]
idx = torch.empty(N, dtype=torch.long, device="cuda")
BS = 16384
cbT = cb.T.contiguous()
for i in range(0, N, BS):
    scores = flat[i:i+BS] @ cbT  # [bs, B]
    idx[i:i+BS] = scores.argmax(-1)  # max cos sim = min angle (cb unit, groups unnormalized direction)

pc = cb[idx]  # [N, 8] unit directions
# per-group LS scale: s = <v, p>/<p,p> = <v,p> (p unit)
s = (flat * pc).sum(-1, keepdim=True).clamp_min(1e-9)
W2q = (s * pc).view(D, I)
yh = h @ W2q.T
print(f"bin W13 + W2 VQ-2.25bpw (random cb, 1 iter): {((yh-yf).norm()/yf.norm()).item()*100:.2f}% norm")

# 2 iterations: re-encode the residual direction? classic: re-fit scale then re-NN with scaled targets
# iteration 2: NN on scale-normalized directions is identical; instead k-means style refine is too slow.
# Try: larger codebook effect via 2 additive codebooks (AQLM 2x8 style): 2x16bit idx = 4 bpw total? no - keep 2.25.

# reference (fixed quant_int)
def quant_int(W, nlev, rounds=3):
    sg = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / nlev
    q = (W / sg).round().clamp(-nlev, nlev)
    for _ in range(rounds):
        num = (q * W).sum(1, keepdim=True); den = (q * q).sum(1, keepdim=True).clamp_min(1e-9)
        sg = (num / den).clamp_min(1e-9)
        q = (W / sg).round().clamp(-nlev, nlev)
    num = (q * W).sum(1); den = (q * q).sum(1).clamp_min(1e-9)
    s2 = num / den
    return q * s2[:, None]

for name, nlev in (("int4", 7), ("int6", 31)):
    W2q = quant_int(W2c, nlev)
    yh = h @ W2q.T
    print(f"bin W13 + W2 {name} (ref)          : {((yh-yf).norm()/yf.norm()).item()*100:.2f}% norm")

# weight-space MSE comparison
mse_vq = ((W2q - W2c) ** 2).mean().item()
W2i6 = quant_int(W2c, 31); mse_i6 = ((W2i6 - W2c) ** 2).mean().item()
W2i4 = quant_int(W2c, 7); mse_i4 = ((W2i4 - W2c) ** 2).mean().item()
print(f"weight MSE: VQ-2.25 {mse_vq:.5f}  int4 {mse_i4:.5f}  int6 {mse_i6:.5f}  (var {W2c.var().item():.5f})")
