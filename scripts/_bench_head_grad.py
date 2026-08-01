"""Decisive test: is bf16 softmax backward safe NOW (z_loss + logit_scale regime)?

Prior rejection was measured BEFORE z_loss/logit_scale/unigram_prior existed:
logits drifted to +-100, exp() overflowed bf16 exponent -> NaN ~step 250.
Current regime pins lse ~ 0 via z_loss and scales logits by ~1/sqrt(H).

Measure grad_weight relative error fp32 vs bf16-maxshift on rare-target rows.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, "src")
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import torch

torch.manual_seed(0)
H, V, N = 1152, 32768, 4096
dev = "cuda"

# Realistic regime: logit_scale ~ 1/sqrt(H), unigram prior, z_loss pins lse ~ 0
h = torch.randn(N, H, device=dev) * (H ** -0.5)          # logit_scale'd hidden
W = torch.randn(V, H, device=dev) * (H ** -0.5)          # codebook rows
# unigram-like prior with rare tail: 2000 rare tokens at log p ~ -12
counts = torch.full((V,), 500, device=dev)
counts[:2000] = 1                                        # rare tail
prior = (counts / counts.sum()).log()
tgt = torch.randint(0, V, (N,), device=dev)
tgt[:200] = torch.randint(0, 2000, (200,), device=dev)   # force rare targets

logits = h @ W.t() + prior.unsqueeze(0)                  # [N, V] with prior bias

def ce_grad(lg, t):
    lg = lg.float()
    lse = lg.logsumexp(-1, keepdim=True)
    probs = (lg - lse).exp()
    g = probs.clone()
    g.scatter_add_(-1, t.unsqueeze(-1), torch.full_like(lse, -1.0))
    gw = g.t() @ h
    gh = g @ W
    return gw, gh, probs

# fp32 reference
gw32, gh32, p32 = ce_grad(logits, tgt)

# bf16 max-shift (what the head would do without the .float() cast)
lg16 = logits.bfloat16()
mx = lg16.max(-1, keepdim=True).values
lse16 = ((lg16 - mx).exp().sum(-1, keepdim=True).log() + mx)
p16 = (lg16 - lse16).exp()
g16 = p16.clone()
g16.scatter_add_(-1, tgt.unsqueeze(-1), torch.full_like(lse16, -1.0))
gw16 = (g16.to(W.dtype).t() @ h)
gh16 = (g16.to(W.dtype) @ W)

rel_err = lambda a, b: (a - b).norm() / a.norm()
print(f"grad_weight rel err: {rel_err(gw32, gw16):.4f}")
print(f"grad_hidden rel err: {rel_err(gh32, gh16):.4f}")
print(f"probs rel err:       {rel_err(p32, p16.float()):.4f}")
print(f"lse err (nats):      {(lse32 := logits.float().logsumexp(-1, keepdim=True)) .sub(lse16.float()).abs().max().item():.4f}")

# worst-case: does any rare row get grad zeroed in bf16 that fp32 keeps nonzero?
rare_rows = tgt[:200]
gr32 = gw32[rare_rows].norm(dim=-1)
gr16 = gw16[rare_rows].norm(dim=-1)
zeroed = (gr16 == 0).sum().item()
print(f"rare rows zeroed in bf16: {zeroed}/200")
print(f"rare grad norm ratio bf16/fp32: {(gr16 / gr32).median().item():.4f}")

# timing on full-size
def bench(fn, name, iters=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
    t0.record()
    for _ in range(iters): fn()
    t1.record(); torch.cuda.synchronize()
    print(f"{name:40s} {t0.elapsed_time(t1)/iters:7.2f} ms")

import torch.nn.functional as F
lg_big = (torch.randn(30720, V, device=dev, dtype=torch.bfloat16))
t_big = torch.randint(0, V, (30720,), device=dev)
bench(lambda: F.cross_entropy(lg_big.float(), t_big), "CE fp32 (current)")
bench(lambda: F.cross_entropy(lg_big, t_big), "CE bf16 (candidate)")
