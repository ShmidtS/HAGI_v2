"""Benchmark MoE dispatch variants at real shapes (N=30720, H=1152, E=8, top_k=1).

Current _dispatch:
  order = argsort(experts_flat, stable=True)
  tokens_sorted = token_src[order]
  per-expert: sel = tokens_sorted[s:e]; expert(flat.index_select(0, sel)); out.index_add_(0, sel, ...)

Candidates:
  A. copy-scatter instead of index_add_ (top_k==1 -> unique token indices, no atomics)
  B. single gather flat[tokens_sorted] + per-expert slices + copy-scatter
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "src")
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import torch

torch.manual_seed(0)
N, H, E, K = 30720, 1152, 8, 1
dev = "cuda"
flat = torch.randn(N, H, device=dev, dtype=torch.bfloat16)
idx = torch.randint(0, E, (N, K), device=dev)
weights = torch.rand(N, K, device=dev, dtype=torch.bfloat16) + 0.5

experts_flat = idx.reshape(-1)
weights_flat = weights.reshape(-1)
token_src = torch.arange(N, device=dev).repeat_interleave(K)

def bench(fn, name, iters=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    print(f"{name:52s} {(time.perf_counter()-t0)/iters*1000:7.2f} ms")

def ref():
    """Current dispatch (sort + per-expert index_select + index_add_)."""
    out = flat.new_zeros(N, H)
    order = torch.argsort(experts_flat, stable=True)
    tokens_sorted = token_src[order]
    counts = torch.bincount(experts_flat[order], minlength=E)
    offsets = torch.cumsum(counts, dim=0) - counts
    off = offsets.tolist(); cnt = counts.tolist()
    for e in range(E):
        c = cnt[e]
        if c == 0: continue
        s = off[e]
        sel = tokens_sorted[s:s+c]
        # fake expert: identity (benchmark dispatch only)
        out.index_add_(0, sel, flat.index_select(0, sel) * weights_flat[s:s+c].unsqueeze(-1))
    return out

def cand_a():
    """copy-scatter replaces index_add_ (unique tokens under top_k=1)."""
    out = flat.new_zeros(N, H)
    order = torch.argsort(experts_flat, stable=True)
    tokens_sorted = token_src[order]
    counts = torch.bincount(experts_flat[order], minlength=E)
    offsets = torch.cumsum(counts, dim=0) - counts
    off = offsets.tolist(); cnt = counts.tolist()
    for e in range(E):
        c = cnt[e]
        if c == 0: continue
        s = off[e]
        sel = tokens_sorted[s:s+c]
        out[sel] = flat[sel] * weights_flat[s:s+c].unsqueeze(-1)
    return out

def cand_b():
    """single gather + per-expert slices + one copy-scatter."""
    order = torch.argsort(experts_flat, stable=True)
    tokens_sorted = token_src[order]
    x_sorted = flat[tokens_sorted]
    w_sorted = weights_flat[order].unsqueeze(-1)
    counts = torch.bincount(experts_flat[order], minlength=E)
    offsets = torch.cumsum(counts, dim=0) - counts
    out_sorted = flat.new_zeros(N, H)
    off = offsets.tolist(); cnt = counts.tolist()
    for e in range(E):
        c = cnt[e]
        if c == 0: continue
        s = off[e]
        out_sorted[s:s+c] = x_sorted[s:s+c] * w_sorted[s:s+c]
    out = flat.new_zeros(N, H)
    out[tokens_sorted] = out_sorted
    return out

bench(ref, "current: argsort + per-expert index_select + index_add_")
bench(cand_a, "A: copy-scatter instead of index_add_")
bench(cand_b, "B: single gather + slices + copy-scatter")

# correctness: all three must agree bitwise (top_k=1, unique assignment)
a, b = cand_a(), cand_b()
print("\nA==B:", torch.equal(a, b))
print("ref==A:", torch.equal(ref(), a))
