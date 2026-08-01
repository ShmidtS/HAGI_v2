"""Hybrid NS: square params need 5 steps (bad spread at 3), tall/wide are fine at 3.

Measure wall-time and quality of the hybrid over the real param mix.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "src")
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import torch
from hagi.train.optim import newton_schulz

torch.manual_seed(0)
dev = "cuda"

def ns_steps(grad, steps):
    return newton_schulz(grad, steps=steps)

def hybrid(grad):
    t = grad.shape[0] > grad.shape[1]
    sq = grad.shape[0] == grad.shape[1]
    return ns_steps(grad, 5 if sq else 3)

# realistic mix: tall/wide dominate, few square
mix = [(2688, 1152)]*80 + [(1152, 2688)]*10 + [(1152, 1152)]*15 + [(576, 1152)]*30 + [(1152, 576)]*5
grads = [torch.randn(30, *s, device=dev, dtype=torch.bfloat16).mean(0) for s in mix]

def spread(x):
    s = torch.linalg.svdvals(x.float())
    return (s.max()/s.clamp_min(1e-9).min()).item()

# quality: worst spread across mix
print("worst spread (lower = more orthogonal):")
print(f"  all-5:      {max(spread(ns_steps(g,5)) for g in grads):8.2f}")
print(f"  hybrid 3/5: {max(spread(hybrid(g)) for g in grads):8.2f}")
print(f"  all-3:      {max(spread(ns_steps(g,3)) for g in grads):8.2f}")

def run(fn, iters=5):
    for _ in range(2): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1000

print("\nwall time over mix (140 params):")
print(f"  all-5:      {run(lambda: [ns_steps(g,5) for g in grads]):7.1f} ms")
print(f"  hybrid 3/5: {run(lambda: [hybrid(g) for g in grads]):7.1f} ms")
print(f"  all-3:      {run(lambda: [ns_steps(g,3) for g in grads]):7.1f} ms")
