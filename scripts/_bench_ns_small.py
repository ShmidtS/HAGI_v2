"""Verify small-Gram polar iteration == current big-Gram quintic, and time both.

Current: x = addmm(a·x, poly, x) with poly = b·(x@x^T) + c·(x@x^T)@(x@x^T)   [M,M]
Small:   P = aI + bG + cG² with G = x^T@x [K,K]; x = x@P
Both compute aX + bX(X^TX) + cX(X^TX)², exactly.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "src")
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import torch
from hagi.train.optim import newton_schulz as current_ns

torch.manual_seed(0)
dev = "cuda"
A, B, C = 3.4445, -4.7750, 2.0315

def small_ns(grad, steps=5, coeffs=(A, B, C)):
    a, b, c = coeffs
    x = grad.bfloat16()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        g = x.t() @ x                       # [K,K] small Gram
        p = torch.addmm(g, g, g, beta=b, alpha=c)  # bG + cG² (fused addmm)
        p.diagonal().add_(a)                # + aI  => P = aI + bG + cG²
        x = x @ p                           # X·P
    if transposed:
        x = x.T
    return x.to(grad.dtype)

# equivalence on representative shapes
shapes = [(2688, 1152), (1152, 2688), (1152, 1152), (576, 1152), (1152, 576)]
print("equivalence (rel err of small vs current):")
for shp in shapes:
    g = torch.randn(30, *shp, device=dev, dtype=torch.bfloat16).mean(0)
    u_cur = current_ns(g, steps=5)
    u_new = small_ns(g, steps=5)
    rel = (u_cur - u_new).norm() / u_cur.norm()
    # exact singular-spread equivalence
    sc = torch.linalg.svdvals(u_cur.float()); sn = torch.linalg.svdvals(u_new.float())
    print(f"  {str(shp):16s} rel={rel:.2e}  spread cur={sc.max()/sc.min():.2f} new={sn.max()/sn.min():.2f}")

# timing: 140 params, realistic mix (tall majority + square)
mix = [(2688, 1152)]*80 + [(1152, 1152)]*30 + [(576, 1152)]*30
grads = [torch.randn(30, *s, device=dev, dtype=torch.bfloat16).mean(0) for s in mix]

def run(fn, iters=5):
    for _ in range(2): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1000

print("\nwall time 140 params (5 steps):")
print(f"  current big-Gram: {run(lambda: [current_ns(g) for g in grads]):7.1f} ms")
print(f"  small-Gram:       {run(lambda: [small_ns(g) for g in grads]):7.1f} ms")

# also 3-step quality comparison on the square case (where current is known bad)
g = torch.randn(30, 1152, 1152, device=dev, dtype=torch.bfloat16).mean(0)
for steps in (3, 5):
    u = small_ns(g, steps=steps)
    s = torch.linalg.svdvals(u.float())
    print(f"  square [1152,1152] small-Gram s{steps} spread={s.max()/s.min():.2f}")
