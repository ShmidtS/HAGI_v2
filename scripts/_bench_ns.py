"""Benchmark Newton-Schulz: accuracy (sv spread) vs steps, and wall-time.

Question: does ns_steps=3 give orthogonal-enough updates at 40% lower cost
(5->3 iters)? And how much of the 650ms is matmul FLOP vs launch overhead?
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "src")
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import torch
from hagi.train.optim import newton_schulz

torch.manual_seed(0)
dev = "cuda"
COEFFS = (3.4445, -4.7750, 2.0315)

def spread(x):
    """max singular / min singular of the update."""
    s = torch.linalg.svdvals(x.float())
    return (s.max() / s.clamp_min(1e-9).min()).item()

# shapes that dominate the model's 2D channel weights
shapes = [
    (1152, 2688),   # gate/up
    (2688, 1152),   # down
    (1152, 1152),   # qkvo
    (1152, 576),    # spectral in_proj
    (576, 1152),    # spectral out
]
print("spread (max/min singular) — target ~1.0 means orthogonal update\n")
for shp in shapes:
    g = torch.randn(30, *shp, device=dev, dtype=torch.bfloat16).mean(0)  # realistic grad scale
    row = f"{str(shp):16s}"
    for steps in (1, 2, 3, 4, 5):
        u = newton_schulz(g, steps=steps, coeffs=COEFFS)
        row += f" s{steps}={spread(u):6.2f}"
    print(row)

# wall time: how much is FLOP vs launch? Run 140 matmuls through the loop.
print("\nwall time, 140 params at [1152,2688]:")
grads = [torch.randn(1152, 2688, device=dev, dtype=torch.bfloat16) for _ in range(140)]
for steps in (3, 5):
    def run():
        for g in grads:
            newton_schulz(g, steps=steps, coeffs=COEFFS)
    for _ in range(3): run()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5): run()
    torch.cuda.synchronize()
    print(f"  ns_steps={steps}: {(time.perf_counter()-t0)/5*1000:8.1f} ms (140 params)")

# FLOP estimate for [1152,2688]: gram [M,M], poly [M,M], apply [M,K]
# one iter = 2*M*M*K (gram) + 2*M^3 (poly) + 2*M*M*K (apply), M=2688, K=1152
M, K = 2688, 1152
per_iter = 2*M*M*K + 2*M**3 + 2*M*M*K
print(f"\nFLOP/iter per [1152,2688] param: {per_iter/1e9:.1f} G; 5 iters x 140 = {5*140*per_iter/1e12:.2f} TFLOP -> at 30 TFLOP/s = {5*140*per_iter/30e12*1000:.0f} ms")
