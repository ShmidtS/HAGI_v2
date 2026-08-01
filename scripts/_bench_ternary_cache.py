"""Benchmark ternary quantization caching vs recompute under realistic Muon updates.

In OFDM/CDMA the precoding matrix is computed once per coherence interval, not
per symbol. Here: the ternarized weight W~ = Q*s is a slow function of W (sign
flips 0.048%/step). If we recompute W~ only when W actually changes meaningfully,
we skip ~140 quantize passes per step.

But Muon updates W every step, so the cache must be invalidated each step anyway
UNLESS we can detect that few entries flipped. Test: cost of recompute vs cost
of a cheap "did it change?" check + occasional recompute.

Honest question: the quantize is 0.12ms on [2688,1152]. With ~140 weights that's
~17ms/forward. The STE backward just passes grad through (free). So caching saves
at most ~34ms/step (fwd+bwd) out of 3229 = 1%. NOT worth the complexity/risk.
Verify this claim and move on.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "src")
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import torch
import torch.nn.functional as F
from hagi.model.ternary import ternarize, _TernarizeSTE

torch.manual_seed(0); dev = "cuda"

# Realistic model: 140 ternary weights, mix of shapes
shapes = [(2688, 1152)]*80 + [(1152, 2688)]*10 + [(1152, 1152)]*15 + [(576, 1152)]*30 + [(1152, 576)]*5
weights = [torch.randn(*s, device=dev, dtype=torch.bfloat16, requires_grad=True) for s in shapes]

def bench(fn, name, iters=30):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    print(f"{name:46s} {(time.perf_counter()-t0)/iters*1000:7.2f} ms")

# full quantize of all 140 weights (what each forward does)
def all_quantize():
    for w in weights: ternarize(w)
bench(all_quantize, "quantize all 140 weights (per forward)")

# just the linear with ternarized weight (the matmul dominates? no — quantize is separate)
def all_linear():
    for w in weights:
        eff, _ = ternarize(w)
        torch.mm(torch.randn(30720, w.shape[1], device=dev, dtype=torch.bfloat16), eff.t())
bench(all_linear, "quantize + linear (what forward does)")

# memory footprint: how much does a quantized weight add? (activation caching cost)
print(f"\nper [2688,1152] weight: {2688*1152*2/1e6:.1f} MB bf16, {2688*1152/1e6:.1f} MB ternarized")
print(f"cache all 140: {sum(w.numel() for w in weights)*1/1e6:.0f} MB (ternarized s, bf16)")
print(f"cache all 140: {sum(w.numel() for w in weights)*2/1e6:.0f} MB (as bf16 full)")

# STE backward cost
def all_backward():
    for w in weights:
        _TernarizeSTE.backward(None, torch.randn_like(w))
bench(all_backward, "STE backward (identity, free?)")
