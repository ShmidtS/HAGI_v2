"""Benchmark the real spectral branch forward+backward at real shapes.

Goal: how much of a step does the spectral path cost? Is the fp32 tiny readout
matmul (out_r @ w_re + out_i @ w_im) material? Measure components separately.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "src")
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import torch
from hagi.config import load_config
from hagi.model.spectral import SpectralRecurrence
from hagi.train.loop import cast_model

torch.manual_seed(0)
cfg = load_config("configs/v33_1b.yaml")
m = SpectralRecurrence(cfg.model.hidden_size, cfg.model.spectral,
                       cfg.model.norm_eps, cfg.model.ternary.enabled, 1.0,
                       cfg.model.init_orthogonal).to("cuda")
cast_model(m, "bf16")
B, T, H = 30, 1024, cfg.model.hidden_size
x = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

def bench(fn, name, iters=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    print(f"{name:52s} {(time.perf_counter()-t0)/iters*1000:7.2f} ms")

def fwd(): return m(x)
def fwd_bwd():
    y = m(x); y.sum().backward()

bench(fwd, "spectral forward (real, bf16 cast)")
bench(fwd_bwd, "spectral forward+backward")

# components: in_proj, scan, readout (fp32), out_proj
h = m.norm(x)
y = m.in_proj(h)
half = m.h_in // 2
xr = y[..., :half][..., :m.num_modes].float()
xi = y[..., half:half*2][..., :m.num_modes].float()
x_c = torch.complex(xr, xi)
out_c, _ = m._scan(x_c, None, False)
bench(lambda: m._scan(x_c, None, False), "  scan alone (complex fp32)")

def readout():
    out_r = out_c.real; out_i = out_c.imag
    w_re = m.w_re.float(); w_im = m.w_im.float()
    return out_r @ w_re + out_i @ w_im
bench(readout, "  readout fp32 (out_r@w_re + out_i@w_im)")
bench(lambda: out_c.real @ m.w_re.float(), "  out_r@w_re only")
