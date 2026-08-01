"""Benchmark real MoE forward+backward at real shapes before/after dispatch change."""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "src")
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import torch
from hagi.config import load_config
from hagi.model.moe import MoE
from hagi.train.loop import cast_model

torch.manual_seed(0)
cfg = load_config("configs/v33_1b.yaml")
m = MoE(cfg.model.hidden_size, 2688, cfg.model.moe, cfg.model.norm_eps,
        cfg.model.ternary.enabled, 1.0, cfg.model.init_orthogonal).to("cuda")
cast_model(m, "bf16")
B, T, H = 30, 1024, cfg.model.hidden_size
x = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)

def bench(fn, name, iters=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    print(f"{name:48s} {(time.perf_counter()-t0)/iters*1000:7.2f} ms")

def fwd():
    return m(x)

def fwd_bwd():
    y = m(x)
    y.sum().backward()

bench(fwd, "MoE forward (bf16, real SwiGLU)")
bench(fwd_bwd, "MoE forward+backward")

# dispatch-only timing (exclude expert matmuls): use identity-ish by calling internals
flat = m.norm(x).reshape(B*T, H)
logits = m.router(flat.float())
_, idx = (logits + m.expert_bias).topk(m.top_k, dim=-1)
sel_logits = logits.gather(-1, idx)
weights = sel_logits.softmax(dim=-1).to(flat.dtype)
def dispatch_only():
    m._dispatch(flat, idx, weights)
bench(dispatch_only, "dispatch only (real MoE _dispatch)")
