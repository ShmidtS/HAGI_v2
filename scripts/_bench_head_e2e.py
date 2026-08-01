"""End-to-end head loss() benchmark: bf16 backward vs the old fp32 cast path."""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, "src")
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import torch
from hagi.config import load_config
from hagi.model.head import LMHead
from hagi.train.loop import cast_model

torch.manual_seed(0)
cfg = load_config("configs/v33_1b.yaml")
# head with tied weight = embedding (as in model.py)
emb = torch.randn(cfg.model.vocab_size, cfg.model.hidden_size, device="cuda", dtype=torch.bfloat16)
m = LMHead(cfg.model.hidden_size, cfg.model.vocab_size, cfg.model.head, tied_weight=emb).to("cuda")
cast_model(m, "bf16")

N = 30720  # B*T = 30*1024
hidden = torch.randn(N, cfg.model.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
tgt = torch.randint(0, cfg.model.vocab_size, (N,), device="cuda")

def bench(fn, name, iters=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    print(f"{name:46s} {(time.perf_counter()-t0)/iters*1000:7.2f} ms")

def fwd_bwd():
    ce, z = m.loss(hidden, tgt)
    (ce + z).backward()

bench(fwd_bwd, "head loss() fwd+bwd (new bf16)")

# Verify grad is finite and nonzero on rare rows
hidden.grad = None
ce, z = m.loss(hidden, tgt)
(ce + z).backward()
print(f"\nce={ce.item():.4f} z={z.item():.4f}")
print(f"hidden.grad finite: {torch.isfinite(hidden.grad).all().item()}")
print(f"grad nonzero rows: {(hidden.grad.norm(dim=-1) > 0).float().mean().item():.4f}")
