"""Copy DeepSeek expert blocks into a GROWING HAGI model (black-box).

The new model is built on HAGI's architecture: ternary SwiGLU experts
(``FeedForward``, H=128, inter=192, BitNet b1.58) that grow by gluing small
blocks — the hidden stream is ``n_blocks * H`` and each block sees its own
slice (block-diagonal, exactly HAGI's merge). We copy the WORK of DeepSeek's
orthogonal experts (black-box: feed input, get output) into this model,
starting from the smallest (1 block) and growing (1 -> 2 -> 4 -> ...).

Only the minimal DeepSeek expert blocks are loaded (a few tensors from
lossless_layers) — no model skeleton, no router, no forward pass. The
floating-point in/out projections are the distillation adapters that bridge
DeepSeek's 4096-dim signal to HAGI's 128-dim hidden stream; the FeedForward
blocks themselves are the HAGI circuit (ternary b1.58).

Usage:
    python scripts/dsv4_distill_hagi.py [layer] [max_experts] [steps] [batch]
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stub_import_tf  # noqa: F401
import dsv4_experts as de
from hagi.model.ffn import FeedForward

LOSSLESS = "C:/HAGI_v2/lossless_layers"
SWIGLU_LIMIT = 10.0
HAGI_H = 128
HAGI_INTER = 192
GROWTH = (1, 2, 4, 8, 16, 32)


def ffn(x, w1, w2, w3):
    """DeepSeek routed-expert SwiGLU (the teacher black box)."""
    gate = (x @ w1.T).clamp(max=SWIGLU_LIMIT)
    up = (x @ w3.T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return (F.silu(gate) * up) @ w2.T


def load_expert(fp, li, k):
    base = f"layers.{li}.ffn.experts.{k}"
    with safe_open(fp, framework="pt", device="cuda") as f:
        return (
            de.dequant_fp4(f.get_tensor(f"{base}.w1.weight"), f.get_tensor(f"{base}.w1.scale")),
            de.dequant_fp4(f.get_tensor(f"{base}.w2.weight"), f.get_tensor(f"{base}.w2.scale")),
            de.dequant_fp4(f.get_tensor(f"{base}.w3.weight"), f.get_tensor(f"{base}.w3.scale")),
        )


class HAGIGlue(nn.Module):
    """Growing HAGI model: n_blocks ternary experts glued block-diagonally.

    ``in_proj`` maps DeepSeek's 4096-dim signal to the hidden stream
    ``n_blocks * H``; each HAGI ``FeedForward`` (ternary SwiGLU) processes its
    own H-slice; ``out_proj`` maps back. Growing ``n_blocks`` grows the hidden
    dimension by gluing small experts — the HAGI growth principle.
    """

    def __init__(self, teacher_dim: int, hidden: int, inter: int, n_blocks: int):
        super().__init__()
        self.n_blocks = n_blocks
        self.in_proj = nn.Linear(teacher_dim, n_blocks * hidden, bias=False)
        self.blocks = nn.ModuleList(
            [FeedForward(hidden, inter, use_ternary=True, init_orthogonal=True)
             for _ in range(n_blocks)]
        )
        self.out_proj = nn.Linear(n_blocks * hidden, teacher_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x).view(-1, self.n_blocks, HAGI_H)
        outs = [self.blocks[i](h[:, i]) for i in range(self.n_blocks)]
        h = torch.stack(outs, dim=1).reshape(-1, self.n_blocks * HAGI_H)
        return self.out_proj(h)


def distill(student, x, y, steps, lr=1e-3, bs=2048, log=50):
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    n = x.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), device=x.device)
        loss = F.mse_loss(student(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log and (step + 1) % log == 0:
            print(f"    step {step + 1}/{steps}: loss={float(loss.detach()):.6f}", flush=True)


def main() -> int:
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
    growth = [g for g in GROWTH if g <= max_n]

    print(f"layer {layer}, growth {growth}, {steps} steps, batch {batch}", flush=True)

    # 1. Load ONLY the minimal DeepSeek expert blocks (black boxes).
    fp = os.path.join(LOSSLESS, f"layers_{layer}_ffn.safetensors")
    print(f"loading {max_n} DeepSeek expert blocks from {os.path.basename(fp)}...", flush=True)
    t0 = time.time()
    experts = [load_expert(fp, layer, k) for k in range(max_n)]
    print(f"  {max_n} experts loaded in {time.time() - t0:.1f}s", flush=True)

    # 2. Feed a test input (Gaussian signal — the black-box probe).
    x = torch.randn(batch, de.DIM, device="cuda")
    print(f"Gaussian test signal [{batch}, {de.DIM}]", flush=True)

    # 3. Input -> output for each expert.
    expert_out = [ffn(x, *e) for e in experts]

    # 4. Calibrate: FIXED teacher (one expert), growing HAGI student.
    base_out = expert_out[0]
    print(f"\n{'K':>3} | {'H stream':>8} | {'params':>9} | {'teacher E':>9} | "
          f"{'MSE':>10} | {'residual %':>10}")
    print("-" * 68, flush=True)

    for n_blocks in growth:
        yg = base_out
        teacher_energy = float((yg.float() ** 2).mean())

        student = HAGIGlue(de.DIM, HAGI_H, HAGI_INTER, n_blocks).cuda()
        n_params = sum(p.numel() for p in student.parameters())
        t0 = time.time()
        distill(student, x, yg, steps)
        with torch.no_grad():
            mse = float(F.mse_loss(student(x), yg))
        residual_pct = mse / max(teacher_energy, 1e-9) * 100
        print(f"{n_blocks:>3} | {n_blocks * HAGI_H:>8} | {n_params / 1e6:>8.1f}M | "
              f"{teacher_energy:>9.4f} | {mse:>10.6f} | {residual_pct:>9.1f}%",
              flush=True)
        if residual_pct < 1.0:
            print(f"\n  -> <1% reached at K={n_blocks} blocks (H={n_blocks * HAGI_H}).", flush=True)
        del student, yg
        torch.cuda.empty_cache()

    print("\nHAGI circuit per block: ternary SwiGLU FeedForward (H=128, inter=192).", flush=True)
    print("'residual %' = fraction of the teacher's energy the growing HAGI model", flush=True)
    print("failed to absorb. Gluing more blocks grows the hidden stream and lowers it.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
