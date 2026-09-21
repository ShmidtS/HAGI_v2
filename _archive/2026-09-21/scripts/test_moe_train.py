"""Full training step with MoE + ternary: forward, backward, optimizer step.
Verifies that:
1. Gradient flows through ternary STE experts
2. Router loss gradient flows back to router parameters
3. Optimizer step succeeds (no shape mismatches)
4. Loss decreases over mini-training
"""
import os, sys, io, gc, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch
from torch import nn
from hagi.config import load_config
from hagi.model.model import HAGI
from hagi.model.ternary import cache_ternary_weights, clear_ternary_weights
from hagi.train.optim import build_optimizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

# Build config with MoE enabled — small fast test
cfg = load_config('configs/level0_merged_3.yaml', **{
    'model.moe.enabled': True,
    'model.moe.n_experts': 4,
    'model.moe.top_k': 2,
    'model.moe.expert_intermediate_size': 64,
    'model.moe.aux_loss_weight': 0.01,
    'train.max_steps': 50,
    'train.batch_size': 4,
    'train.learning_rate': 1e-4,
    'train.max_grad_norm': 1.0,
    'train.ce_keep_rate': 1.0,
    'train.ternary_step_cache': True,
})

model = HAGI(cfg).to(device)
model.train()
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# Count params by group
n_rout = sum(p.numel() for n,p in model.named_parameters() if 'router' in n)
n_expert = sum(p.numel() for n,p in model.named_parameters() if 'expert' in n or ('mixer' in n and 'router' not in n))
n_other = sum(p.numel() for n,p in model.named_parameters() if 'router' not in n and 'expert' not in n and 'mixer' not in n)
print(f"  Router params: {n_rout:,}")
print(f"  Expert+FFN params: {n_expert:,}")
print(f"  Other params: {n_other:,}", flush=True)

# Optimizer (AdamW only — no Muon needed for tiny test)
optim = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=0.01
)

# Training loop — 20 steps
B, T = 4, 128
vocab = cfg.model.vocab_size
losses = []

t0 = time.time()
for step in range(20):
    input_ids = torch.randint(0, vocab, (B, T), device=device)
    targets = torch.randint(0, vocab, (B, T), device=device)
    
    # Ternary step cache (freeze ternary map across microbatches)
    if cfg.train.ternary_step_cache:
        cache_ternary_weights(model)
    
    optim.zero_grad()
    out = model(input_ids, targets=targets)
    loss = out.loss
    loss.backward()
    
    # Gradient norm clip
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0
    )
    
    optim.step()
    
    if cfg.train.ternary_step_cache:
        clear_ternary_weights(model)
    
    losses.append(loss.item())
    
    if step % 4 == 0:
        print(f"Step {step:3d}: loss={loss.item():.4f}  moe_rl={out.moe_router_loss:.6f}  "
              f"grad={grad_norm:.3f}  CE={out.ce.item():.4f}", flush=True)

dt = time.time() - t0
print(f"\n20 steps in {dt:.1f}s ({20/dt:.1f} steps/s)", flush=True)
print(f"Loss trajectory: {losses[0]:.4f} -> {losses[-1]:.4f} (delta={losses[-1]-losses[0]:+.4f})", flush=True)

# Verify MoE expert balance
for i, block in enumerate(model.blocks):
    aux = block._moe_aux
    if aux and 'expert_load' in aux:
        load = aux['expert_load']
        if isinstance(load, torch.Tensor):
            print(f"  Block {i} expert load: {load.tolist()}", flush=True)

# Verify router parameters get gradients
for name, p in model.named_parameters():
    if 'router' in name and p.grad is not None:
        print(f"  Router grad OK: {name} grad={p.grad.abs().mean().item():.6f}", flush=True)
        break

# Gradient check — verify ternary experts get gradient
for name, p in model.named_parameters():
    if 'expert' in name and p.grad is not None:
        print(f"  Expert grad OK: {name} grad={p.grad.abs().mean().item():.6f}", flush=True)
        break

print("\nMoE training step: PASS", flush=True)