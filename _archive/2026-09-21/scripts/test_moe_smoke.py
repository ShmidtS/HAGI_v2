"""Sparse MoE integration smoke test: build model with MoE, run forward, collect losses.
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch
from hagi.config import load_config, MoEConfig
from hagi.model.model import HAGI
from hagi.model.moe import SparseMoE, TopKRouter, ExpertSwiGLU

# Build a tiny MoE config
cfg = load_config('configs/level0_merged_3.yaml', **{
    'model.moe.enabled': True,
    'model.moe.n_experts': 4,
    'model.moe.top_k': 2,
    'model.moe.expert_intermediate_size': 64,
    'model.moe.aux_loss_weight': 0.01,
    'train.max_steps': 10,
    'train.batch_size': 2,
})
print(f"MoE enabled: {cfg.model.moe.enabled}")
print(f"Experts={cfg.model.moe.n_experts}, top_k={cfg.model.moe.top_k}")
print(f"H={cfg.model.hidden_size}, L={cfg.model.num_layers}")

model = HAGI(cfg)
model.train()

# Forward pass with dummy data
B, T = 2, 32
input_ids = torch.randint(0, cfg.model.vocab_size, (B, T))
targets = torch.randint(0, cfg.model.vocab_size, (B, T))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
input_ids, targets = input_ids.to(device), targets.to(device)

out = model(input_ids, targets=targets)
print(f"\nForward OK: CE={out.ce.item():.4f}, Loss={out.loss.item():.4f}")
print(f"MoE router_loss={out.moe_router_loss:.6f}")
print(f"MoE expert_balance={out.moe_expert_balance:.6f}")
assert out.loss is not None and not torch.isnan(out.loss), "Loss is NaN!"
assert out.moe_router_loss > 0, "Router loss is zero!"

# Check Block MOE aux
for i, block in enumerate(model.blocks[:2]):
    aux = block._moe_aux
    if aux:
        print(f"Block {i} MoE aux: router_loss={aux.get('router_loss', 0):.6f}, balance={aux.get('expert_balance', 0):.6f}")
        print(f"  expert_load={aux.get('expert_load', None)}")

print("\nMoE integration smoke: PASS", flush=True)