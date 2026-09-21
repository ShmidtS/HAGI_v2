"""Minimal STE test on 1 expert: verify autograd + transposes work.
Expected: loss decreases, gradients non-zero, completes in <60s.
"""
import sys, os, time
os.chdir(r'C:/HAGI_v2')
sys.path.insert(0, '.')
import torch, torch.nn.functional as F
from torch import nn, autograd
import numpy as np

dev = 'cuda'
D, I = 4096, 2048
HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'

class TSTE(autograd.Function):
    @staticmethod
    def forward(ctx, w, eps=1e-8):
        s = w.abs().mean(1, keepdim=True).clamp_min(eps)
        return (w / s).clamp(-1., 1.).round() * s
    @staticmethod
    def backward(ctx, g):
        return g, None

def load_expert(il, e):
    """Returns (Wg, Wu, Wd) where Wg: (D, I), Wu: (D, I), Wd: (I, D)
       This matches the GGUF layout [rows, cols, experts]."""
    from safetensors import safe_open
    import json
    _wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
    g = {}
    for p, o, i, tag in [('gate_proj', D, I, 'g'), ('up_proj', D, I, 'u'), ('down_proj', I, D, 'd')]:
        k = f'model.language_model.layers.{il}.mlp.experts.{e}.{p}.weight'
        with safe_open(f'{HF}/{_wm[k]}', 'pt', 'cpu') as sf:
            w = sf.get_tensor(k).float()  # safetensors saves as (out, in)
            s = sf.get_tensor(k.replace('.weight', '.weight_scale_inv')).float()
        # Reshape from blockscale: weights stored as (2048, 128, 4096, 128)
        # Final shape should be (o, i) matching GGUF layout
        g[tag] = (w.reshape(s.shape[0], 128, s.shape[1], 128) * s[:, None, :, None]).reshape(o, i)
    return g['g'], g['u'], g['d']

print("Loading expert...", flush=True)
t0 = time.time()
Wg, Wu, Wd = load_expert(3, 0)
# Convert to GPU Parameters
Wg_p = nn.Parameter(Wg.to(dev))
Wu_p = nn.Parameter(Wu.to(dev))
Wd_p = nn.Parameter(Wd.to(dev))
print(f"  Wg={Wg_p.shape} Wu={Wu_p.shape} Wd={Wd_p.shape} load={time.time()-t0:.1f}s", flush=True)

# Random input matching D=4096
z = torch.randn(128, D, device=dev)  # 128 tokens, D=4096

# Teacher output (fp16)
with torch.no_grad():
    teacher = (F.silu(z @ Wg_p.T) * (z @ Wu_p.T)) @ Wd_p.T  # (128, D)
t1 = time.time()
print(f"Teacher: shape={teacher.shape} mag={teacher.norm():.3f} t={t1-t0:.1f}s", flush=True)

# Train with STE
opt = torch.optim.AdamW([Wg_p, Wu_p, Wd_p], lr=1e-4)
for st in range(5):
    t = time.time()
    opt.zero_grad()
    # STE path: ternarize then forward
    wq_g = TSTE.apply(Wg_p)
    wq_u = TSTE.apply(Wu_p)
    wq_d = TSTE.apply(Wd_p)
    student = (F.silu(z @ wq_g.T) * (z @ wq_u.T)) @ wq_d.T
    loss = F.mse_loss(student, teacher)
    loss.backward()
    gnorm = torch.cat([p.grad.flatten() for p in [Wg_p, Wu_p, Wd_p]]).norm()
    opt.step()
    print(f"  step {st}: loss={loss.item():.6e} gnorm={gnorm:.4e} dt={time.time()-t:.2f}s", flush=True)

print(f"\nDONE: total={time.time()-t0:.1f}s", flush=True)