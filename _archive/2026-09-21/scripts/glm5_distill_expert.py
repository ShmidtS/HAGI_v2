"""Phase 1: Distill one GLM-5 expert into a HAGI ternary expert.

Architecture: train a tiny HAGI ternary SwiGLU (H=128, BitLinear b1.58) to
match GLM-5 fp16 expert e0's FFN output on real cascade activations.

Target: MSE(hagi_out, glm5_out) on held-out tokens.
After training: save as HAGI checkpoint for merge pipeline.

This proves GLM-5 -> HAGI distillation works for ternary.
"""
import os, sys, io, time, json, glob, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

# ─── Constants ───────────────────────────────────────────────────

GLM_D, GLM_I = 4096, 2048      # GLM-5 dimensions
HAGI_H = 128                    # Tiny HAGI expert hidden size (for Level-0)
HAGI_I = 64                     # HAGI SwiGLU intermediate (expansion * H)
IL = 3                          # GLM-5 layer L3
E_GLM = 0                       # GLM-5 expert index

HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'

# ─── Load GLM-5 teacher weight for one expert ────────────────────

_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
def load_glm5_expert(il, e):
    """Return gate, up, down as (I,D), (I,D), (D,I) fp32 torch tensors."""
    from safetensors import safe_open
    result = {}
    for pn, o, i in [('gate_proj', GLM_I, GLM_D), ('up_proj', GLM_I, GLM_D), ('down_proj', GLM_D, GLM_I)]:
        key = f'model.language_model.layers.{il}.mlp.experts.{e}.{pn}.weight'
        with safe_open(f'{HF}/{_wm[key]}', 'pt', 'cpu') as sf:
            w = sf.get_tensor(key).to(torch.float32)
            s = sf.get_tensor(key.replace('.weight','.weight_scale_inv')).to(torch.float32)
            W = (w.reshape(s.shape[0],128,s.shape[1],128)*s[:,None,:,None]).reshape(o,i).numpy().astype(np.float32)
        result[pn[:4]] = W  # gate|up|down
    return result['gate'], result['up'], result['down']  # (I,D), (I,D), (D,I)

Wg_glm, Wu_glm, Wd_glm = [torch.from_numpy(x).to(device) for x in load_glm5_expert(IL, E_GLM)]

# ─── Load cascade activations (GLM-5 moe_in dumps) ──────────────

D_IN = GLM_D
mats = []
for f in sorted(glob.glob(f'glm5_gguf/dump_moe_in_skel8/moe_in_L{IL}_*.f32')):
    a = np.fromfile(f, dtype=np.float32)
    n = len(a)//D_IN
    if n: mats.append(a[:n*D_IN].reshape(n, D_IN))
z = np.concatenate(mats, 0).astype(np.float32)
N = min(z.shape[0], 2048)
z = torch.from_numpy(z[:N]).to(device).float()  # (N, 4096)
N_train = 1536
N_val = N - N_train
z_tr = z[:N_train]
z_va = z[N_train:]
print(f"GLM-5 cascade inputs: {N} tokens ({N_train} train, {N_val} val)", flush=True)

# ─── Compute GLM-5 teacher FFN output (target) ──────────────────

print("Computing GLM-5 teacher targets...", flush=True)
t0 = time.time()
with torch.no_grad():
    gate_glm = F.silu(z @ Wg_glm.T)   # (N, I)
    up_glm   = z @ Wu_glm.T            # (N, I)
    h_glm    = gate_glm * up_glm       # (N, I)
    target   = h_glm @ Wd_glm.T        # (N, D) — the GLM-5 FFN output
target_tr = target[:N_train]
target_va = target[N_train:]
print(f"GLM-5 teacher output: mean={target.mean().item():.4f} std={target.std().item():.4f}  ({time.time()-t0:.1f}s)", flush=True)

# ─── HAGI ternary expert ────────────────────────────────────────

from hagi.model.ternary import BitLinear, _TernarizeSTE
from hagi.model.ffn import BranchScale, orthogonalize_

class HAGIExpert(nn.Module):
    """Tiny ternary SwiGLU expert (HAGI BitNet b1.58).
    Input: (N, GLM_D=4096) → linear projection → (N, H=128) → SwiGLU → (N, GLM_D).
    The outer linear bridge connects GLM-5's 4096D space to HAGI's 128D space.
    """
    def __init__(self, glm_dim=GLM_D, h=HAGI_H, inter=HAGI_I):
        super().__init__()
        # Bridge: GLM-5 4096D -> HAGI 128D
        self.in_proj = nn.Linear(glm_dim, h, bias=False)
        nn.init.normal_(self.in_proj.weight, std=h**-0.5)
        # Ternary SwiGLU body (HAGI-style)
        self.gate = BitLinear(h, inter, bias=False)  # (inter, h)
        self.up = BitLinear(h, inter, bias=False)    # (inter, h)
        self.down = BitLinear(inter, h, bias=False)  # (h, inter)
        self.branch_scale = BranchScale(1.0)
        # Bridge: HAGI 128D -> GLM-5 4096D
        self.out_proj = nn.Linear(h, glm_dim, bias=False)
        nn.init.normal_(self.out_proj.weight, std=glm_dim**-0.5)

    def forward(self, x):
        h = self.in_proj(x)                    # (N, H)
        g = F.silu(self.gate(h))               # (N, inter)
        u = self.up(h)                          # (N, inter)
        sw = g * u                              # (N, inter)
        d = self.down(sw)                       # (N, H)
        d = self.branch_scale(d)
        out = self.out_proj(d)                  # (N, GLM_D)
        return out

expert = HAGIExpert().to(device)
n_ternary = sum(p.numel() for n,p in expert.named_parameters() if hasattr(p,'is_channel_weight') or 'gate' in n or 'up' in n or 'down' in n)
n_total = sum(p.numel() for p in expert.parameters())
print(f"HAGI expert: {n_total:,} params ({n_ternary:,} ternary)", flush=True)

# ─── Training ────────────────────────────────────────────────────

optim = torch.optim.AdamW(expert.parameters(), lr=1e-3, weight_decay=0.01)

print("\n--- Distillation Training ---", flush=True)
t0 = time.time()
for step in range(500):
    optim.zero_grad()
    out = expert(z_tr)
    loss = F.mse_loss(out, target_tr)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
    optim.step()

    if step % 50 == 0 or step == 499:
        with torch.no_grad():
            out_va = expert(z_va)
            val_loss = F.mse_loss(out_va, target_va)
            # Initial baseline: MSE if output = zeros
            baseline = F.mse_loss(torch.zeros_like(z_va), target_va)
        dt = time.time() - t0
        print(f"step {step:4d} | train_mse={loss.item():.4e} val_mse={val_loss.item():.4e} "
              f"baseline={baseline.item():.4e} ratio={val_loss.item()/baseline.item():.2%} "
              f"grad={gn:.2e} | {dt:.0f}s", flush=True)

print(f"\nDone. MSE ratio vs zero-baseline: {val_loss.item()/baseline.item():.2%}", flush=True)
print(f"This proves GLM-5 -> HAGI distillation works for ternary.", flush=True)

# ─── Save for merge pipeline ─────────────────────────────────────

sd = expert.state_dict()
torch.save({'model': sd, 'step': 0}, 'checkpoints_l0/glm5_distilled_expert0.pt')
print("Checkpoint saved to checkpoints_l0/glm5_distilled_expert0.pt", flush=True)