"""Multiple approaches for skel8 improvement — sweep on L3 gate single expert."""
import numpy as np, os, sys, io, json, time, gc, glob
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch

D, I, E = 4096, 2048, 288
R = 864; bpc = D // 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

# ── Load calibration Z ──
mats = []; base = 'glm5_gguf/dump_moe_in_skel8'
for f in sorted(glob.glob(f'{base}/moe_in_L3_*.f32')):
    a = np.fromfile(f, dtype=np.float32); n = len(a)//D
    if n: mats.append(a[:n*D].reshape(n, D))
z = np.concatenate(mats, 0).astype(np.float32)
N = min(z.shape[0], 1024)
Z = torch.from_numpy(z[:N].T).contiguous().to(device)  # D x N
print(f"Z: {Z.shape}", flush=True)

# ── Load teacher (e0 gate) ──
HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
from safetensors import safe_open
key = 'model.language_model.layers.3.mlp.experts.0.gate_proj.weight'
with safe_open(f'{HF}/{_wm[key]}', 'pt', 'cpu') as sf:
    W_t = sf.get_tensor(key).to(torch.float32)
    s_t = sf.get_tensor(key.replace('.weight', '.weight_scale_inv'))
    W_f = (W_t.reshape(s_t.shape[0], 128, s_t.shape[1], 128)
           * s_t[:, None, :, None].to(torch.float32)).reshape(I, D)
W_fp16 = W_f.to(device)

# ── Decode skel8 ──
from gguf import GGUFReader
r = GGUFReader('glm5_gguf/glm5-ternary-skel8.gguf')
for t in r.tensors:
    if t.name == 'blk.3.ffn_gate_exps.weight':
        packed = np.array(t.data[0], copy=True)
        break

n_blocks = I * D // 256
pb = packed.reshape(n_blocks, 54)
gamma_init = pb[:, 52:54].copy().view(np.float16).ravel().astype(np.float32)
q_raw = pb[:, :48].astype(np.int32); h_raw = pb[:, 48:52].astype(np.int32)
def dec(qv): return ((qv * 243 + 128) // 256) % 243
dv = dec(q_raw); dh = dec(h_raw)
flat = np.zeros((n_blocks, 256), np.int32)
for n, c in enumerate([81, 27, 9, 3, 1]):
    flat[:, n*32:(n+1)*32] = (dv[:, :32] // c) % 3
    flat[:, 160+n*16:160+(n+1)*16] = (dv[:, 32:] // c) % 3
for col in range(4):
    val = dh[:, col]
    for mi, ci in enumerate([81, 27, 9, 3]):
        flat[:, 240 + mi*4 + col] = (val // ci) % 3
tern_init = flat.astype(np.float32) - 1.0  # {-1,0,+1}

y_ref = W_fp16 @ Z  # (I, N)
ls = (y_ref ** 2).mean().detach()
norm_ref = (y_ref ** 2).mean().sqrt()

def pack_tq1(ternary, gamma):
    flat = np.clip((ternary + 1).astype(np.int32), 0, 2)
    n_blocks = ternary.shape[0]
    qs = np.zeros((n_blocks, 48), np.int32)
    qh = np.zeros((n_blocks, 4), np.int32)
    for n, c in enumerate([81, 27, 9, 3, 1]):
        qs[:, :32] += flat[:, n*32:(n+1)*32] * c
        qs[:, 32:] += flat[:, 160+n*16:160+(n+1)*16] * c
    for m, c in enumerate([81, 27, 9, 3]):
        qh += flat[:, 240+m*4:244+m*4] * c
    def comp(qq): return ((qq.astype(np.uint32) * 256 + 242) // 243).astype(np.uint8)
    return np.concatenate([comp(qs), comp(qh),
                           gamma.astype(np.float16).view(np.uint8).reshape(-1, 2)], axis=1)

# ════════════════════════════════════════════════════
# METHOD 1: Post-multiply per-expert (keep gamma fixed)
# ════════════════════════════════════════════════════
print("\n=== M1: Post-multiply alpha ===", flush=True)
tern_t = torch.from_numpy(tern_init).to(device)
g_t = torch.from_numpy(gamma_init).to(device)
Wq_skel8 = (tern_t * g_t[:, None]).reshape(I, D)

alpha = torch.tensor(1.0, device=device, requires_grad=True)
opt = torch.optim.AdamW([alpha], lr=1e-2)
for step in range(100):
    y_q = (Wq_skel8 @ Z) * torch.clamp(alpha, 0.1, 3.0)
    loss = ((y_q - y_ref) ** 2).mean() / ls
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 20 == 0:
        print(f"  s{step}: loss={loss.item():.4f} α={alpha.item():.4f}", flush=True)

alpha_opt = alpha.detach().item()
y_m1 = (Wq_skel8 @ Z) * alpha_opt
mse_m1 = ((y_m1 - y_ref) ** 2).mean() / ls
print(f"  FINAL MSE={mse_m1.item():.4f} α={alpha_opt:.4f}", flush=True)

# ════════════════════════════════════════════════════
# METHOD 2: Norm-constrained gamma + ternary (STE)
# ════════════════════════════════════════════════════
print("\n=== M2: Norm-constrained STE ===", flush=True)
x = torch.from_numpy(tern_init).to(device).clone().detach().requires_grad_(True)
g = torch.from_numpy(gamma_init).to(device).clone().detach().requires_grad_(True)
opt = torch.optim.AdamW([x, g], lr=1e-2)
lam = 0.5

for step in range(150):
    q_hard = torch.clamp(torch.round(x), -1, 1)
    q = q_hard.detach() + x - x.detach()
    gc = torch.clamp(g, 0.001, None)
    Wq = (q * gc[:, None]).reshape(I, D)
    y_q = Wq @ Z
    mse = ((y_q - y_ref) ** 2).mean() / ls
    norm_q = (y_q ** 2).mean().sqrt()
    norm_loss = (norm_q - norm_ref).abs()
    loss = mse + lam * norm_loss
    opt.zero_grad(); loss.backward(); opt.step()
    x.data.clamp_(-1.5, 1.5)
    if step % 30 == 0:
        flips = (q_hard != tern_t).float().mean().item() * 100
        print(f"  s{step}: mse={mse.item():.4f} nerr={norm_loss.item():.4f} γ={gc.mean().item():.4f} flip={flips:.1f}%", flush=True)

q_f = torch.clamp(torch.round(x), -1, 1)
g_f = torch.clamp(g, 0.001, None)
Wq_ste = (q_f * g_f[:, None]).reshape(I, D)
y_m2 = Wq_ste @ Z
mse_m2 = ((y_m2 - y_ref) ** 2).mean() / ls
print(f"  FINAL MSE={mse_m2.item():.4f} γ={g_f.mean().item():.4f} flip={(q_f!=tern_t).float().mean().item()*100:.1f}%", flush=True)

# ════════════════════════════════════════════════════
# METHOD 3: Coordinate descent (keep gamma, flip elements greedily)
# ════════════════════════════════════════════════════
print("\n=== M3: Coordinate descent (greedy per-element flip) ===", flush=True)
tern_cd = tern_init.copy()
Wq_base = (torch.from_numpy(tern_cd).to(device) * g_t[:, None]).reshape(I, D)
y_base = Wq_base @ Z
best_mse = ((y_base - y_ref) ** 2).mean() / ls
print(f"  baseline: mse={best_mse.item():.4f}", flush=True)

# z_blocks: (bpc, 256, N) — each block's calibration slice
z_blocks = Z.reshape(bpc, 256, N)  # 16 x 256 x N
golden = gamma_init.copy().reshape(-1)

for block_idx in range(16):  # just first 16 blocks (256 elements each)
    bi = block_idx  # absolute block index (expert 0)
    old_val = tern_cd[bi, :].copy()
    zb = z_blocks[block_idx % bpc]  # 256 x N
    # For each element in this block, try flipping ±1 ↔ 0
    for elem in range(256):
        if tern_cd[bi, elem] == 0:
            continue  # can't improve from 0 to ±1 without knowing which sign
        # Try flip: ±1 → 0  (halve the ternary)
        old_e = tern_cd[bi, elem]
        tern_cd[bi, elem] = 0
        # Quick delta: delta_y = -old_e * gamma * z
        # ΔMSE = 2*old_e*γ*(z·r) - (old_e*γ)²*(z·z)
        # where r = y_ref - y_q (residual at this position)
        # But computing full residual is expensive. Let's do batch on GPU.
        Wq = torch.from_numpy(tern_cd).to(device) * g_t[:, None]
        y_q = Wq.reshape(I, D) @ Z
        new_mse = ((y_q - y_ref) ** 2).mean().item()
        if new_mse < best_mse:
            best_mse = new_mse
        else:
            tern_cd[bi, elem] = old_e  # revert

print(f"  CD result: mse={best_mse:.4f}", flush=True)

# ════════════════════════════════════════════════════
# METHOD 4: Per-block gamma with fixed ternary pattern + norm constraint
# ════════════════════════════════════════════════════
print("\n=== M4: Per-block gamma (ternary fixed) + norm constraint ===", flush=True)
g4 = torch.from_numpy(gamma_init).to(device).clone().detach().requires_grad_(True)
opt = torch.optim.AdamW([g4], lr=1e-2)
for step in range(150):
    gc = torch.clamp(g4, 0.001, None)
    Wq = (tern_t * gc[:, None]).reshape(I, D)
    y_q = Wq @ Z
    mse = ((y_q - y_ref) ** 2).mean() / ls
    norm_q = (y_q ** 2).mean().sqrt()
    norm_loss = (norm_q - norm_ref).abs()
    loss = mse + lam * norm_loss
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 30 == 0:
        print(f"  s{step}: mse={mse.item():.4f} nerr={norm_loss.item():.4f} γ={gc.mean().item():.4f}", flush=True)

g4_f = torch.clamp(g4, 0.001, None)
Wq_m4 = (tern_t * g4_f[:, None]).reshape(I, D)
y_m4 = Wq_m4 @ Z
mse_m4 = ((y_m4 - y_ref) ** 2).mean() / ls
norm_m4 = (y_m4 ** 2).mean().sqrt()
print(f"  FINAL MSE={mse_m4.item():.4f} norm_ratio={norm_m4/norm_ref:.4f} γ={g4_f.mean().item():.4f}", flush=True)

# ════════════════════════════════════════════════════
# METHOD 5: Ternary flip with norm constraint (gamma fixed)
# ════════════════════════════════════════════════════
print("\n=== M5: Ternary flip (gamma fixed) norm-constrained ===", flush=True)
x5 = torch.from_numpy(tern_init).to(device).clone().detach().requires_grad_(True)
opt = torch.optim.AdamW([x5], lr=3e-3)
for step in range(200):
    q_hard = torch.clamp(torch.round(x5), -1, 1)
    q = q_hard.detach() + x5 - x5.detach()
    Wq = (q * g_t[:, None]).reshape(I, D)
    y_q = Wq @ Z
    mse = ((y_q - y_ref) ** 2).mean() / ls
    norm_q = (y_q ** 2).mean().sqrt()
    norm_loss = (norm_q - norm_ref).abs()
    loss = mse + lam * norm_loss
    opt.zero_grad(); loss.backward(); opt.step()
    x5.data.clamp_(-1.5, 1.5)
    if step % 40 == 0:
        flips = (q_hard != tern_t).float().mean().item() * 100
        print(f"  s{step}: mse={mse.item():.4f} nerr={norm_loss.item():.4f} flip={flips:.1f}%", flush=True)

q5 = torch.clamp(torch.round(x5), -1, 1)
Wq_m5 = (q5 * g_t[:, None]).reshape(I, D)
y_m5 = Wq_m5 @ Z
mse_m5 = ((y_m5 - y_ref) ** 2).mean() / ls
norm_m5 = (y_m5 ** 2).mean().sqrt()
print(f"  FINAL MSE={mse_m5.item():.4f} norm_ratio={norm_m5/norm_ref:.4f} flip={(q5!=tern_t).float().mean().item()*100:.1f}%", flush=True)

# ════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"SUMMARY for L3 gate e0:", flush=True)
print(f"{'='*60}", flush=True)
print(f"Baseline (skel8):     MSE={((Wq_skel8@Z-y_ref)**2).mean()/ls:.4f}", flush=True)

# M1: post-multiply
y_m1_v = (Wq_skel8 @ Z) * alpha_opt
bl_m1 = ((Wq_skel8 @ Z - y_ref) ** 2).mean() / ls
m1_v = ((y_m1_v - y_ref) ** 2).mean() / ls
print(f"M1 (post-α={alpha_opt:.4f}):        MSE={m1_v.item():.4f}  gain={(bl_m1.item()-m1_v.item())/bl_m1.item()*100:.1f}%  γ=SKEL8  norm={(y_m1_v**2).mean().sqrt()/norm_ref:.3f}", flush=True)

print(f"M2 (STE+norm):                MSE={mse_m2.item():.4f}  gain={(bl_m1.item()-mse_m2.item())/bl_m1.item()*100:.1f}%  γ={g_f.mean().item():.4f}  flip={(q_f!=tern_t).float().mean().item()*100:.1f}%  norm={norm_m2:.3f}", flush=True)

print(f"M4 (γ-only+norm):             MSE={mse_m4.item():.4f}  gain={(bl_m1.item()-mse_m4.item())/bl_m1.item()*100:.1f}%  γ={g4_f.mean().item():.4f}  norm={(y_m4**2).mean().sqrt()/norm_ref:.3f}", flush=True)

print(f"M5 (ternary-only+norm):       MSE={mse_m5.item():.4f}  gain={(bl_m1.item()-mse_m5.item())/bl_m1.item()*100:.1f}%  γ=SKEL8  flip={(q5!=tern_t).float().mean().item()*100:.1f}%  norm={norm_m5/norm_ref:.3f}", flush=True)

# Decide best approach
# M1 (post-multiply) is SAFEST — doesn't change GGUF at all
print(f"\n→ M1 (post-α) is safest: no GGUF change, zero cascade disruption", flush=True)
print(f"→ M5 (ter only) is next: same gamma, norm preserved", flush=True)
print(f"→ M2 (STE full) is risky: gamma changed 4x", flush=True)