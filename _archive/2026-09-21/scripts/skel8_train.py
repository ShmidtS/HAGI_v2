"""Skel8-TQ1 Training — Straight-Through Estimator.
Per expert: parameterize continuous pre-round values + gamma,
train via STE gradient to minimize output MSE vs fp16 teacher.
https://arxiv.org/abs/1906.05635 (Esser et al, LSQ)
"""
import numpy as np, os, sys, io, json, time, gc, glob, shutil
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch

D, I, E = 4096, 2048, 288
bpc = D // 256  # 16
R = 54 * D // 256  # 864
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

# ── 1. Calibration activations (from fresh skel8 dump) ──
cal_z = {}
base = 'glm5_gguf/dump_moe_in_skel8'
for il in range(3, 46):
    mats = []
    for f in sorted(glob.glob(f'{base}/moe_in_L{il}_*.f32')):
        if os.path.getsize(f) < 1024: continue
        a = np.fromfile(f, dtype=np.float32); n = len(a) // D
        if n: mats.append(a[:n * D].reshape(n, D))
    if mats:
        z_full = np.concatenate(mats, axis=0).astype(np.float32)
        N = min(z_full.shape[0], 1024)
        cal_z[il] = torch.from_numpy(z_full[:N].T).contiguous().to(device)  # D x N
        print(f"  L{il}: {N} tokens", flush=True)
print(f"Loaded {len(cal_z)} layers", flush=True)

# ── 2. FP16 teacher weights ──
HF = r'//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']

def load_W_fp16(il, exprs, p):
    from safetensors import safe_open
    shard_map = defaultdict(list)
    kt = f'model.language_model.layers.{il}.mlp.experts.{{e}}.{p}_proj.weight'
    for e in exprs: shard_map[_wm[kt.format(e=e)]].append(e)
    res = torch.empty((len(exprs), I, D), dtype=torch.float32, device='cpu')
    for sh, se in shard_map.items():
        with safe_open(f'{HF}/{sh}', 'pt', 'cpu') as sf:
            for e in se:
                key = kt.format(e=e)
                s_t = sf.get_tensor(key.replace('.weight', '.weight_scale_inv'))
                W_t = (sf.get_tensor(key).to(torch.float32)
                       .reshape(s_t.shape[0], 128, s_t.shape[1], 128)
                       * s_t[:, None, :, None].to(torch.float32))
                res[exprs.index(e)] = W_t.reshape(I, D)
    return res.to(device)

# ── 3. TQ1 Pack/Decode helpers ──
def decode_tq1_block(packed_flat):
    """packed_flat: (n_blocks, 54) → ternary: (n_blocks, 256), gamma: (n_blocks,)"""
    gamma = packed_flat[:, 52:54].copy().view(np.float16).ravel().astype(np.float32)
    q_raw = packed_flat[:, :48].astype(np.int32)
    h_raw = packed_flat[:, 48:52].astype(np.int32)
    def dec(qv): return ((qv * 243 + 128) // 256) % 243
    dv = dec(q_raw); dh = dec(h_raw)
    flat = np.zeros((packed_flat.shape[0], 256), dtype=np.int32)
    for n, c in enumerate([81, 27, 9, 3, 1]):
        flat[:, n*32:(n+1)*32] = (dv[:, :32] // c) % 3
        flat[:, 160+n*16:160+(n+1)*16] = (dv[:, 32:] // c) % 3
    for col in range(4):
        val = dh[:, col]
        for mi, ci in enumerate([81, 27, 9, 3]):
            flat[:, 240 + mi*4 + col] = (val // ci) % 3
    return flat.astype(np.float32) - 1.0, gamma  # ternary {-1,0,+1}, gamma

def pack_tq1_block(ternary, gamma):
    """ternary: (n_blocks, 256), gamma: (n_blocks,) → packed uint8 (n_blocks, 54)"""
    flat = np.clip((ternary + 1).astype(np.int32), 0, 2)  # {-1→0, 0→1, +1→2}
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

# ── 4. Per-expert STE Training ──
def train_expert(W_fp16, packed_init, Z, steps=120, lr=1e-2):
    """Train one expert.
    W_fp16: (I, D) fp32 teacher
    packed_init: (I, R) packed TQ1 (skel8 initial)
    Z: (D, N) calibration activations
    Returns: (I, R) trained packed TQ1
    """
    N = Z.shape[1]
    I_, D_ = W_fp16.shape
    n_blocks = I_ * D_ // 256
    
    y_ref = W_fp16 @ Z  # (I, N)
    loss_scale = (y_ref ** 2).mean().detach()  # normalize
    
    # Decode initial state → blocks
    pb = packed_init.reshape(n_blocks, 54)
    tern_init, gamma_init = decode_tq1_block(pb)  # (n_blocks, 256), (n_blocks,)
    
    tern_t = torch.from_numpy(tern_init).to(device)  # (n_blocks, 256)
    gamma_t = torch.from_numpy(gamma_init).to(device)  # (n_blocks,)
    
    # Parameters: learnable pre-round values + gamma
    # Initialized from current ternary { -1,0,+1 } — start near convergence
    x = tern_t.clone().detach().to(device)  # (n_blocks, 256)
    x.requires_grad_(True)
    
    g = gamma_t.clone().detach().to(device)  # (n_blocks,)
    g.requires_grad_(True)
    
    opt = torch.optim.AdamW([x, g], lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    
    best_loss = float('inf')
    best_packed = None
    
    for step in range(steps):
        # ── Forward ──
        # STE: round() in forward, identity gradient in backward
        q_hard = torch.clamp(torch.round(x), -1, 1)  # (n_blocks, 256) ternary values
        # STE: combine hard output with soft gradient (straight-through)
        q = q_hard.detach() + x - x.detach()  # (n_blocks, 256)
        
        # gamma is already differentiable — no STE needed
        gamma_cl = torch.clamp(g, 0.001, None)  # prevent zero
        Wq_blocks = q * gamma_cl[:, None]  # (n_blocks, 256)
        Wq = Wq_blocks.reshape(I_, D_)
        
        y_q = Wq @ Z  # (I, N)
        loss = ((y_q - y_ref) ** 2).mean() / loss_scale
        
        opt.zero_grad()
        loss.backward()
        
        # Clamp gamma gradients for stability
        if g.grad is not None:
            g.grad.data.clamp_(-0.2, 0.2)
        
        opt.step()
        sched.step()
        
        # Clamp x to [-1.5, 1.5] — beyond that ternary round doesn't change
        x.data.clamp_(-1.5, 1.5)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            with torch.no_grad():
                q_best = torch.clamp(torch.round(x), -1, 1)
                g_best = torch.clamp(g, 0.001, None)
                t_best = q_best.cpu().numpy()
                g_best_np = g_best.cpu().numpy()
                best_packed = pack_tq1_block(t_best, g_best_np).reshape(I_, R)
        
        if step % 30 == 0 or step == steps - 1:
            nz = (q_hard.abs() > 0.1).float().mean().item()
            print(f"    s{step:3d} loss={loss.item():.4f} nz={nz:.3f} γ={gamma_cl.mean().item():.4f}", flush=True)
    
    return best_packed

# ── 5. Run ──
os.makedirs('glm5_pod', exist_ok=True)

for il in range(3, 46):
    if il not in cal_z:
        print(f"L{il}: no cal data, copy skel8 → tq1ste", flush=True)
        for pname in ('gate','up'):
            shutil.copy2(f'glm5_pod/tq1raw_L{il}_{pname}.npy',
                         f'glm5_pod/tq1ste_L{il}_{pname}.npy')
        continue
    
    Z = cal_z[il]
    t0 = time.time()
    
    for pname in ('gate', 'up'):
        W_teacher = load_W_fp16(il, range(E), pname)  # (E, I, D)
        packed_src = np.load(f'glm5_pod/tq1raw_L{il}_{pname}.npy')  # (E, I, R)
        result = np.zeros_like(packed_src)
        
        for e_idx in range(E):
            print(f"L{il} {pname} e{e_idx:3d}:", flush=True)
            result[e_idx] = train_expert(W_teacher[e_idx], packed_src[e_idx], Z)
        
        np.save(f'glm5_pod/tq1ste_L{il}_{pname}.npy', result)
        dt = time.time() - t0
        print(f"  L{il} {pname}: {E} experts done in {dt:.0f}s", flush=True)
        
        del W_teacher; gc.collect(); torch.cuda.empty_cache()

print("DONE", flush=True)