"""Per-expert I-vector correction for TQ1: W_corrected = W_skel8 + e_i@z
where e_i is a learnable per-expert vector (I,) that acts as bias in output space.
Computed on calibration data, zero change to GGUF.
"""
import numpy as np, os, sys, io, json, time, gc, glob
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch

D, I, E = 4096, 2048, 288
bpc = D // 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

# ── Calibration Z (skel8 activations) ──
cal_z = {}
base = 'glm5_gguf/dump_moe_in_skel8'
for il in range(3, 46):
    mats = []
    for f in sorted(glob.glob(f'{base}/moe_in_L{il}_*.f32')):
        if os.path.getsize(f) < 1024: continue
        a = np.fromfile(f, dtype=np.float32); n = len(a)//D
        if n: mats.append(a[:n*D].reshape(n, D))
    if mats:
        z_full = np.concatenate(mats, 0).astype(np.float32)
        N = min(z_full.shape[0], 1024)
        cal_z[il] = torch.from_numpy(z_full[:N].T).contiguous().to(device)
        print(f"  L{il}: {N} tokens", flush=True)
print(f"Loaded {len(cal_z)} layers", flush=True)

# ── FP16 teacher ──
HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
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

# ── Decode TQ1 ──
def decode_tq1_expert(packed_flat):
    n_blk = I * D // 256
    pb = packed_flat.reshape(n_blk, 54)
    gamma = pb[:, 52:54].copy().view(np.float16).ravel().astype(np.float32)
    q_raw = pb[:, :48].astype(np.int32); h_raw = pb[:, 48:52].astype(np.int32)
    def dec(qv): return ((qv * 243 + 128) // 256) % 243
    dv = dec(q_raw); dh = dec(h_raw)
    flat = np.zeros((n_blk, 256), np.int32)
    for n, c in enumerate([81, 27, 9, 3, 1]):
        flat[:, n*32:(n+1)*32] = (dv[:, :32] // c) % 3
        flat[:, 160+n*16:160+(n+1)*16] = (dv[:, 32:] // c) % 3
    for col in range(4):
        val = dh[:, col]
        for mi, ci in enumerate([81, 27, 9, 3]):
            flat[:, 240 + mi*4 + col] = (val // ci) % 3
    return flat.astype(np.float32) - 1.0, gamma  # ternary, gamma

# ── Per-expert correction: learn e_i (I,) per expert ──
os.makedirs('glm5_pod', exist_ok=True)

for il in range(3, 46):
    if il not in cal_z:
        print(f"L{il}: no cal data, skip", flush=True)
        continue
    Z = cal_z[il]  # D x N
    t0 = time.time()
    
    for pname in ('gate', 'up'):
        W_teacher = load_W_fp16(il, range(E), pname)  # (E, I, D)
        packed_src = np.load(f'glm5_pod/tq1raw_L{il}_{pname}.npy')  # (E, I, 864)
        
        # Decode all experts
        Wq_all = np.zeros((E, I, D), dtype=np.float32)
        for e in range(E):
            tern, gamma = decode_tq1_expert(packed_src[e])
            Wq_all[e] = (tern * gamma[:, None]).reshape(I, D)
        
        Wq_t = torch.from_numpy(Wq_all).to(device)
        W_tch = W_teacher.to(device)
        
        y_ref = W_tch @ Z  # (E, I, N)
        y_q = Wq_t @ Z     # (E, I, N)
        
        # Learn per-expert I-vector correction
        # y_corrected = y_q + e_i * z_mean  (where e_i is per-expert I-dim vector)
        # Simpler: e_i is a bias on W: W_corrected = Wq + diag(e_i) [broadcast]
        # y_corrected = Wq@z + e_i * z_mean  — but z changes per token
        # Better: learn correction e_i directly in weight space
        # e_i = argmin ||(Wq + e_i * mask_z) @ Z - y_ref||
        # Still: linear in z: y_corr = Wq@z + e@z = y_q + e_i@z
        # Where e_i shapes (I, D) — no, too big
        #
        # Use projection: correction = alpha_i * (W_err_mean)  where alpha_i is per-expert scalar
        # Or just per-expert I-dim: e_i (I,) where correction = e_i * z (vector-matrix product?)
        # Wait: e_i (I,) @ z(D,N) = scalar per token. Can't reshape I-dim to (I,D) easily.
        #
        # Most practical: per-expert bias on output
        # y_corr = y_q + b_i   where b_i (I,) = argmin ||y_q+b_i - y_ref||
        # This is just: b_i = mean(y_ref - y_q, dim=1) = (E, I)
        
        bias = (y_ref - y_q).mean(dim=2)  # (E, I) — per-expert per-output-dim bias
        
        # Apply
        y_corr = y_q + bias[:, :, None]  # (E, I, N)
        mse_raw = ((y_q - y_ref) ** 2).mean(dim=(1, 2))
        mse_corr = ((y_corr - y_ref) ** 2).mean(dim=(1, 2))
        y_ref_var = (y_ref ** 2).mean(dim=(1, 2))
        
        # Per-expert quality
        rel_raw = (mse_raw / y_ref_var).mean().item()
        rel_corr = (mse_corr / y_ref_var).mean().item()
        print(f"L{il} {pname}: MSE rel {rel_raw:.4f} → {rel_corr:.4f}  (gain {(rel_raw-rel_corr)/rel_raw*100:.1f}%)", flush=True)
        
        # Save bias
        bias_np = bias.cpu().numpy().astype(np.float16)
        np.save(f'glm5_pod/dither_L{il}_{pname}.npy', bias_np)
        
        dt = time.time() - t0
        print(f"  Time: {dt:.1f}s", flush=True)
        
        del W_teacher, Wq_t, W_tch; gc.collect(); torch.cuda.empty_cache()

print("DONE", flush=True)