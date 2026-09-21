"""Train L3 gate experts: ternary + norm constraint, then keep gamma=skel8 + global alpha.
Build minimal GGUF and test generation."""
import numpy as np, os, sys, io, json, time, gc, glob
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch, shutil

D, I, E = 4096, 2048, 288; R = 864; bpc = D // 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

# ── Load Z ──
mats = []; base = 'glm5_gguf/dump_moe_in_skel8'
for f in sorted(glob.glob(f'{base}/moe_in_L3_*.f32')):
    a = np.fromfile(f, dtype=np.float32); n = len(a)//D
    if n: mats.append(a[:n*D].reshape(n, D))
z = np.concatenate(mats, 0).astype(np.float32)
N = min(z.shape[0], 1024)
Z = torch.from_numpy(z[:N].T).contiguous().to(device)

# ── Teacher weights ──
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

W_fp16 = load_W_fp16(3, range(E), 'gate')  # (E, I, D)

# ── Decode TQ1 ──
def decode_tq1_expert(packed_flat):
    n_blocks = I * D // 256
    pb = packed_flat.reshape(n_blocks, 54)
    gamma = pb[:, 52:54].copy().view(np.float16).ravel().astype(np.float32)
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
    return flat.astype(np.float32) - 1.0, gamma

def pack_tq1_expert(ternary, gamma):
    flat = np.clip((ternary + 1).astype(np.int32), 0, 2)
    n_blocks = I * D // 256
    qs = np.zeros((n_blocks, 48), np.int32)
    qh = np.zeros((n_blocks, 4), np.int32)
    for n, c in enumerate([81, 27, 9, 3, 1]):
        qs[:, :32] += flat[:, n*32:(n+1)*32] * c
        qs[:, 32:] += flat[:, 160+n*16:160+(n+1)*16] * c
    for m, c in enumerate([81, 27, 9, 3]):
        qh += flat[:, 240+m*4:244+m*4] * c
    def comp(qq): return ((qq.astype(np.uint32) * 256 + 242) // 243).astype(np.uint8)
    return np.concatenate([comp(qs), comp(qh), gamma.astype(np.float16).view(np.uint8).reshape(-1, 2)], axis=1).reshape(I, R)

# Load packed
from gguf import GGUFReader
r = GGUFReader('glm5_gguf/glm5-ternary-skel8.gguf')
packed_init = {}
for t in r.tensors:
    if 'ffn_gate_exps' in t.name:
        blk = t.name.split('.')[1]
        packed_init[int(blk)] = np.array(t.data, copy=True)

packed_src = packed_init[3]  # (E, I, R)

# ── Train ALL gate experts ──
n_blk = I * D // 256
trained_ternary = np.zeros((E, n_blk, 256), dtype=np.float32)
trained_gamma = np.zeros((E, n_blk), dtype=np.float32)
alpha_per_expert = np.ones(E, dtype=np.float32)

y_ref_all = W_fp16 @ Z  # (E, I, N)
norm_ref_all = (y_ref_all ** 2).mean(dim=(1, 2)).sqrt()

for e_idx in range(E):
    y_ref = W_fp16[e_idx] @ Z
    ls = (y_ref ** 2).mean().detach()
    norm_ref = (y_ref ** 2).mean().sqrt()
    
    tern_init, gamma_init = decode_tq1_expert(packed_src[e_idx])
    tern_t = torch.from_numpy(tern_init).to(device)
    g_t = torch.from_numpy(gamma_init).to(device)
    
    x = tern_t.clone().detach().requires_grad_(True)
    g = torch.log(g_t.clone().detach()).requires_grad_(True)
    
    opt = torch.optim.AdamW([x, g], lr=1e-2)
    lam = 1.0
    
    for step in range(150):
        q_hard = torch.clamp(torch.round(x), -1, 1)
        q = q_hard.detach() + x - x.detach()
        gc = torch.exp(torch.clamp(g, -5, 0))
        Wq = (q * gc[:, None]).reshape(I, D)
        y_q = Wq @ Z
        mse = ((y_q - y_ref) ** 2).mean() / ls
        norm_q = (y_q ** 2).mean().sqrt()
        norm_loss = (norm_q - norm_ref).abs()
        loss = mse + lam * norm_loss
        opt.zero_grad(); loss.backward(); opt.step()
        x.data.clamp_(-1.5, 1.5)
    
    q_trained = torch.clamp(torch.round(x), -1, 1).detach()
    g_trained = torch.exp(torch.clamp(g.detach(), -5, 0))
    
    # Keep gamma = skel8, compute global alpha for norm
    Wq_skel8g = (q_trained * g_t[:, None]).reshape(I, D)
    y_q_skel8g = Wq_skel8g @ Z
    norm_q = (y_q_skel8g ** 2).mean().sqrt()
    alpha = norm_ref / norm_q
    
    # Final evaluation
    y_adj = y_q_skel8g * alpha
    final_mse = ((y_adj - y_ref) ** 2).mean() / ls
    final_norm = (y_adj ** 2).mean().sqrt()
    flips = (q_trained != tern_t).float().mean().item() * 100
    
    trained_ternary[e_idx] = q_trained.cpu().numpy()
    trained_gamma[e_idx] = gamma_init  # SKEL8 gamma — unchanged
    alpha_per_expert[e_idx] = alpha.item()
    
    print(f"  e{e_idx:3d}: MSE={final_mse.item():.4f} α={alpha.item():.4f} norm={final_norm/norm_ref:.4f} flip={flips:.1f}%", flush=True)
    
    del y_ref, ls, norm_ref, tern_t, g_t, x, g; torch.cuda.empty_cache()

# Save
os.makedirs('glm5_pod', exist_ok=True)
for e_idx in range(E):
    packed_out = pack_tq1_expert(trained_ternary[e_idx], trained_gamma[e_idx])
    if e_idx == 0:
        packed_all = packed_out[None, :, :]
    else:
        packed_all = np.concatenate([packed_all, packed_out[None, :, :]], axis=0)
np.save('glm5_pod/tq1ste_L3_gate.npy', packed_all)
np.save('glm5_pod/alpha_L3_gate.npy', alpha_per_expert.astype(np.float16))
print(f"Saved: tq1ste_L3_gate.npy alpha_L3_gate.npy", flush=True)

# ── Build minimal GGUF (L3 gate only, rest from skel8) ──
print("Building test GGUF...", flush=True)
DST = 'glm5_gguf/glm5-ternary-tq1ste_L3.gguf'
SRC = 'glm5_gguf/glm5-ternary-skel8.gguf'

from gguf import GGUFReader, GGUFWriter
r2 = GGUFReader(SRC)
tensors_src = [(t.name, np.array(t.data, copy=True), t.tensor_type) for t in r2.tensors]
arch = r2.fields['general.architecture'].contents()
kv = [(k, v.contents(), v.types[0]) for k, v in r2.fields.items() if not k.startswith('GGUF.')]
del r2; import gc as gc_mod; gc_mod.collect()

w = GGUFWriter(DST, arch, use_temp_file=True)
for k, val, typ in kv:
    try: w.add_key_value(k, val, typ)
    except: pass
for name, data, dtype in tensors_src:
    if name == 'blk.3.ffn_gate_exps.weight':
        w.add_tensor(name, packed_all, raw_dtype=34, raw_shape=packed_all.shape)
    else:
        w.add_tensor(name, data, raw_dtype=dtype)
    gc.collect()
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"GGUF -> {DST}", flush=True)

# ── Test generation ──
print("Testing generation...", flush=True)
import subprocess
result = subprocess.run(
    ['llama-glm5/build-vulkan/bin/llama.exe', 'cli',
     '-m', DST, '-ngl', '99', '-c', '4096',
     '-p', 'Привет', '--temp', '0.7', '-n', '40'],
    capture_output=True, text=True, timeout=180, cwd='C:/HAGI_v2'
)
print("STDOUT:", result.stdout[-500:] if result.stdout else "EMPTY", flush=True)
print("STDERR:", result.stderr[-200:] if result.stderr else "NONE", flush=True)
print("DONE", flush=True)