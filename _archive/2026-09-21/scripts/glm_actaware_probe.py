"""Activation-aware ternary for gate/up, coordinate-descent style, honest metric.
Per expert: choose ternary pattern + per-block scale to minimize ||Wq z - W z||²
on calibration activations. Then re-solve down under new h. Measure FFN error.
All-ternary (gate/up/down stay TQ1; only per-block scale is float per block = same as gamma).
"""
import numpy as np, os, sys, io, json, time, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch
D, I, E = 4096, 2048, 288
bpc = D // 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

def decode_tq1_row(packed_2d):
    n_rows, R = packed_2d.shape
    bp = R // 54
    pb = packed_2d.reshape(n_rows * bp, 54)
    gamma = pb[:, 52:54].copy().view(np.float16).ravel().astype(np.float32)
    q_raw = pb[:, :48].astype(np.int32); h_raw = pb[:, 48:52].astype(np.int32)
    def dec(qv): return ((qv * 243 + 128) // 256) % 243
    dv = dec(q_raw); dh = dec(h_raw)
    flat = np.zeros((pb.shape[0], 256), np.int32)
    for n, c in enumerate([81, 27, 9, 3, 1]):
        flat[:, n*32:(n+1)*32] = (dv[:, :32] // c) % 3
        flat[:, 160+n*16:160+(n+1)*16] = (dv[:, 32:] // c) % 3
    for col in range(4):
        val = dh[:, col]
        for mi, ci in enumerate([81, 27, 9, 3]):
            flat[:, 240 + mi*4 + col] = (val // ci) % 3
    tern = flat.astype(np.float32) - 1.0
    return (tern * gamma[:, None]).reshape(n_rows, bp * 256), tern, gamma

HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
def load_fp16(il, p, e):
    from safetensors import safe_open
    key = f'model.language_model.layers.{il}.mlp.experts.{e}.{p}_proj.weight'
    with safe_open(f'{HF}/{_wm[key]}', 'pt', 'cpu') as sf:
        w = sf.get_tensor(key).to(torch.float32)
        s = sf.get_tensor(key.replace('.weight','.weight_scale_inv')).to(torch.float32)
        return (w.reshape(s.shape[0],128,s.shape[1],128)*s[:,None,:,None]).reshape(w.shape[0], w.shape[1]).to(device)

base = 'glm5_gguf/dump_moe_in_skel8'
mats=[]
for f in sorted(glob.glob(f'{base}/moe_in_L3_*.f32')):
    a=np.fromfile(f,dtype=np.float32); n=len(a)//D
    if n: mats.append(a[:n*D].reshape(n,D))
z=np.concatenate(mats,0).astype(np.float32); N=min(z.shape[0],2048)
Z=torch.from_numpy(z[:N].T).contiguous().to(device)  # D x N
Zb = Z.reshape(bpc, 256, N)  # (bpc,256,N)
print(f"cal N={N}", flush=True)

from gguf import GGUFReader
r=GGUFReader('glm5_gguf/glm5-ternary-skel8.gguf')
tensors={t.name:t for t in r.tensors}
def packed_expert(blk, pname, e):
    key=f'blk.{blk}.ffn_{pname}_exps.weight'
    return np.array(tensors[key].data[e], copy=True)  # (rows, R)

def pack_tq1(ternary, gamma, R):
    """ternary (nblocks,256), gamma (nblocks,) -> (nrows, R). nblocks = nrows*bpc"""
    nblocks = ternary.shape[0]
    nrows = nblocks // bpc
    flat = np.clip((ternary + 1).astype(np.int32), 0, 2)
    qs = np.zeros((nblocks, 48), np.int32); qh = np.zeros((nblocks, 4), np.int32)
    for n, c in enumerate([81, 27, 9, 3, 1]):
        qs[:, :32] += flat[:, n*32:(n+1)*32] * c
        qs[:, 32:] += flat[:, 160+n*16:160+(n+1)*16] * c
    for m, c in enumerate([81, 27, 9, 3]):
        qh += flat[:, 240+m*4:244+m*4] * c
    def comp(qq): return ((qq.astype(np.uint32) * 256 + 242) // 243).astype(np.uint8)
    packed = np.concatenate([comp(qs), comp(qh), gamma.astype(np.float16).view(np.uint8).reshape(-1,2)],1)
    return packed.reshape(nrows, R)

# ── Activation-aware ternary optimization for gate (and up) of expert e ──
# We optimize Wq to minimize ||Wq z - W z||². Given per-block structure:
# For output row r, W[r,:] lives across D//256=16 blocks (one per column group).
# The activation z for block (row r, block bc) is Zb[bc, :, :] (256 x N).
# Contribution: W[r, bc*256:(bc+1)*256] @ Zb[bc]. 
# Ternary: Wq[r, :] = q[r,bc,:] * gamma[r,bc]. Optimize q in {-1,0,1}^256 per (r,bc).
#
# Best sign per position given scale gamma:
#   For fixed gamma, min over q∈{-1,0,+1} of ||gamma*q_sel z - W_sel z||² ...
# Use continuous relaxation then threshold. We solve per block: 
#   minimize ||W_sel z - gamma * q_sel z||²  → equivalent proximity to W_sel z projected.
# Standard: optimal real q for that block = (W_sel z)(gamma z)ᵀ / (gamma² z zᵀ)... 
# More robust: minimize least squares: q*= argmin  ||W_sel z - gamma q z||
#   = argmin_q qᵀ (gamma² z zᵀ) q - 2 gamma qᵀ(W_sel z zᵀ) ... 
# Simply: let a = W_sel z (1 x N) broadcasting; b = gamma z (256 x N).
# residual squared = ||a - q b||² = q prosthesis... we minimize per position independently
# if we ignore cross terms: q_i* = clip(round( <a, z_i> / (gamma ||z_i||²) ) , -1,1)
# where a = W_sel z summed over N : <a, z_i> = W_i·(z zᵀ)_i. => q_i ∝ W_i·G_i with G=z zᵀ.
# So: q_i = clip(round( (W G)_i / (gamma G_ii) ), -1,1) where G = Zb Zbᵀ (256x256) for that block.
# That's the activation-weighted sign! Max-gamma used W_i directly; this uses (W G)_i.

e = 0
pname = 'gate'
Wf = load_fp16(3, pname, e)  # (I, D)
Wf_b = Wf.reshape(I, bpc, 256)  # (I, bpc, 256)
# For each block bc, compute G_bc = Zb[bc] @ Zb[bc].T  (256x256)
# and (W@G) activation-weighted: per row r, block bc: Wf_b[r,bc,:] @ G_bc  (256,)
# optimal gamma_q: q ~ activation-derived. But gamma and q are coupled.

# We use two-stage: (1) get ternary from activation-projection, (2) per-block scale LSQ.
gamma = torch.zeros(I, bpc, device=device)
q = torch.zeros(I, bpc, 256, device=device)
for bc in range(bpc):
    zb = Zb[bc]  # (256, N)
    G = zb @ zb.T  # (256,256)
    diag = torch.diag(G).clamp(min=1e-6)
    # activation-weighted "importance" per weight element
    WG = Wf_b[:, bc, :] @ G  # (I, 256)
    # scale to match W magnitude st dev: pick gamma = sqrt(mean(W^2)/... )
    # ternary sign: sign(WG) with zeroing small ones
    q[:, bc, :] = torch.sign(WG)
    # per-row optimal scale
    # minimize ||W_sel z - gamma q_sel z||²  → gamma = <q z, W z>/<q z,q z>
    qz = q[:, bc, :].unsqueeze(1) @ zb  # (I,1,N)
    Wz = Wf_b[:, bc, :] @ zb   # (I, N)
    gamma[:, bc] = (qz.squeeze(1) * Wz).sum(1) / (qz.squeeze(1)**2).sum(1).clamp(min=1e-6)

# Now sparsify: zero out q where activation impact is negligible.
# We want ~same nz density as skel8 (0.244). Keep top-24.4% |WG| per block.
nz_target = 0.244
for bc in range(bpc):
    mag = q[:, bc, :].abs() * 1.0  # placeholder, use WG magnitude
    imp = (Wf_b[:, bc, :].abs() * torch.diag(G).sqrt().unsqueeze(0))  # roughly |W|*std(z)
    # Actually use WG magnitude (activation-weighted)
    WG_mag = (Wf_b[:, bc, :] @ G).abs()  # (I,256)
    thr = torch.quantile(WG_mag, 1-nz_target, dim=1, keepdim=True)  # per row
    keep = WG_mag >= thr
    q[:, bc, :][~keep] = 0
    q[:, bc, :] = torch.sign(WG_mag) * (q[:, bc, :]!=0).float()

# reconstruct decoded Wq
Wq_new = (q * gamma[:, :, None]).reshape(I, D)
# honest gate output error
y_ref = Wf @ Z
y_qn = Wq_new @ Z
rel = ((y_qn-y_ref)**2).mean() / (y_ref**2).mean()
print(f"expert {e} {pname}: activation-aware ternary out_rel = {rel.item()*100:.0f}%")
# baseline skel8
packed0 = packed_expert(3,pname,e)
Wq0, _, _ = decode_tq1_row(packed0)
Wq0t = torch.from_numpy(Wq0).to(device)
y_q0 = Wq0t @ Z
rel0 = ((y_q0-y_ref)**2).mean()/(y_ref**2).mean()
print(f"   baseline max-gamma out_rel = {rel0.item()*100:.0f}%")

# sparsity check
print(f"  nz frac: skel8={np.mean(packed0[:1])*0+np.mean((decode_tq1_row(packed0)[1]!=0)):.3f} new={((q!=0).float().mean()).item():.3f}")
torch.cuda.empty_cache()
print("DONE single-expert probe", flush=True)