"""Honest per-expert/per-layer diagnostics for GLM5 TQ1 (skel8).
Decodes gate/up/down TQ1 from GGUF, measures per-expert output error
vs fp16 teacher on calibration activations, and cascade FFN output error.
"""
import numpy as np, os, sys, io, json, time, gc, glob
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch

D, I, E, N_FF = 4096, 2048, 288, 2048
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

# ── universal TQ1 decode: brings (n_rows, R) -> (n_rows, n_out) fp32 ──
def decode_tq1_row(packed_2d):
    """packed_2d: (n_rows, R) uint8. Returns (n_rows, n_out) fp32.
    Each row packs n_out elements in D/256=16 blocks of 54 bytes.
    n_out = (R // 54) * 256 = R * 256/54 = R*128/27
    """
    n_rows, R = packed_2d.shape
    bpc = R // 54               # blocks per column (= n_out // 256)
    n_out = bpc * 256
    pb = packed_2d.reshape(n_rows * bpc, 54)
    gamma = pb[:, 52:54].copy().view(np.float16).ravel().astype(np.float32)
    q_raw = pb[:, :48].astype(np.int32)
    h_raw = pb[:, 48:52].astype(np.int32)
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
    W = (tern * gamma[:, None]).reshape(n_rows, n_out)
    return W

# ── load fp16 teacher for a whole layer ──
HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']

def load_W_fp16(il, p):
    """Returns per-expert matrix as (out, in).
    gate/up: (in=I,out=D); down: (in=ff=D,out=I) via blockwise scale.
    Return shape (E, out_dim, in_dim)."""
    from safetensors import safe_open
    kt = f'model.language_model.layers.{il}.mlp.experts.{{e}}.{p}_proj.weight'
    E_ = E
    # peek first to get out/in dims
    key0 = kt.format(e=0)
    with safe_open(f'{HF}/{_wm[key0]}', 'pt', 'cpu') as sf:
        w0 = sf.get_tensor(key0)
        s0 = sf.get_tensor(key0.replace('.weight', '.weight_scale_inv'))
    out_dim, in_dim = w0.shape
    res = torch.empty((E_, out_dim, in_dim), dtype=torch.float32)
    for e in range(E_):
        key = kt.format(e=e)
        sh = _wm[key]
        with safe_open(f'{HF}/{sh}', 'pt', 'cpu') as sf:
            w = sf.get_tensor(key).to(torch.float32)
            s = sf.get_tensor(key.replace('.weight', '.weight_scale_inv')).to(torch.float32)
            # s shape (out//128, in//128); broadcast over 128x128 blocks
            s_b = s[:, None, :, None]  # (ob,1,ib,1)
            W_e = w.reshape(s.shape[0], 128, s.shape[1], 128) * s_b
            res[e] = W_e.reshape(out_dim, in_dim)
    return res.to(device)

# ── calibration activations ──
base = 'glm5_gguf/dump_moe_in_skel8'
def load_cal_z(il, maxN=2048):
    mats = []
    for f in sorted(glob.glob(f'{base}/moe_in_L{il}_*.f32')):
        a = np.fromfile(f, dtype=np.float32); n = len(a)//D
        if n: mats.append(a[:n*D].reshape(n, D))
    if not mats: return None
    z = np.concatenate(mats, 0).astype(np.float32)
    N = min(z.shape[0], maxN)
    return torch.from_numpy(z[:N].T).contiguous().to(device)  # D x N

# ── GGUF tensors ──
from gguf import GGUFReader
r = GGUFReader('glm5_gguf/glm5-ternary-skel8.gguf')
tensors = {t.name: t for t in r.tensors}

def get_packed_blk(blk, pname):
    """Return packed (E, nrows, R) for gate/up/down of a block."""
    if pname in ('gate', 'up'):
        key = f'blk.{blk}.ffn_{pname}_exps.weight'
        t = tensors[key]
        data = np.array(t.data, copy=True)   # (E, nrows, R)
        return data
    else:  # down: (E, 4096, 432)
        key = f'blk.{blk}.ffn_down_exps.weight'
        t = tensors[key]
        return np.array(t.data, copy=True)

# ═══════════════ DIAG: L3 per-expert gate/up output error + cascade ═══════════════
il = 3
Z = load_cal_z(il)   # D x N
N = Z.shape[1]
print(f"\nL{il}: calibration N={N}", flush=True)

W_gate_fp16 = load_W_fp16(il, 'gate')   # (E, I, D)
W_up_fp16   = load_W_fp16(il, 'up')
W_down_fp16 = load_W_fp16(il, 'down')   # down teacher also I x D? check shape
# down_proj is (D, I) in original: input intermediate (2048), output hidden (4096)
# our load_W_fp16 returns (E, I=2048, D=4096). For down that's reversed: down: (4096->2048)
# So re-load down with swapped to get (E, in_ff=2048, out=4096)? Actually down maps ff(2048)->D(4096).
print("down fp16 shape:", W_down_fp16.shape, flush=True)

pg = get_packed_blk(il, 'gate'); pu = get_packed_blk(il, 'up'); pd = get_packed_blk(il, 'down')
print("packed gate/up/down:", pg.shape, pu.shape, pd.shape, flush=True)

# Decode gate/up: (E, I, D). down: (E, 4096, 2048) -- need transpose for (E, ff, D)
Wg_q = np.array([decode_tq1_row(pg[e]) for e in range(E)])          # (E, I, D)
Wu_q = np.array([decode_tq1_row(pu[e]) for e in range(E)])          # (E, I, D)
Wd_q = np.array([decode_tq1_row(pd[e]) for e in range(E)])          # (E, out=4096, in=2048)

Wg_q_t = torch.from_numpy(Wg_q).to(device)
Wu_q_t = torch.from_numpy(Wu_q).to(device)
Wd_q_t = torch.from_numpy(Wd_q).to(device)

# per-expert gate output error
z_t = Z  # D x N
with torch.no_grad():
    g_ref = W_gate_fp16 @ z_t   # (E, I, N)   [gate in is D=4096]
    u_ref = W_up_fp16 @ z_t     # (E, I, N)

# process in expert batches; N=2048, E=288
batch_e = 16
N_tok = z_t.shape[1]
var_y = None; mse_g = None; mse_u = None; mse_y = None
for e0 in range(0, E, batch_e):
    e1 = min(e0+batch_e, E)
    g_ref_e = W_gate_fp16[e0:e1] @ z_t   # (b, I, N)
    u_ref_e = W_up_fp16[e0:e1] @ z_t
    g_q_e   = Wg_q_t[e0:e1] @ z_t
    u_q_e   = Wu_q_t[e0:e1] @ z_t
    h_ref_e = g_ref_e * torch.nn.functional.silu(u_ref_e)   # (b, I, N)
    h_q_e   = g_q_e * torch.nn.functional.silu(u_q_e)
    # down: out(4096) = h(I) @ W(I,4096). W fp16 (E,4096,2048)
    # y = h^T (b,N,I) @ W^T (b,I,4096) = (b,N,4096)
    y_ref_e = torch.matmul(h_ref_e.transpose(1, 2), W_down_fp16[e0:e1].transpose(1, 2))
    y_q_e   = torch.matmul(h_q_e.transpose(1, 2),   Wd_q_t[e0:e1].transpose(1, 2))
    # accumulate stats per-expert
    var_e = (y_ref_e**2).mean(dim=(1,2))
    me_g  = ((g_q_e-g_ref_e)**2).mean(dim=(1,2))
    me_u  = ((u_q_e-u_ref_e)**2).mean(dim=(1,2))
    me_y  = ((y_q_e-y_ref_e)**2).mean(dim=(1,2))
    
    if var_y is None: var_y=var_e; mse_g=me_g; mse_u=me_u; mse_y=me_y
    else: var_y=torch.cat([var_y,var_e]); mse_g=torch.cat([mse_g,me_g]); mse_u=torch.cat([mse_u,me_u]); mse_y=torch.cat([mse_y,me_y])
    
    del g_ref_e,u_ref_e,g_q_e,u_q_e,h_ref_e,h_q_e,y_ref_e,y_q_e
    torch.cuda.empty_cache()
    
# save a small clean y_ref/y_q for later checks on a few experts


print(f"\n=== L{il} per-expert FFN output rel error (mean/median/max over 288 experts) ===", flush=True)
def stats(name, mse, var):
    rel = mse/var
    print(f"  {name}: mean={rel.mean().item()*100:.1f}% median={rel.median().item()*100:.1f}% "
          f"p90={rel.quantile(0.9).item()*100:.1f}% max={rel.max().item()*100:.1f}%", flush=True)
stats("gate out ", mse_g, (g_ref**2).mean(dim=(1,2)))
stats("up   out ", mse_u, (u_ref**2).mean(dim=(1,2)))
stats("FFN  out ", mse_y, var_y)

# Worst experts for FFN output
order = torch.argsort(mse_y/var_y, descending=True)
print(f"\n  Worst 10 experts (FFN output rel err):", flush=True)
for e in order[:10].tolist():
    print(f"    e{e}: gate={((mse_g[e]/(g_ref[e]**2).mean()).item()*100):.0f}% "
          f"up={((mse_u[e]/(u_ref[e]**2).mean()).item()*100):.0f}% "
          f"ffn_out={((mse_y[e]/var_y[e]).item()*100):.0f}%", flush=True)

# ── cascade hypothesis: does down-fp16 (ideal) applied to quantized h largely fix? ──
# recompute per-batch to avoid storing h_q globally
mse_y_fp16down = []; mse_y_rg = []; wd_new_all = []
lam = 1e-3
# for ridge we need per-expert H (N,I) and Y (N,4096); do per-expert in smaller loop
for e0 in range(0, E, batch_e):
    e1 = min(e0+batch_e, E)
    b = e1 - e0
    g_ref_e = W_gate_fp16[e0:e1] @ z_t
    u_ref_e = W_up_fp16[e0:e1] @ z_t
    g_q_e   = Wg_q_t[e0:e1] @ z_t
    u_q_e   = Wu_q_t[e0:e1] @ z_t
    h_ref_e = g_ref_e * torch.nn.functional.silu(u_ref_e)
    h_q_e   = g_q_e * torch.nn.functional.silu(u_q_e)
    H_ref = h_ref_e.transpose(1,2)  # (b,N,I)
    H_q   = h_q_e.transpose(1,2)    # (b,N,I)
    Y_ref = torch.matmul(H_ref, W_down_fp16[e0:e1].transpose(1,2))  # (b,N,4096)
    del g_ref_e,u_ref_e,g_q_e,u_q_e,h_ref_e; torch.cuda.empty_cache()
    
    # fp16-down applied to quantized h
    y_f16d = torch.matmul(H_q, W_down_fp16[e0:e1].transpose(1,2))
    mse_y_fp16down.append(((y_f16d - Y_ref)**2).mean(dim=(1,2)))
    del y_f16d; torch.cuda.empty_cache()
    
    # per-expert ridge solve
    for j in range(b):
        H = H_q[j]  # (N,I)
        Y = Y_ref[j]  # (N,4096)
        G = H.T @ H + lam * torch.eye(I, device=device)
        B = H.T @ Y
        W = torch.linalg.solve(G, B)
        wd_new_all.append(W)
    del H_q, H_ref, Y_ref; torch.cuda.empty_cache()
    
wd_new = torch.stack(wd_new_all)  # (E, I, 4096)
mse_y_fp16down = torch.cat(mse_y_fp16down)

# evaluate ridge-solved down on quantized h (batch)
mse_y_rg = []
for e0 in range(0, E, batch_e):
    e1 = min(e0+batch_e, E)
    g_q_e = Wg_q_t[e0:e1] @ z_t; u_q_e = Wu_q_t[e0:e1] @ z_t
    h_q_e = g_q_e * torch.nn.functional.silu(u_q_e)
    del g_q_e, u_q_e
    H_q = h_q_e.transpose(1,2)
    y_rg = torch.matmul(H_q, wd_new[e0:e1])
    mse_y_rg.append(((y_rg - (H_q @ W_down_fp16[e0:e1].transpose(1,2)))**2).mean(dim=(1,2)))
    del H_q, h_q_e, y_rg; torch.cuda.empty_cache()
mse_y_rg = torch.cat(mse_y_rg)

print(f"\n  Replacing down with fp16 teacher: FFN out rel={((mse_y_fp16down/var_y).mean()*100):.1f}% "
      f"(vs tq1-down {(mse_y/var_y).mean()*100:.1f}%)", flush=True)
print(f"  Cascade W2 ridge re-solve (fp32 down): FFN out rel={((mse_y_rg/var_y).mean()*100):.1f}% "
      f"(vs tq1-down {(mse_y/var_y).mean()*100:.1f}%)", flush=True)

# ── how many experts improve? ──
impr = (mse_y_rg < mse_y)
print(f"  Experts improved by ridge solve: {impr.sum().item()}/288", flush=True)

# save the solved fp32 down (I,4096) per expert — this is the cascade fix
np.save(f'glm5_pod/wdown_fp32_L{il}.npy', wd_new.cpu().numpy())
print(f"Saved wdown_fp32_L{il}.npy", flush=True)
print("DONE", flush=True)