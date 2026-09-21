"""Cascade fit L3 (DSv4-style, all-ternary) — batched to fit memory.
1. Honest per-expert gate/up/FFN output rel err (skel8 as-is).
2. Per-expert alpha to kill the 1.51x norm inflation.
3. Ridge re-solve down under rescaled quantized h.
"""
import numpy as np, os, sys, io, json, time, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch
D, I, E = 4096, 2048, 288
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

def decode_tq1_row(packed_2d):
    n_rows, R = packed_2d.shape
    bpc = R // 54
    pb = packed_2d.reshape(n_rows * bpc, 54)
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
    return (tern * gamma[:, None]).reshape(n_rows, R // 54 * 256)

HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
def load_fp16_batch(il, p, e_list):
    from safetensors import safe_open
    kt = f'model.language_model.layers.{il}.mlp.experts.{{e}}.{p}_proj.weight'
    key0 = kt.format(e=0)
    with safe_open(f'{HF}/{_wm[key0]}', 'pt', 'cpu') as sf:
        out_dim, in_dim = tuple(sf.get_tensor(key0).shape)
    res = torch.empty((len(e_list), out_dim, in_dim), dtype=torch.float32)
    for j, e in enumerate(e_list):
        key = kt.format(e=e)
        with safe_open(f'{HF}/{_wm[key]}', 'pt', 'cpu') as sf:
            w = sf.get_tensor(key).to(torch.float32)
            s = sf.get_tensor(key.replace('.weight','.weight_scale_inv')).to(torch.float32)
            res[j] = (w.reshape(s.shape[0],128,s.shape[1],128)*s[:,None,:,None]).reshape(out_dim,in_dim)
    return res.to(device)

base = 'glm5_gguf/dump_moe_in_skel8'
mats=[]
for f in sorted(glob.glob(f'{base}/moe_in_L3_*.f32')):
    a=np.fromfile(f,dtype=np.float32); n=len(a)//D
    if n: mats.append(a[:n*D].reshape(n,D))
z=np.concatenate(mats,0).astype(np.float32); N=min(z.shape[0],2048)
Z=torch.from_numpy(z[:N].T).contiguous().to(device)
print(f"cal N={N}", flush=True)

from gguf import GGUFReader
r=GGUFReader('glm5_gguf/glm5-ternary-skel8.gguf')
tensors={t.name:t for t in r.tensors}
def decoded_exps(blk, pname):
    key=f'blk.{blk}.ffn_{pname}_exps.weight'
    data=np.array(tensors[key].data, copy=True)
    return np.ascontiguousarray([decode_tq1_row(data[e]) for e in range(E)], dtype=np.float32)

Wg_q=torch.from_numpy(decoded_exps(3,'gate')).to(device)
Wu_q=torch.from_numpy(decoded_exps(3,'up')).to(device)
Wd_q=torch.from_numpy(decoded_exps(3,'down')).to(device)
print("quantized tensors loaded on", device, flush=True)
del tensors, r; torch.cuda.empty_cache()

def silu(x): return x * torch.sigmoid(x)

BS=12
mse_g=[]; mse_u=[]; mse_y=[]; var_y=[]; alpha_list=[]; gvp=[]; uvp=[]
wd_new=[]
t0=time.time()
for e0 in range(0, E, BS):
    e1=min(e0+BS, E)
    eb=Wg_q[e0:e1].shape[0]
    Wg_fp=load_fp16_batch(3,'gate',list(range(e0,e1)))
    Wu_fp=load_fp16_batch(3,'up',list(range(e0,e1)))
    Wd_fp=load_fp16_batch(3,'down',list(range(e0,e1)))
    g_ref=Wg_fp@Z; u_ref=Wu_fp@Z
    g_q=Wg_q[e0:e1]@Z; u_q=Wu_q[e0:e1]@Z
    h_ref=g_ref*silu(u_ref); h_q=g_q*silu(u_q)
    y_ref=torch.matmul(h_ref.transpose(1,2), Wd_fp.transpose(1,2))  # (b,N,D)
    y_q=torch.matmul(h_q.transpose(1,2), Wd_q[e0:e1].transpose(1,2))

    var_y.append((y_ref**2).mean(dim=(1,2)))
    gvp.append((g_ref**2).mean(dim=(1,2)))
    uvp.append((u_ref**2).mean(dim=(1,2)))
    mse_g.append(((g_q-g_ref)**2).mean(dim=(1,2)))
    mse_u.append(((u_q-u_ref)**2).mean(dim=(1,2)))
    mse_y.append(((y_q-y_ref)**2).mean(dim=(1,2)))

    # alpha per expert = sqrt( <h_ref²>/<h_q²> )
    a=torch.sqrt(((h_ref**2).mean(dim=(1,2)))/ ((h_q**2).mean(dim=(1,2))) + 1e-6)
    alpha_list.append(a)

    # rescaled h and ridge down solve
    h_qs=h_q*a[:,None,None]
    H=h_qs.transpose(1,2)  # (b,N,I)
    for j in range(eb):
        G=H[j].T@H[j]+1e-3*torch.eye(I, device=device)
        B=H[j].T@y_ref[j]
        wd_new.append(torch.linalg.solve(G,B))

    del Wg_fp,Wu_fp,Wd_fp,g_ref,u_ref,g_q,u_q,h_ref,h_q,h_qs,H,y_ref,y_q
    torch.cuda.empty_cache()
    if (e0//BS)%2==0: print(f"  batch {e0}-{e1} {time.time()-t0:.0f}s", flush=True)

var_y=torch.cat(var_y); mse_g=torch.cat(mse_g); mse_u=torch.cat(mse_u); mse_y=torch.cat(mse_y)
gvp=torch.cat(gvp); uvp=torch.cat(uvp)
alpha=torch.cat(alpha_list)
wd_new=torch.stack(wd_new)  # (E, I, 4096)

print(f"\n=== L3 (skel8 as-is) ===", flush=True)
print(f"  gate out rel: mean={((mse_g/gvp).mean()*100):.0f}%", flush=True)
print(f"  up   out rel: mean={((mse_u/uvp).mean()*100):.0f}%", flush=True)
print(f"  FFN  out rel: mean={((mse_y/var_y).mean()*100):.0f}%", flush=True)
print(f"  alpha mean={alpha.mean().item():.4f} (fixes 1.51x inflation -> ~0.66)", flush=True)

# Evaluate alpha + ridge HONESTLY: y_ref must come from fp16 h_ref, not rescaled q.
# We need h_ref (fp16 gate/up). Recompute per batch.
mse_y_asis=[]; mse_y_f16down=[]; mse_y_alpha_f16=[]; mse_y_alpha_ridge=[]; var_y2=[]
for e0 in range(0, E, BS):
    e1=min(e0+BS, E); eb=Wg_q[e0:e1].shape[0]
    Wg_fp=load_fp16_batch(3,'gate',list(range(e0,e1)))
    Wu_fp=load_fp16_batch(3,'up',list(range(e0,e1)))
    Wd_fp=load_fp16_batch(3,'down',list(range(e0,e1)))
    g_ref=Wg_fp@Z; u_ref=Wu_fp@Z
    h_ref=(g_ref*silu(u_ref))
    g_q=Wg_q[e0:e1]@Z; u_q=Wu_q[e0:e1]@Z
    h_q=(g_q*silu(u_q))
    h_qs=h_q*alpha[e0:e1][:,None,None]
    H_ref=h_ref.transpose(1,2)  # (b,N,I)
    H_q=h_q.transpose(1,2)
    H_qs=h_qs.transpose(1,2)
    y_ref=torch.matmul(H_ref, Wd_fp.transpose(1,2))          # fp16 teacher target
    var_y2.append((y_ref**2).mean(dim=(1,2)))
    mse_y_asis.append(((torch.matmul(H_q, Wd_q[e0:e1].transpose(1,2))-y_ref)**2).mean(dim=(1,2)))
    mse_y_f16down.append(((torch.matmul(H_q, Wd_fp.transpose(1,2))-y_ref)**2).mean(dim=(1,2)))
    mse_y_alpha_f16.append(((torch.matmul(H_qs, Wd_fp.transpose(1,2))-y_ref)**2).mean(dim=(1,2)))
    mse_y_alpha_ridge.append(((torch.matmul(H_qs, wd_new[e0:e1])-y_ref)**2).mean(dim=(1,2)))
    del Wg_fp,Wu_fp,Wd_fp,g_ref,u_ref,h_ref,g_q,u_q,h_q,h_qs,H_ref,H_q,H_qs,y_ref
    torch.cuda.empty_cache()
var_y2=torch.cat(var_y2)
mse_asis=torch.cat(mse_y_asis); mse_f16d=torch.cat(mse_y_f16down)
mse_a16=torch.cat(mse_y_alpha_f16); mse_ar=torch.cat(mse_y_alpha_ridge)

print(f"\n=== HONEST L3 FFN output rel error (vs fp16 teacher) ===", flush=True)
print(f"  as-is (q gate/up + q down):     {((mse_asis/var_y2).mean()*100):.0f}%", flush=True)
print(f"  q gate/up + fp16 down:          {((mse_f16d/var_y2).mean()*100):.0f}%", flush=True)
print(f"  alpha*q + fp16 down:            {((mse_a16/var_y2).mean()*100):.0f}%", flush=True)
print(f"  alpha*q + W2 ridge down:        {((mse_ar/var_y2).mean()*100):.0f}%", flush=True)
print(f"  experts ridge<f16down:          {(mse_ar<mse_a16).sum().item()}/{E}", flush=True)
print(f"  experts alpha-f16<asis:         {(mse_a16<mse_asis).sum().item()}/{E}", flush=True)

np.save('glm5_pod/alpha_L3_gateup.npy', alpha.cpu().numpy().astype(np.float16))
np.save('glm5_pod/wdown_fp32_L3.npy', wd_new.cpu().numpy().astype(np.float16))
print("saved alpha_L3_gateup.npy + wdown_fp32_L3.npy", flush=True)
print("DONE", flush=True)