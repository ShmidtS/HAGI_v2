"""Full cascade fit ALL layers 3..45: quantized gate/up + re-solved down (W2).
DSv4 cascade principle: downstream compensates for ternary gate/up noise.
Per expert: alpha (norm fix) + W2 re-solve, VALIDATED on held-out tokens.
Saves: alpha + W2 (fp32) + honest train/val FFN rel error per layer.
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
    return (tern * gamma[:, None]).reshape(n_rows, bp * 256)

HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
def load_fp16_batch(il, p, e_list):
    from safetensors import safe_open
    kt = f'model.language_model.layers.{il}.mlp.experts.{{e}}.{p}_proj.weight'
    key0 = kt.format(e=0)
    with safe_open(f'{HF}/{_wm[key0]}', 'pt', 'cpu') as sf:
        o, ii = tuple(sf.get_tensor(key0).shape)
    res = torch.empty((len(e_list), o, ii), dtype=torch.float32)
    for j, e in enumerate(e_list):
        key = kt.format(e=e)
        with safe_open(f'{HF}/{_wm[key]}', 'pt', 'cpu') as sf:
            w = sf.get_tensor(key).to(torch.float32)
            s = sf.get_tensor(key.replace('.weight','.weight_scale_inv')).to(torch.float32)
            res[j] = (w.reshape(s.shape[0],128,s.shape[1],128)*s[:,None,:,None]).reshape(o, ii)
    return res.to(device)

base = 'glm5_gguf/dump_moe_in_skel8'
def cal_z(il, n_use):
    mats = []
    for f in sorted(glob.glob(f'{base}/moe_in_L{il}_*.f32')):
        a = np.fromfile(f, dtype=np.float32); n = len(a)//D
        if n: mats.append(a[:n*D].reshape(n, D))
    z = np.concatenate(mats, 0).astype(np.float32)
    return torch.from_numpy(z[:n_use].T).contiguous().to(device)  # D x m

from gguf import GGUFReader
r = GGUFReader('glm5_gguf/glm5-ternary-skel8.gguf')
tensors = {t.name: t for t in r.tensors}
def decoded_exps(blk, pname):
    key = f'blk.{blk}.ffn_{pname}_exps.weight'
    data = np.array(tensors[key].data, copy=True)
    return torch.from_numpy(np.ascontiguousarray([decode_tq1_row(data[e]) for e in range(E)], dtype=np.float32)).to(device)

def silu(x): return x * torch.sigmoid(x)

NTRAIN = 2048  # fit on first 2048, validate on more
results = {}
os.makedirs('glm5_pod', exist_ok=True)

for il in range(3, 46):
    # load quantized gate/up/down (all ternary) once
    Wg_q = decoded_exps(il,'gate'); Wu_q = decoded_exps(il,'up'); Wd_q = decoded_exps(il,'down')
    Ztr = cal_z(il, NTRAIN)      # D x 2048
    # validation: use tokens beyond train (up to 3586)
    Ztr_full = cal_z(il, 3000)
    Zva = Ztr_full[:, 2048:3000] # D x 952 held-out
    del Ztr_full; torch.cuda.empty_cache()

    t0 = time.time()
    alpha = torch.zeros(E, device=device)
    wd_new = torch.zeros(E, I, D, dtype=torch.float32, device=device)
    mse_g_tr=[]; mse_ffn_tr=[]; mse_ffn_va=[]; mse_ffn_nofix_tr=[]; var_tr=[]; var_va=[]
    BS = 12
    for e0 in range(0, E, BS):
        e1 = min(e0+BS, E); eb = e1-e0
        Wg_fp = load_fp16_batch(il,'gate',list(range(e0,e1)))
        Wu_fp = load_fp16_batch(il,'up',list(range(e0,e1)))
        Wd_fp = load_fp16_batch(il,'down',list(range(e0,e1)))
        # train batch
        g_ref = Wg_fp@Ztr; u_ref = Wu_fp@Ztr
        h_ref = g_ref*silu(u_ref)
        g_q = Wg_q[e0:e1]@Ztr; u_q = Wu_q[e0:e1]@Ztr
        h_q = g_q*silu(u_q)
        H_ref = h_ref.transpose(1,2); H_q = h_q.transpose(1,2)
        y_ref = torch.matmul(H_ref, Wd_fp.transpose(1,2))
        y_q_nofix = torch.matmul(H_q, Wd_q[e0:e1].transpose(1,2))
        mse_ffn_nofix_tr.append(((y_q_nofix-y_ref)**2).mean(dim=(1,2)))
        var_tr.append((y_ref**2).mean(dim=(1,2)))
        # honest gate error
        mse_g_tr.append(((g_q-g_ref)**2).mean(dim=(1,2)))
        # alpha + W2 solve
        a = torch.sqrt(((h_ref**2).mean(dim=(1,2)))/((h_q**2).mean(dim=(1,2)))+1e-6)
        alpha[e0:e1]=a
        h_qs = h_q*a[:,None,None]
        H_qs = h_qs.transpose(1,2)
        for j in range(eb):
            G = H_qs[j].T@H_qs[j] + 1e-3*torch.eye(I, device=device)
            B = H_qs[j].T @ y_ref[j]
            wd_new[e0+j] = torch.linalg.solve(G,B)
        # train error after fix
        mse_ffn_tr.append(((torch.matmul(H_qs, wd_new[e0:e1])-y_ref)**2).mean(dim=(1,2)))
        # validation error (same fixed alpha/W2 on held-out)
        g_ref_v = Wg_fp@Zva; u_ref_v = Wu_fp@Zva
        h_ref_v = g_ref_v*silu(u_ref_v)
        g_q_v = Wg_q[e0:e1]@Zva; u_q_v = Wu_q[e0:e1]@Zva
        h_q_v = g_q_v*silu(u_q_v)
        H_ref_v = h_ref_v.transpose(1,2); H_q_v = (h_q_v*a[:,None,None]).transpose(1,2)
        y_ref_v = torch.matmul(H_ref_v, Wd_fp.transpose(1,2))
        mse_ffn_va.append(((torch.matmul(H_q_v, wd_new[e0:e1])-y_ref_v)**2).mean(dim=(1,2)))
        var_va.append((y_ref_v**2).mean(dim=(1,2)))
        del Wg_fp,Wu_fp,Wd_fp,g_ref,u_ref,h_ref,g_q,u_q,h_q,h_qs,H_ref,H_q,H_qs,y_ref,y_q_nofix
        del g_ref_v,u_ref_v,h_ref_v,g_q_v,u_q_v,h_q_v,H_ref_v,H_q_v,y_ref_v
        torch.cuda.empty_cache()

    mse_g_tr=torch.cat(mse_g_tr); mse_ffn_tr=torch.cat(mse_ffn_tr); mse_ffn_va=torch.cat(mse_ffn_va)
    mse_ffn_nofix_tr=torch.cat(mse_ffn_nofix_tr); var_tr=torch.cat(var_tr); var_va=torch.cat(var_va)

    # gate reference variance for rel (recompute quickly per batch skip; use h_ref var approx)
    gate_rel = (mse_g_tr/ (mse_g_tr[0]*0+1)).mean()  # placeholder -> we'll compute properly
    # recompute gate ref variance
    grv=[]; urv=[]
    for e0 in range(0,E,BS):
        e1=min(e0+BS,E)
        Wg_fp=load_fp16_batch(il,'gate',list(range(e0,e1)))
        Wu_fp=load_fp16_batch(il,'up',list(range(e0,e1)))
        grv.append(((Wg_fp@Ztr)**2).mean(dim=(1,2))); urv.append(((Wu_fp@Ztr)**2).mean(dim=(1,2)))
        del Wg_fp,Wu_fp; torch.cuda.empty_cache()
    grv=torch.cat(grv); urv=torch.cat(urv)
    gate_rel = ((mse_g_tr/grv).mean()*100).item()
    ffntr = ((mse_ffn_tr/var_tr).mean()*100).item()
    ffva = ((mse_ffn_va/var_va).mean()*100).item()
    ffn_nofix = ((mse_ffn_nofix_tr/var_tr).mean()*100).item()

        # store alpha + W2 (cast to fp16 for storage; matmul already done in fp32)
    np.save(f'glm5_pod/alpha_L{il}_gateup.npy', alpha.cpu().numpy().astype(np.float16))
    np.save(f'glm5_pod/wdown_fp32_L{il}.npy', wd_new.cpu().numpy().astype(np.float16))

    results[il] = dict(gate_rel=gate_rel, ffn_nofix=ffn_nofix, ffn_tr=ffntr, ffn_va=ffva)
    print(f"L{il}: gate_rel={gate_rel:.0f}% ffn_nofix={ffn_nofix:.0f}% "
          f"ffn_fit_tr={ffntr:.0f}% ffn_fit_va={ffva:.0f}%  ({time.time()-t0:.0f}s)", flush=True)

    del Wg_q, Wu_q, Wd_q, Ztr, Zva; torch.cuda.empty_cache()

import json as _json
with open('glm5_pod/cascade_results.json','w') as f:
    _json.dump({str(k):{kk:round(vv,3) for kk,vv in v.items()} for k,v in results.items()}, f, indent=2)
print("\nDONE all layers", flush=True)