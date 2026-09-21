"""Full activation-aware ternary quantization of gate/up for ALL layers.
Fit per-expert ternary on train tokens (2048), validate FFN output on held-out (952).
Cheap: G = Zb Zb^T computed once per layer/block, shared across experts.
Saves packed TQ1 gate/up + honest cascade FFN output rel error (train/val).
Down stays as-is (ternary) for now; we measure before deciding.
"""
import numpy as np, os, sys, io, json, time, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch
D, I, E = 4096, 2048, 288
bpc = D // 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
def load_fp16(il, p, e_list, o_dim, i_dim):
    from safetensors import safe_open
    kt = f'model.language_model.layers.{il}.mlp.experts.{{e}}.{p}_proj.weight'
    res = torch.empty((len(e_list), o_dim, i_dim), dtype=torch.float32)
    for j, e in enumerate(e_list):
        key = kt.format(e=e)
        with safe_open(f'{HF}/{_wm[key]}', 'pt', 'cpu') as sf:
            w = sf.get_tensor(key).to(torch.float32)
            s = sf.get_tensor(key.replace('.weight','.weight_scale_inv')).to(torch.float32)
            res[j] = (w.reshape(s.shape[0],128,s.shape[1],128)*s[:,None,:,None]).reshape(o_dim, i_dim)
    return res.to(device)

def load_fp16_single(il, p, e):
    from safetensors import safe_open
    key = f'model.language_model.layers.{il}.mlp.experts.{e}.{p}_proj.weight'
    with safe_open(f'{HF}/{_wm[key]}', 'pt', 'cpu') as sf:
        w = sf.get_tensor(key).to(torch.float32)
        s = sf.get_tensor(key.replace('.weight','.weight_scale_inv')).to(torch.float32)
        return (w.reshape(s.shape[0],128,s.shape[1],128)*s[:,None,:,None]).reshape(w.shape[0], w.shape[1]).to(device)

base = 'glm5_gguf/dump_moe_in_skel8'
def cal_z(il, n_use):
    mats = []
    for f in sorted(glob.glob(f'{base}/moe_in_L{il}_*.f32')):
        a = np.fromfile(f, dtype=np.float32); n = len(a)//D
        if n: mats.append(a[:n*D].reshape(n, D))
    z = np.concatenate(mats, 0).astype(np.float32)
    return torch.from_numpy(z[:n_use].T).contiguous().to(device)

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

def pack_tq1(ternary, gamma, R):
    """ternary (nrows*bpc,256), gamma(nrows*bpc,) -> (nrows, R) uint8"""
    if torch.is_tensor(ternary): ternary = ternary.detach().cpu().numpy()
    if torch.is_tensor(gamma): gamma = gamma.detach().cpu().numpy()
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

def silu(x): return x * torch.sigmoid(x)

from gguf import GGUFReader
r = GGUFReader('glm5_gguf/glm5-ternary-skel8.gguf')
tensors = {t.name: t for t in r.tensors}

# gate/up: rows=I(2048) cols=D(4096); down: rows=D(4096) cols=I(2048)
# For gate/up act-aware: Wf shape (I, D)?? No — gate/up proj: input D(4096) -> output I(2048)
# Wait gate_proj: (I,D)=(2048,4096): output rows are the FFN hidden dim? Let's verify.
# Earlier load_fp16 for gate returned (2048, 4096)?? Actually gate_proj maps hidden(D=4096)->ffn(I=2048)?
# But we index Wf_b as (out_dim, bpc, 256). For gate: what is "out"? The FFN intermediate dim.
# q (out,bpc,256) reconstructs rows=out. y_ref=Wf@Z: (out, in=N). If Wf is (I,D)=(2048,4096)
# and Z is (D,N)=(4096,N) then y=(2048,N) for I=out=2048. So gate out_dim=I=2048, in=D=4096.
# bpc=4096/256=16 blocks per out row. Consistent.

NTRAIN = 2048
NZ = 0.244
results = {}
os.makedirs('glm5_pod', exist_ok=True)

import sys as _sys
START=max(3,int(_sys.argv[1]) if len(_sys.argv)>1 else 3)
END=min(46,int(_sys.argv[2]) if len(_sys.argv)>2 else 46)
for il in range(START, END):
    Ztr = cal_z(il, NTRAIN)
    Zva = cal_z(il, 3000)[:, NTRAIN:3000]
    Nva = Zva.shape[1]
    # shared per-block gram G
    Ztr_b = Ztr.reshape(bpc, 256, NTRAIN)
    G_block = [ (Ztr_b[bc] @ Ztr_b[bc].T) for bc in range(bpc) ]  # (256,256) each
    t0 = time.time()

    # load quantized down (decode) once
    pd = np.array(tensors[f'blk.{il}.ffn_down_exps.weight'].data, copy=True)
    Wd_q = torch.from_numpy(np.ascontiguousarray([decode_tq1_row(pd[e]) for e in range(E)], dtype=np.float32)).to(device)  # (E,4096,2048)

    # Preallocate new packed gate/up
    packed_g_out = np.empty((E, I, 54*bpc), dtype=np.uint8)
    packed_u_out = np.empty((E, I, 54*bpc), dtype=np.uint8)

    mse_ffn_tr = []; mse_ffn_va = []; var_tr = []; var_va = []
    BS = 8
    for e0 in range(0, E, BS):
        e1 = min(e0+BS,E); eb = e1-e0; elist=list(range(e0,e1))
        # fp16 gate/up/down batch
        Wg_fp = load_fp16(il,'gate',elist,I,D)    # (b, I, D)
        Wu_fp = load_fp16(il,'up',elist,I,D)      # (b, I, D)
        Wd_fp = load_fp16(il,'down',elist,D,I)    # (b, D, I)
        # activation-aware ternary for gate and up
        qg = torch.zeros(eb, I, bpc, 256, device=device); gg=torch.zeros(eb,I,bpc,device=device)
        qu = torch.zeros(eb, I, bpc, 256, device=device); gu=torch.zeros(eb,I,bpc,device=device)
        for bc in range(bpc):
            G = G_block[bc]
            # gate: Wg_fp (eb,I,256-slice) @ G
            WG_g = Wg_fp[:, :, bc*256:(bc+1)*256] @ G  # (eb,I,256)
            WG_u = Wu_fp[:, :, bc*256:(bc+1)*256] @ G
            qsg = torch.sign(WG_g); qsu = torch.sign(WG_u)
            thr_g = torch.quantile(WG_g.abs(), 1-NZ, dim=2, keepdim=True)
            thr_u = torch.quantile(WG_u.abs(), 1-NZ, dim=2, keepdim=True)
            qsg[WG_g.abs()<thr_g]=0; qsu[WG_u.abs()<thr_u]=0
            qg[:,:,bc,:]=qsg; qu[:,:,bc,:]=qsu
            zb = Ztr_b[bc]
            qzg = qsg.reshape(eb,-1,1).repeat(1,1,1) # handled below
            # gamma per (expert,row): <qz,Wz>/<qz,qz>
            qz_g = torch.einsum('eri,rn->er n'.replace(' ',''),qsg,zb) if False else qsg.unsqueeze(2)@zb.unsqueeze(0) # (eb,I,1,N)
            # simpler: qz = qsg @ zb : (eb,I,N)
            qz_g = torch.matmul(qsg, zb)            # (eb,I,Ntr)
            Wz_g = torch.matmul(Wg_fp[:, :, bc*256:(bc+1)*256], zb)  # (eb,I,Ntr)
            gg[:,:,bc] = (qz_g*Wz_g).sum(2)/(qz_g**2).sum(2).clamp(min=1e-6)
            qz_u = torch.matmul(qsu, zb)
            Wz_u = torch.matmul(Wu_fp[:, :, bc*256:(bc+1)*256], zb)
            gu[:,:,bc] = (qz_u*Wz_u).sum(2)/(qz_u**2).sum(2).clamp(min=1e-6)
        # reconstruct Wq gate/up
        Wg_q = (qg*gg[:,:,:,None]).reshape(eb, I, D)
        Wu_q = (qu*gu[:,:,:,None]).reshape(eb, I, D)
        # cascade FFN: h = g*silu(u); y = h^T @ down^T
        g_ref = Wg_fp@Ztr; u_ref = Wu_fp@Ztr; h_ref = g_ref*silu(u_ref)
        g_q = Wg_q@Ztr; u_q = Wu_q@Ztr; h_q = g_q*silu(u_q)
        H_ref = h_ref.transpose(1,2); H_q = h_q.transpose(1,2)
        y_ref = torch.matmul(H_ref, Wd_fp.transpose(1,2))   # (eb,Ntr,D)
        y_q = torch.matmul(H_q, Wd_q[e0:e1].transpose(1,2))  # (eb,Ntr,D)
        var_tr.append((y_ref**2).mean(dim=(1,2)))
        mse_ffn_tr.append(((y_q-y_ref)**2).mean(dim=(1,2)))
        # validation
        g_ref_v = Wg_fp@Zva; u_ref_v = Wu_fp@Zva; h_ref_v = g_ref_v*silu(u_ref_v)
        g_q_v = Wg_q@Zva; u_q_v = Wu_q@Zva; h_q_v = g_q_v*silu(u_q_v)
        H_ref_v = h_ref_v.transpose(1,2); H_q_v = h_q_v.transpose(1,2)
        y_ref_v = torch.matmul(H_ref_v, Wd_fp.transpose(1,2))
        y_q_v = torch.matmul(H_q_v, Wd_q[e0:e1].transpose(1,2))
        var_va.append((y_ref_v**2).mean(dim=(1,2)))
        mse_ffn_va.append(((y_q_v-y_ref_v)**2).mean(dim=(1,2)))
        # pack
        for j in range(eb):
            packed_g_out[e0+j] = pack_tq1(qg[j].reshape(I*bpc,256), gg[j].reshape(-1), 54*bpc)
            packed_u_out[e0+j] = pack_tq1(qu[j].reshape(I*bpc,256), gu[j].reshape(-1), 54*bpc)
        del Wg_fp,Wu_fp,Wd_fp,qsg,qsu,WG_g,WG_u,qz_g,Wz_g,qz_u,Wz_u
        torch.cuda.empty_cache()

    var_tr=torch.cat(var_tr); var_va=torch.cat(var_va)
    mse_ffn_tr=torch.cat(mse_ffn_tr); mse_ffn_va=torch.cat(mse_ffn_va)
    from gguf import GGUFReader as _G
    np.save(f'glm5_pod/tq1aw_L{il}_gate.npy', packed_g_out)
    np.save(f'glm5_pod/tq1aw_L{il}_up.npy', packed_u_out)
    rel_tr = (mse_ffn_tr/var_tr).mean()*100
    rel_va = (mse_ffn_va/var_va).mean()*100
    results[il] = dict(ffn_tr=rel_tr.item(), ffn_va=rel_va.item())
    print(f"L{il}: FFN out rel train={rel_tr:.0f}% val={rel_va:.0f}%  ({time.time()-t0:.0f}s)", flush=True)
    del Ztr,Zva,packed_g_out,packed_u_out,Wd_q; torch.cuda.empty_cache()

with open('glm5_pod/actaware_results.json','w') as f:
    json.dump({str(k):{kk:round(vv,3) for kk,vv in v.items()} for k,v in results.items()}, f, indent=2)
print("\nDONE all layers", flush=True)