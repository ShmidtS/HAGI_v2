"""Decisive test: per-output-channel absmean ternary (HAGI/BitNet b1.58 scheme)
vs block-max-gamma (skel8) for FULL FFN cascade on held-out activations.
Measures gate, up, and FFN-output rel error — does per-channel scale fix the cascade?
"""
import numpy as np, os, sys, io, json, time, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch
D, I, E = 4096, 2048, 288
device = 'cuda' if torch.cuda.is_available() else 'cpu'

HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
def load_fp16(il, p, e):
    from safetensors import safe_open
    key = f'model.language_model.layers.{il}.mlp.experts.{e}.{p}_proj.weight'
    with safe_open(f'{HF}/{_wm[key]}', 'pt', 'cpu') as sf:
        w = sf.get_tensor(key).to(torch.float32)
        s = sf.get_tensor(key.replace('.weight','.weight_scale_inv')).to(torch.float32)
        return (w.reshape(s.shape[0],128,s.shape[1],128)*s[:,None,:,None]).reshape(w.shape[0], w.shape[1]).numpy()

def cal(il, n_use):
    mats=[]
    for f in sorted(glob.glob(f'glm5_gguf/dump_moe_in_skel8/moe_in_L{il}_*.f32')):
        a=np.fromfile(f,dtype=np.float32); n=len(a)//D
        if n: mats.append(a[:n*D].reshape(n,D))
    z=np.concatenate(mats,0).astype(np.float32)
    return z[:n_use].T  # (D, m)

def absmean_ternary(Wf):
    """Per-output-channel absmean scale (b1.58). Wf: (out, in)"""
    s = np.mean(np.abs(Wf), axis=1, keepdims=True)
    q = np.clip(np.round(Wf/np.maximum(s,1e-8)), -1, 1)
    return q*s

def maxgamma_ternary(Wf):
    """Block-max gamma (skel8), 256-col blocks. Returns same shape."""
    out, inn = Wf.shape
    nb = inn//256
    B = Wf.reshape(out, nb, 256)
    g = np.abs(B).max(2, keepdims=True)
    q = np.clip(np.round(B*0.999/g), -1, 1)
    return (q*g).reshape(out, inn)

def silu(x): return x/(1+np.exp(-x))

# measure on a few experts, layer 3
il=3
Ztr = cal(il,2048); Zva = cal(il,3000)[:, 2048:3000]
def cascade_err(exps, scheme_f):
    # returns mean rel FFN output error on train and val
    g_err_tr=[]; g_err_va=[]; ffn_tr=[]; ffn_va=[]
    for e in exps:
        Wg=load_fp16(il,'gate',e); Wu=load_fp16(il,'up',e); Wd=load_fp16(il,'down',e)
        Wgq=scheme_f(Wg); Wuq=scheme_f(Wu); Wdq=scheme_f(Wd)
        # train
        g_ref=Wg@Ztr; u_ref=Wu@Ztr; h_ref=g_ref*silu(u_ref)
        g_q=Wgq@Ztr; u_q=Wuq@Ztr; h_q=g_q*silu(u_q)
        y_ref=h_ref.T@Wd.T; y_q=h_q.T@Wdq.T
        g_err_tr.append(np.mean((g_q-g_ref)**2)/np.mean(g_ref**2))
        ffn_tr.append(np.mean((y_q-y_ref)**2)/np.mean(y_ref**2))
        # val
        g_ref=Wg@Zva; u_ref=Wu@Zva; h_ref=g_ref*silu(u_ref)
        g_q=Wgq@Zva; u_q=Wuq@Zva; h_q=g_q*silu(u_q)
        y_ref=h_ref.T@Wd.T; y_q=h_q.T@Wdq.T
        g_err_va.append(np.mean((g_q-g_ref)**2)/np.mean(g_ref**2))
        ffn_va.append(np.mean((y_q-y_ref)**2)/np.mean(y_ref**2))
    return (np.mean(g_err_tr),np.mean(g_err_va),np.mean(ffn_tr),np.mean(ffn_va))

exps=[0,1,5,50,200]
print("=== L3 cascade: block-max-gamma (skel8) ===")
gt,gv,ft,fv=cascade_err(exps,maxgamma_ternary)
print(f"  gate rel: train={gt*100:.0f}% val={gv*100:.0f}% | FFN rel: train={ft*100:.0f}% val={fv*100:.0f}%")
print("=== L3 cascade: per-channel absmean (b1.58) ===")
gt,gv,ft,fv=cascade_err(exps,absmean_ternary)
print(f"  gate rel: train={gt*100:.0f}% val={gv*100:.0f}% | FFN rel: train={ft*100:.0f}% val={fv*100:.0f}%")
print("DONE", flush=True)