"""Per-channel absmean ternary (HAGI/BitNet b1.58 scheme) for all layers.
Optimal per-output-channel scale s=mean|W| along input dim. TEST: does this
cleanest scheme beat skel8 (PPL 27.5) at same 1.58 bpw?
No G-matrix (fast, vectorized). Packs into tq1_0 (all 16 blocks per row = s_row).
"""
import numpy as np, os, sys, io, json, time, glob, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch
D, I, E = 4096, 2048, 288
bpc = D // 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
def load_batch(il, p, e_list, o_dim, i_dim):
    from safetensors import safe_open
    kt = f'model.language_model.layers.{il}.mlp.experts.{{e}}.{p}_proj.weight'
    res = np.empty((len(e_list), o_dim, i_dim), dtype=np.float32)
    for j, e in enumerate(e_list):
        key = kt.format(e=e)
        with safe_open(f'{HF}/{_wm[key]}', 'pt', 'cpu') as sf:
            w = sf.get_tensor(key).to(torch.float32)
            s = sf.get_tensor(key.replace('.weight','.weight_scale_inv')).to(torch.float32)
            res[j] = (w.reshape(s.shape[0],128,s.shape[1],128)*s[:,None,:,None]).reshape(o_dim,i_dim).numpy()
    return res

def pack_ternary(Wq, s, R):
    """Wq (nrows,in), s (nrows,). Pack tq1_0. Each row -> in/256 blocks, all gamma=s[row]."""
    nrows, inn = Wq.shape
    nblocks = nrows * (inn//256)
    # ternary = Wq already = q*s; recover q = round(Wq/s)
    s_b = np.repeat(s, inn//256)  # per block gamma (nblocks,)
    B = Wq.reshape(nblocks, 256)
    flat = np.clip(np.round(B / np.maximum(s_b[:,None],1e-8)).astype(np.int32) + 1, 0, 2)
    qs = np.zeros((nblocks, 48), np.int32); qh = np.zeros((nblocks, 4), np.int32)
    for n, c in enumerate([81, 27, 9, 3, 1]):
        qs[:, :32] += flat[:, n*32:(n+1)*32] * c
        qs[:, 32:] += flat[:, 160+n*16:160+(n+1)*16] * c
    for m, c in enumerate([81, 27, 9, 3]):
        qh += flat[:, 240+m*4:244+m*4] * c
    def comp(qq): return ((qq.astype(np.uint32) * 256 + 242) // 243).astype(np.uint8)
    packed = np.concatenate([comp(qs), comp(qh), s_b.astype(np.float16).view(np.uint8).reshape(-1,2)],1)
    return packed.reshape(nrows, R)

from gguf import GGUFReader
r = GGUFReader('glm5_gguf/glm5-ternary-skel8.gguf')
tensors = {t.name: t for t in r.tensors}
# Skel8 raw data shapes to confirm R per matrix
by = tensors
for tn in ['blk.3.ffn_gate_exps.weight','blk.3.ffn_up_exps.weight','blk.3.ffn_down_exps.weight']:
    t = by[tn]
    pn = tn.split('.')[2]
    print(f"{pn}: R={t.data.shape[-1]} raw={t.data.shape} logical={list(t.shape)}")
print("Building absmean ternary...", flush=True)

START = int(sys.argv[1]) if len(sys.argv)>1 else 3
END = int(sys.argv[2]) if len(sys.argv)>2 else 46
os.makedirs('glm5_pod', exist_ok=True)
BS = 12
for il in range(START, END):
    t0 = time.time()
    for pname, o_dim, i_dim, R in [('gate', I, D, 864), ('up', I, D, 864), ('down', D, I, 432)]:
        out = np.empty((E, o_dim, R), dtype=np.uint8)
        for e0 in range(0, E, BS):
            e1 = min(e0+BS, E)
            W = load_batch(il, pname, list(range(e0,e1)), o_dim, i_dim)  # (b, o, i)
            s = np.mean(np.abs(W), axis=2, keepdims=True)  # (b, o, 1)
            q = np.clip(np.round(W/np.maximum(s,1e-8)), -1, 1)
            Wq = q * s
            for j in range(e1-e0):
                out[e0+j] = pack_ternary(Wq[j], s[j, :, 0], R)
        np.save(f'glm5_pod/tq1abs_L{il}_{pname}.npy', out)
        del out; gc.collect()
    print(f"L{il}: {time.time()-t0:.0f}s", flush=True)
    torch.cuda.empty_cache()
print("DONE", flush=True)