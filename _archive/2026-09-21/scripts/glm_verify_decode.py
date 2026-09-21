"""Verify TQ1 decode integrity + measure real norm inflation.
Compares: (a) decode→encode roundtrip on skel8 bytes; (b) decoded weight norm
vs fp16 teacher; (c) is the 1.5x inflation consistent across experts.
"""
import numpy as np, os, sys, io, json, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
import torch
D, I, E = 4096, 2048, 288
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def decode_tq1_row(packed_2d):
    n_rows, R = packed_2d.shape
    bpc = R // 54
    n_out = bpc * 256
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
    return (tern * gamma[:, None]).reshape(n_rows, n_out), tern, gamma

def encode_tq1_row(Wq):
    """Wq: (n_rows, n_out) fp32 → packed (n_rows, R). Mirrors llama.cpp tq1 packing."""
    n_rows, n_out = Wq.shape
    bpc = n_out // 256
    R = bpc * 54
    flat = np.clip(np.round(Wq.reshape(n_rows * bpc, 256) / 
                   np.abs(Wq.reshape(n_rows * bpc, 256)).max(1, keepdims=True) * 0.999), -1, 1)
    # quantize to {-1,0,+1}: round(w/max) already gives that roughly
    tern = flat
    # For our purpose: reconstruct ternary indices. We'll just verify decode of known bytes.
    return tern

# Load skel8 gate e0 and decode
from gguf import GGUFReader
r = GGUFReader('glm5_gguf/glm5-ternary-skel8.gguf')
tensors = {t.name: t for t in r.tensors}

# Fresh decode of e0 gate
packed = np.array(tensors['blk.3.ffn_gate_exps.weight'].data[0], copy=True)  # (2048,864)
Wq, tern, gamma = decode_tq1_row(packed)
print(f"e0 gate decoded: Wq shape {Wq.shape}")
print(f"  ternary: nz frac={np.mean(tern!=0):.4f}, unique={np.unique(tern)}")
print(f"  gamma: mean={gamma.mean():.5f} min={gamma.min():.6f} max={gamma.max():.5f}")
print(f"  ||Wq||_F = {np.linalg.norm(Wq):.1f}")

# ROUNDTRIP: re-encode decode-able ternary and check it's consistent
# We know pack formula. Verify by re-packing Wq from decoded ternary.
tern_idx = (tern + 1).astype(np.int32)  # {0,1,2}
nb = tern_idx.shape[0]
qs = np.zeros((nb, 48), np.int32); qh = np.zeros((nb, 4), np.int32)
for n, c in enumerate([81, 27, 9, 3, 1]):
    qs[:, :32] += tern_idx[:, n*32:(n+1)*32] * c
    qs[:, 32:] += tern_idx[:, 160+n*16:160+(n+1)*16] * c
for m, c in enumerate([81, 27, 9, 3]):
    qh += tern_idx[:, 240+m*4:244+m*4] * c
def comp(qq): return ((qq.astype(np.uint32) * 256 + 242) // 243).astype(np.uint8)
repacked = np.concatenate([comp(qs), comp(qh), gamma.astype(np.float16).view(np.uint8).reshape(-1,2)],1).reshape(2048,864)
# Compare with original bytes
byte_match = np.mean(repacked == packed) * 100
print(f"  Roundtrip byte match: {byte_match:.2f}%")

# Now decode ALL experts for L3 gate and get norm distribution
norms_q = []; norms_f = []
HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
from safetensors import safe_open
gate_data = np.array(tensors['blk.3.ffn_gate_exps.weight'].data, copy=True)
for e in range(20):
    Wq_e, tern_e, g_e = decode_tq1_row(gate_data[e])
    norms_q.append(np.linalg.norm(Wq_e))
    # fp16 teacher
    key = f'model.language_model.layers.3.mlp.experts.{e}.gate_proj.weight'
    with safe_open(f'{HF}/{_wm[key]}', 'pt', 'cpu') as sf:
        w = sf.get_tensor(key).to(torch.float32)
        s = sf.get_tensor(key.replace('.weight','.weight_scale_inv')).to(torch.float32)
        Wf = (w.reshape(s.shape[0],128,s.shape[1],128)*s[:,None,:,None]).reshape(I,D).numpy()
    norms_f.append(np.linalg.norm(Wf))
norms_q=np.array(norms_q); norms_f=np.array(norms_f)
print(f"\n  First 20 experts: ||Wq||/||Wf|| ratios:")
print(f"    mean={np.mean(norms_q/norms_f):.3f} min={np.min(norms_q/norms_f):.3f} max={np.max(norms_q/norms_f):.3f}")
print(f"  ALL ratios >1 => systematic inflation, not decode bug (if roundtrip matches)")