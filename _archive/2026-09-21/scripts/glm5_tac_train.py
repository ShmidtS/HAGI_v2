"""Train GLM-5 TAC correction adapters on one layer.

Architecture: one MoE layer (L3, 288 experts) with ternary STE + correction
adapter + per-expert alpha. Training objective: MSE against fp16 teacher's
FFN output on cascade activations (dumped from skel8).

No post-hoc pattern selection — the adapter LEARNS to correct the cascade
error through backprop.

Usage: python scripts/glm5_tac_train.py [--layer 3] [--rank 8] [--lr 1e-4] [--steps 500]
"""
import os, sys, io, time, json, glob, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.chdir(r'C:/HAGI_v2')
sys.path.insert(0, 'glm5_pod')
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

D, I, E_PER_LAYER = 4096, 2048, 288
HF = '//wsl.localhost/Ubuntu-24.04/mnt/samsung/home/shmid/glm53'
_wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']

def load_fp16_expert(il, e, pname):
    """Load fp16 teacher weight for one expert. pname=gate_proj|up_proj|down_proj."""
    from safetensors import safe_open
    key = f'model.language_model.layers.{il}.mlp.experts.{e}.{pname}.weight'
    with safe_open(f'{HF}/{_wm[key]}', 'pt', 'cpu') as sf:
        w = sf.get_tensor(key).to(torch.float32)
        s = sf.get_tensor(key.replace('.weight','.weight_scale_inv')).to(torch.float32)
    return (w.reshape(s.shape[0],128,s.shape[1],128)*s[:,None,:,None]).reshape(w.shape[0],w.shape[1]).numpy()

def compute_teacher_ffn(il, z, e_list):
    """Compute fp16 teacher FFN output for given experts and input.
    z: (D, N) or (N, D) numpy. Returns (N, D) fp32."""
    if z.ndim == 2 and z.shape[0] == D:
        z = z.T  # (N, D)
    N = z.shape[0]
    out = np.zeros((N, D), dtype=np.float32)
    router_w = np.fromfile(f'glm5_gguf/dump_moe_in_skel8/moe_in_L{il}_router_expert_weights.f32', dtype=np.float32)
    # Load router weights from the dump directory
    # GLM-5: the gate_inp (D, E) determines expert selection
    from gguf import GGUFReader
    r = GGUFReader('glm5_gguf/glm5-ternary-skel8.gguf')
    gate_inp = np.array([t.data for t in r.tensors if f'blk.{il}.ffn_gate_inp' in t.name][0])  # (D, E)
    logits = z @ gate_inp.astype(np.float32).T  # (N, E) — gate_inp is (E, D)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights = weights / weights.sum(axis=1, keepdims=True)
    topk = min(6, E_PER_LAYER)
    top_idx = np.argsort(-logits, axis=1)[:, :topk]
    top_w = np.take_along_axis(weights, top_idx, axis=1)
    top_w = top_w / top_w.sum(axis=1, keepdims=True)
    
    # Only compute for experts that are actually used (top-k)
    used = np.unique(top_idx)
    expert_out = {}
    for e in used:
        Wg = load_fp16_expert(il, e, 'gate_proj')
        Wu = load_fp16_expert(il, e, 'up_proj')
        Wd = load_fp16_expert(il, e, 'down_proj')
        g = z @ Wg.T * (1.0 / (1.0 + np.exp(-(z @ Wu.T))))  # silu(gate) = gate * sigmoid(gate)
        # Actually silu = x * sigmoid(x); compute properly
        gate_h = z @ Wg.T
        up_h = z @ Wu.T
        h = gate_h * (1.0 / (1.0 + np.exp(-gate_h))) * up_h  # silu(gate) * up
        expert_out[e] = h @ Wd.T
    
    for n in range(N):
        for k in range(topk):
            e = top_idx[n, k]
            w = top_w[n, k]
            out[n] += w * expert_out[e][n]
    return out

def compute_teacher_targets(il, z_token_by_token):
    """Simpler: compute FFN output per expert independently, return dict.
    Each value: (I, N) gate+up pre-activation, or just return expert FFN output (N,I)->(N,D)."""
    # We'll compute per-expert FFN output on the fly during training
    pass

# Actually — simplest approach: teacher_target = z (identity) + learn adapter to
# match what z should become. No — we need the real teacher output.

# Better approach: use pre-computed fp16 model to produce target FFN outputs
# for each expert. Cache them per-expert on first use.

def target_for_expert(il, e, z_np):
    """Compute fp16 teacher output for ONE expert.
    z_np: (N, D) numpy. Returns (N, D) the expert's FFN contribution.
    The output = down(silu(gate(z)) * up(z))."""
    Wg = torch.from_numpy(load_fp16_expert(il, e, 'gate_proj')).to(device)
    Wu = torch.from_numpy(load_fp16_expert(il, e, 'up_proj')).to(device)
    Wd = torch.from_numpy(load_fp16_expert(il, e, 'down_proj')).to(device)
    z_t = torch.from_numpy(z_np).to(device).float()
    with torch.no_grad():
        g = F.silu(z_t @ Wg.T)
        u = z_t @ Wu.T
        h = g * u
        out = h @ Wd.T
    return out

# Main training
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, default=3)
parser.add_argument('--rank', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--steps', type=int, default=500)
parser.add_argument('--n_tokens', type=int, default=1024)
parser.add_argument('--log_every', type=int, default=50)
args = parser.parse_args()

il = args.layer
rank = args.rank
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device} | Layer: L{il} | Rank: {rank} | LR: {args.lr} | Steps: {args.steps}", flush=True)

from glm5_tac import AdaExpert, TACRouter, GLM5TAC, E_PER_LAYER, D

# Load cascade input activations
mats = []
for f in sorted(glob.glob(f'glm5_gguf/dump_moe_in_skel8/moe_in_L{il}_*.f32')):
    a = np.fromfile(f, dtype=np.float32)
    n = len(a)//D
    if n: mats.append(a[:n*D].reshape(n, D))
z_all = np.concatenate(mats, 0).astype(np.float32)
print(f"Loaded {len(z_all)} tokens from L{il}", flush=True)

# Split train/val
N_train = min(args.n_tokens, len(z_all) - 200)
N_val = min(200, len(z_all) - N_train)
z_tr = z_all[:N_train]
z_va = z_all[N_train:N_train+N_val]
print(f"Train: {z_tr.shape[0]}, Val: {z_va.shape[0] if z_va is not None else 0}", flush=True)

# Build a single-layer model with correction adapters
# We use top-6 routing and only train adapters on the top-6-most-used experts
# to save compute. Actually train ALL adapters.
print("Building AdaExperts (288)...", flush=True)
t0 = time.time()
experts = nn.ModuleList([AdaExpert(il, e, rank=rank) for e in range(E_PER_LAYER)])
experts.to(device)
print(f"Built in {time.time()-t0:.0f}s", flush=True)

# Router (trainable too, but we'll fix it for this test — use fp16 teacher's router)
from gguf import GGUFReader
r = GGUFReader('glm5_gguf/glm5-ternary-skel8.gguf')
gate_inp_np = np.array([t.data for t in r.tensors if f'blk.{il}.ffn_gate_inp' in t.name][0]).astype(np.float32)
gate_inp = torch.from_numpy(gate_inp_np).to(device)

def route(z_t):
    logits = z_t @ gate_inp.T  # (N, E) — gate_inp is (E, D)
    weights = F.softmax(logits.float(), dim=-1)
    top_w, top_idx = weights.topk(6, dim=-1)
    return top_w, top_idx

# Training: compute teacher targets per used expert in each step
# Alternative: pre-compute all expert outputs and cache
print("Pre-computing teacher targets for frequently used experts...", flush=True)

# Determine most-used experts
z_short = torch.from_numpy(z_tr[:min(512, len(z_tr))]).to(device)
_, top_idx = route(z_short)
used = top_idx.cpu().unique().tolist()
print(f"Frequent experts ({len(used)}/288)", flush=True)

# Pre-compute teacher output for ALL experts that appear in train set
teacher_cache = {}
for e in used:
    with torch.no_grad():
        teacher_cache[e] = target_for_expert(il, e, z_tr)

# Optimizer: only adapter + alpha params
adapter_params = []
for expert in experts:
    for name, p in expert.named_parameters():
        if 'adapter' in name or 'log_alpha' in name:
            adapter_params.append(p)
print(f"Trainable: {len(adapter_params)} tensors", flush=True)

optim = torch.optim.AdamW(adapter_params, lr=args.lr, weight_decay=0.01)

# Training loop
print(f"\n--- Training L{il} correction adapters ---", flush=True)
t0 = time.time()
for step in range(args.steps):
    optim.zero_grad()
    z_t = torch.from_numpy(z_tr).to(device)
    
    # Teacher routing
    top_w, top_idx = route(z_t)  # (N, 6), (N, 6)
    
    # Student forward with adapters + alpha
    # Dispatch each token to its experts
    N = z_t.shape[0]
    student_total = torch.zeros(N, D, device=device)
    teacher_total = torch.zeros(N, D, device=device)
    
    nll = 0  # load loss
    for e_id in used:
        mask = (top_idx == e_id)
        if not mask.any():
            continue
        positions = mask.nonzero(as_tuple=False)
        tok_idx = positions[:, 0]
        selected = z_t[tok_idx]
        
        # Student
        out_s = experts[e_id](selected)  # (n_selected, D), with correction
        # Teacher
        out_t = teacher_cache[e_id].to(device)  # cached on first use
        # (but teacher cache was computed on all z_tr, so index)
        
        w = top_w[tok_idx, positions[:, 1]].unsqueeze(-1)
        student_total.index_add_(0, tok_idx, out_s * w)
        nll_contrib = (w.squeeze() * (out_s - out_t[tok_idx]).pow(2).mean(dim=1))
        nll = nll + nll_contrib.sum()
    
    loss = nll / N
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
    optim.step()
    
    if step % args.log_every == 0 or step == args.steps - 1:
        with torch.no_grad():
            # Validation on val split
            z_v = torch.from_numpy(z_va).to(device)
            top_w_v, top_idx_v = route(z_v)
            Nv = z_v.shape[0]
            val_mse = 0.0
            for e_id in used:
                mask = (top_idx_v == e_id)
                if not mask.any(): continue
                positions = mask.nonzero(as_tuple=False)
                tok_idx = positions[:, 0]
                selected = z_v[tok_idx]
                out_s = experts[e_id](selected)
                out_t = teacher_cache[e_id][tok_idx].to(device)
                w = top_w_v[tok_idx, positions[:, 1]].unsqueeze(-1)
                diff = (out_s * w - out_t * w).pow(2).mean()
                val_mse = val_mse + diff * tok_idx.shape[0]
            val_mse = val_mse / Nv
            
            # Expert balance
            counts = top_idx.unique(return_counts=True)
            balance = counts[1].float().std() / counts[1].float().mean()
            
            # First expert's alpha
            alpha0 = experts[0].alpha.item()
            # First expert adapter gain
            gain0 = experts[0].adapter.gain.item()
            
            dt = time.time() - t0
            print(f"step {step:4d} | train_mse={loss.item():.4e} val_mse={val_mse:.4e} | "
                  f"gn={gn:.4e} α0={alpha0:.4f} gain0={gain0:.4f} "
                  f"balance={(balance if isinstance(balance, float) else balance.item()):.4f} "
                  f"step/s={step/(dt+1e-8):.2f}", flush=True)

# Final alpha distribution
alphas = [experts[e].alpha.item() for e in used]
print(f"\nDone: L{il} correction trained.", flush=True)
print(f"Alpha: mean={np.mean(alphas):.4f} std={np.std(alphas):.4f} max={max(alphas):.4f} min={min(alphas):.4f}", flush=True)

# Save correction params
sd = {}
for e in range(E_PER_LAYER):
    for name, p in experts[e].named_parameters():
        if 'adapter' in name or 'log_alpha' in name:
            sd[f'L{il}_e{e}_{name}'] = p.detach().cpu()
torch.save(sd, f'glm5_pod/tac_L{il}_correction.pt')
print(f"Corrections saved to tac_L{il}_correction.pt ({len(sd)} tensors)", flush=True)