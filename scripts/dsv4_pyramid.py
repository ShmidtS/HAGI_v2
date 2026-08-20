"""Pyramid layer rotation — research prototype (does NOT touch the flat pipeline).

Rotates the 43 flat FFN layers into a pyramid of parallel branches, reusing
the existing reduced experts (ternary core + int4 Q) and POD input projection.

Topology:
    flat:     x -> L0 -> L1 -> ... -> L42
    pyramid:  x -> [L0..L15] -> mix -> [L16..L23] -> mix -> ... -> y
    Branches of a level run in PARALLEL on the SAME input; their outputs are
    mixed (residual + mean) and fed to the next level.

Scope / limitations (documented honestly):
    - Rotates the FFN/MoE path only. Attention is left sequential (causal KV)
      and is OUT OF SCOPE here — it changes `x` in its own way and would need
      its own re-collection.
    - Hash layers (0,1,2) need token ids for tid2eid routing; without ids the
      prototype falls back to top-k score routing (same as MoE layers).
    - The honest refit target is the ORIGINAL (exact) expert output on the
      pyramid's mixed inputs. This prototype collects (z, y) under the new
      mixing from the REDUCED experts as a warm-start demonstration; a full
      run should collect `y` from the exact checkpoint before refit.

Usage:
    python scripts/dsv4_pyramid.py [--n-layers N] [--refit] [--steps S]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from dsv4_generate_reduced import (
    unpack_ternary,
    unpack_int4,
    load_shared,
    N_LAYERS,
    HASH_LAYERS,
    TOP_K,
    ROUTED_SCALE,
    SWIGLU_LIMIT,
    REDUCED,
)
import dsv4_experts as de

K = 512
D = 4096
POD_ACTS = 'checkpoints_dsv4/pod_all_tokens'


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------

def pyramid_plan(n_layers=N_LAYERS, factor=2, wide_base=True):
    """Split n_layers into pyramid levels.

    wide_base=True : narrow top -> wide base  (1, 2, 4, 8, 16, remainder)
    wide_base=False: wide base -> narrow top  (reverse order)

    Returns list of levels; each level is a list of layer indices (branches).
    """
    sizes = []
    s = 1
    while sum(sizes) + s <= n_layers:
        sizes.append(s)
        s *= factor
    rem = n_layers - sum(sizes)
    if rem > 0:
        sizes.append(rem)
    sizes = [sz for sz in sizes if sz > 0]
    if not wide_base:
        sizes = sizes[::-1]
    levels, i = [], 0
    for sz in sizes:
        levels.append(list(range(i, i + sz)))
        i += sz
    return levels


# --------------------------------------------------------------------------
# Layer FFN (shared + top-k MoE), branch of a pyramid level
# --------------------------------------------------------------------------

def reduced_ffn_z(z, e):
    """Ternary SwiGLU + int4 Q forward, with ternary padding trimmed.

    Mirrors current_resid/resid_weights_full from dsv4_refit_experts.py:
    the packed ternary weights decode to padded width (e.g. 515), so they are
    sliced to the true POD rank Kk = z.shape[1] before the matmul.
    """
    Kk = z.shape[1]
    zz = z.to(torch.bfloat16)
    w1 = unpack_ternary(e['w1']).to(torch.bfloat16)[:, :Kk] * e['w1s'].to(torch.bfloat16)[:, None]
    w3 = unpack_ternary(e['w3']).to(torch.bfloat16)[:, :Kk] * e['w3s'].to(torch.bfloat16)[:, None]
    w2 = unpack_ternary(e['w2']).to(torch.bfloat16) * e['w2s'].to(torch.bfloat16)[:, None]
    Q = unpack_int4(e['Q']).to(torch.bfloat16) * e['Qs'].to(torch.bfloat16)[None, :]
    gate = (zz @ w1.T).clamp(max=SWIGLU_LIMIT)
    up = (zz @ w3.T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    h = F.silu(gate) * up
    w2 = w2[:, :h.shape[1]]  # trim inter padding
    return (h @ w2.T @ Q.T).float()


def load_reduced_layer_sparse(li):
    """Like load_reduced_layer, but tolerant to incomplete layers (refit in
    progress): loads P/mu and only the expert_*.pt files that exist."""
    d = os.path.join(REDUCED, f'layer_{li}')
    P = torch.load(os.path.join(d, 'P.pt'), map_location='cuda')
    mu_path = os.path.join(d, 'mu.pt')
    mu = torch.load(mu_path, map_location='cuda') if os.path.exists(mu_path) else None
    experts = {}
    for fp in sorted(glob.glob(os.path.join(d, 'expert_*.pt'))):
        k = int(os.path.basename(fp)[len('expert_'):-len('.pt')])
        e = torch.load(fp, map_location='cuda')
        experts[k] = {
            'w1': e['w1'], 'w1s': e['w1_scale'].to('cuda'),
            'w3': e['w3'], 'w3s': e['w3_scale'].to('cuda'),
            'w2': e['w2'], 'w2s': e['w2_scale'].to('cuda'),
            'Q': e['Q'].to('cuda'), 'Qs': e['Q_scale'].to('cuda'),
        }
    return {'P': P, 'mu': mu, 'experts': experts}


def ffn_layer(flat, li, red_cache, shared_cache, router, current_ids=None):
    """One layer's FFN: shared expert + top-k reduced MoE experts.

    Returns (out [n,D], z [n,K], per_expert_y {expert_id: y [n,D]}).
    `z` is the POD projection of the layer input (shared by all experts of the
    branch); `per_expert_y` holds the 4096-dim output of each routed expert.
    """
    w = router['w'][li]
    scores = F.softplus(flat @ w.T).sqrt()
    if li in HASH_LAYERS and current_ids is not None:
        indices = router['tid'][li][current_ids.reshape(-1)]
    else:
        bias = router['bias'].get(li)
        if bias is not None:
            indices = torch.topk(scores + bias, TOP_K, dim=-1).indices
        else:
            indices = torch.topk(scores, TOP_K, dim=-1).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * ROUTED_SCALE

    sh = shared_cache[li]
    out = (F.silu((flat @ sh['w1'].T).clamp(max=SWIGLU_LIMIT))
           * (flat @ sh['w3'].T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)) @ sh['w2'].T

    red = red_cache[li]
    if red.get('mu') is not None:
        z = (flat - red['mu']) @ red['P']
    else:
        z = flat @ red['P']

    per_expert_y = {}
    for k in indices.unique().tolist():
        expert = red['experts'].get(k)
        if expert is None:
            # expert not yet refit (incomplete layer) -> skip its contribution
            continue
        y = reduced_ffn_z(z, expert)
        per_expert_y[k] = y
        for kk in range(TOP_K):
            m = indices[:, kk] == k
            if m.any():
                out[m] += weights[m, kk, None] * y[m]
    return out, z, per_expert_y


# --------------------------------------------------------------------------
# Pyramid forward + activation collection
# --------------------------------------------------------------------------

def pyramid_ffn_forward(x, plan, red_cache, shared_cache, router,
                        current_ids=None, mix='mean'):
    """Run x through pyramid levels, collecting per-expert (z, y).

    Returns (final_x [n,D], collected).
    collected: list over levels of dict {(li, expert_id): (z, y)}.
    """
    x = x.float()
    collected = []
    for level in plan:
        branch_outs = []
        level_acts = {}
        for li in level:
            out, z, per_expert_y = ffn_layer(
                x, li, red_cache, shared_cache, router, current_ids)
            branch_outs.append(out)
            for k, y in per_expert_y.items():
                level_acts[(li, k)] = (z.clone(), y.clone())
        collected.append(level_acts)

        stacked = torch.stack(branch_outs, dim=0)  # [B, n, D]
        if mix == 'mean':
            mix_out = stacked.mean(dim=0)
        elif mix == 'sum':
            mix_out = stacked.sum(dim=0)
        else:
            raise ValueError(f'unknown mix {mix!r}')
        x = x + mix_out  # residual around the level
    return x, collected


def flat_ffn_forward(x, n_layers, red_cache, shared_cache, router,
                     current_ids=None):
    """Baseline: sequential flat forward (same FFN layers, no rotation)."""
    x = x.float()
    for li in range(n_layers):
        out, _, _ = ffn_layer(x, li, red_cache, shared_cache, router, current_ids)
        x = x + out  # residual, standard transformer
    return x


# --------------------------------------------------------------------------
# Refit preparation (reuses train_batch from the flat pipeline)
# --------------------------------------------------------------------------

def collect_refit_pairs(collected):
    """Flatten collected {(li,k): (z,y)} into refit pairs grouped by branch.

    Returns {li: list of (z [n,K], y [n,D])} for each branch layer.
    """
    by_layer = {}
    for level_acts in collected:
        for (li, k), (z, y) in level_acts.items():
            by_layer.setdefault(li, []).append((z, y))
    return by_layer


def refit_pyramid(pairs_by_layer, steps=1000, stop_threshold=1e-4):
    """Refit each branch's experts on the pyramid-collected activations.

    `pairs_by_layer`: {li: list of (z [n,K], y [n,D])}.
    Runs train_batch per (li, k) pair (one expert = one communication channel).
    """
    from dsv4_refit_experts import train_batch, safe_svd_q, pick_config, K as _K

    results = {}
    for li, pairs in pairs_by_layer.items():
        for (z, y) in pairs:
            n_k = z.shape[0]
            if n_k == 0:
                continue
            inter, kp = pick_config(n_k)
            Q0 = safe_svd_q(y.float().cuda(), kp)
            trained = train_batch(
                [(z.float().cuda(), y.float().cuda(), Q0)],
                inter=inter, steps=steps, kp=kp,
                check_every=25, patience=200, stop_threshold=stop_threshold,
                n_real=n_k)
            results[(li, id(z))] = trained
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-layers', type=int, default=8,
                    help='how many flat layers to rotate into the pyramid')
    ap.add_argument('--factor', type=int, default=2)
    ap.add_argument('--wide-base', action='store_true',
                    help='wide base (narrow top), else narrow base')
    ap.add_argument('--refit', action='store_true')
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--mix', default='mean', choices=['mean', 'sum'])
    args = ap.parse_args()

    plan = pyramid_plan(args.n_layers, args.factor, args.wide_base)
    print('pyramid plan (levels -> layers):')
    for i, lvl in enumerate(plan):
        print(f'  level {i}: {lvl}  ({len(lvl)} branches)')

    torch.set_default_device('cuda')
    print('loading shared experts...', flush=True)
    shared_cache = load_shared()
    print('loading router...', flush=True)
    snap = de.default_snapshot()
    wm = de.load_index(snap)['weight_map']
    router = {'w': {}, 'bias': {}, 'tid': {}}
    for li in range(args.n_layers):
        p = f'layers.{li}.ffn.gate'
        router['w'][li] = de.read_tensor(snap, wm, f'{p}.weight', device='cuda').to(torch.float32)
        if li in HASH_LAYERS:
            router['tid'][li] = de.read_tensor(snap, wm, f'{p}.tid2eid', device='cuda').to(torch.long)
        else:
            router['bias'][li] = de.read_tensor(snap, wm, f'{p}.bias', device='cuda').to(torch.float32)

    print(f'loading reduced experts for {args.n_layers} layers...', flush=True)
    red_cache = {li: load_reduced_layer_sparse(li) for li in range(args.n_layers)}

    # Real activation inputs: concatenate all layer-0 expert inputs as the
    # pyramid's x0 (approximation of the full FFN input stream).
    acts0 = torch.load(os.path.join(POD_ACTS, 'acts_layer0.pt'),
                       map_location='cuda', weights_only=False)
    x0 = torch.cat([v[0].float().cuda() for v in acts0.values()], dim=0)
    print(f'x0 shape: {tuple(x0.shape)} (from {len(acts0)} layer-0 experts)', flush=True)

    t0 = time.time()
    with torch.no_grad():
        xp, collected = pyramid_ffn_forward(
            x0, plan, red_cache, shared_cache, router, current_ids=None, mix=args.mix)
    print(f'pyramid forward: {time.time()-t0:.1f}s, out norm {xp.norm().item():.2f}')

    n_pairs = sum(len(v) for v in collect_refit_pairs(collected).values())
    print(f'collected {n_pairs} per-expert activation pairs across '
          f'{len(plan)} levels / {args.n_layers} branches')

    # Coverage sanity: how many distinct experts got routed per level.
    for i, level_acts in enumerate(collected):
        experts = sorted({k for (li, k) in level_acts})
        print(f'  level {i}: {len(experts)} distinct experts routed')

    if args.refit:
        print('refit on pyramid activations...', flush=True)
        by_layer = collect_refit_pairs(collected)
        results = refit_pyramid(by_layer, steps=args.steps, stop_threshold=1e-4)
        print(f'refit done: {len(results)} expert fits')

    torch.set_default_device('cpu')


if __name__ == '__main__':
    main()
