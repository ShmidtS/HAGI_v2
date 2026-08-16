"""Pack the reduced HAGI model into a single safetensors bundle.

Collects, for every layer (0..42):
  - POD basis P [4096, 512] and input mean mu [4096]
  - every reduced expert (whatever exists): ternary w1/w3/w2 packed uint8
    + per-row scales, int4 Q packed uint8 (2 nibbles/byte) + scale + Q_bits
  - shared FFN (lossless) w1/w2/w3  [optional --no-shared]
  - router gate weight/bias + tid2eid for hash layers  [optional --no-router]

Writes ONE safetensors file. Metadata records the reduced architecture so a
future loader can reconstruct shapes without scanning files.

Usage:
    python scripts/dsv4_pack_safetensors.py --out dsv4_reduced.safetensors
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch
from safetensors.torch import save_file

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

REDUCED = 'C:/HAGI_v2/dsv4_reduced'
LOSSLESS = 'C:/HAGI_v2/lossless_layers'
KV_POD_DIR = 'C:/HAGI_v2/checkpoints_dsv4/pod_reduced'
N_LAYERS = 43
HASH_LAYERS = {0, 1, 2}
K = 512
INTER = 1024
KP = 512
Q_BITS = 4
Q_DIVISOR = 28
KV_RANK = 256
YARN_FACTOR = 8
PYRAMID_WINDOW = 4096
PYRAMID_MIN_RANK = 16


def pack_layer(li: int, tensors: dict, include_shared: bool, include_router: bool) -> int:
    """Collect one layer's tensors into `tensors`. Returns number of experts."""
    ld = os.path.join(REDUCED, f'layer_{li}')
    P = torch.load(os.path.join(ld, 'P.pt'), map_location='cpu', weights_only=False)
    tensors[f'layers.{li}.P'] = P.float().contiguous()
    mu_path = os.path.join(ld, 'mu.pt')
    if os.path.exists(mu_path):
        mu = torch.load(mu_path, map_location='cpu', weights_only=False)
        tensors[f'layers.{li}.mu'] = mu.float().reshape(-1).contiguous()

    experts = sorted(glob.glob(os.path.join(ld, 'expert_*.pt')),
                     key=lambda p: int(os.path.basename(p)[7:-3]))
    n = 0
    for ep in experts:
        k = int(os.path.basename(ep)[7:-3])
        e = torch.load(ep, map_location='cpu', weights_only=False)
        tensors[f'layers.{li}.experts.{k}.w1'] = e['w1'].to(torch.uint8).contiguous()
        tensors[f'layers.{li}.experts.{k}.w1_scale'] = e['w1_scale'].float().contiguous()
        tensors[f'layers.{li}.experts.{k}.w3'] = e['w3'].to(torch.uint8).contiguous()
        tensors[f'layers.{li}.experts.{k}.w3_scale'] = e['w3_scale'].float().contiguous()
        tensors[f'layers.{li}.experts.{k}.w2'] = e['w2'].to(torch.uint8).contiguous()
        tensors[f'layers.{li}.experts.{k}.w2_scale'] = e['w2_scale'].float().contiguous()
        tensors[f'layers.{li}.experts.{k}.Q'] = e['Q'].to(torch.uint8).contiguous()
        tensors[f'layers.{li}.experts.{k}.Q_scale'] = e['Q_scale'].float().contiguous()
        tensors[f'layers.{li}.experts.{k}.Q_bits'] = torch.tensor([int(e.get('Q_bits', Q_BITS))],
                                                                  dtype=torch.int64)
        n += 1

    # KV-cache POD (attention improvement): basis [512, 256] + mean [512]
    kv_basis_path = os.path.join(KV_POD_DIR, f'P_kv_L{li}.pt')
    if os.path.exists(kv_basis_path):
        pkv = torch.load(kv_basis_path, map_location='cpu', weights_only=False)
        tensors[f'layers.{li}.P_kv'] = pkv.float().contiguous()
    kv_mean_path = os.path.join(KV_POD_DIR, f'mean_kv_L{li}.pt')
    if os.path.exists(kv_mean_path):
        mkv = torch.load(kv_mean_path, map_location='cpu', weights_only=False)
        tensors[f'layers.{li}.mean_kv'] = mkv.float().reshape(-1).contiguous()

    if include_shared:
        fp = os.path.join(LOSSLESS, f'layers_{li}_ffn.safetensors')
        if os.path.exists(fp):
            from safetensors import safe_open
            prefix = f'layers.{li}.ffn'
            with safe_open(fp, framework='pt', device='cpu') as f:
                for key in f.keys():
                    tensors[f'shared.{li}.{key[len(prefix) + 1:]}'] = f.get_tensor(key).float().contiguous()

    if include_router:
        import dsv4_experts as de
        snap = de.default_snapshot()
        wm = de.load_index(snap)['weight_map']
        p = f'layers.{li}.ffn.gate'
        tensors[f'router.{li}.weight'] = de.read_tensor(snap, wm, f'{p}.weight', device='cpu').float().contiguous()
        if li in HASH_LAYERS:
            tensors[f'router.{li}.tid2eid'] = de.read_tensor(snap, wm, f'{p}.tid2eid', device='cpu').long().contiguous()
        else:
            tensors[f'router.{li}.bias'] = de.read_tensor(snap, wm, f'{p}.bias', device='cpu').float().contiguous()

    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='C:/HAGI_v2/dsv4_reduced.safetensors')
    ap.add_argument('--no-shared', action='store_true')
    ap.add_argument('--no-router', action='store_true')
    ap.add_argument('--start-layer', type=int, default=0)
    ap.add_argument('--end-layer', type=int, default=N_LAYERS)
    args = ap.parse_args()

    tensors: dict = {}
    total_experts = 0
    for li in range(args.start_layer, args.end_layer):
        n = pack_layer(li, tensors, not args.no_shared, not args.no_router)
        total_experts += n
        print(f'layer {li}: {n} experts collected (total {total_experts})', flush=True)

    metadata = {
        'format': 'hagi-reduced-v1',
        'n_layers': str(args.end_layer - args.start_layer),
        'n_experts': str(total_experts),
        'k': str(K),
        'inter': str(INTER),
        'kp': str(KP),
        'q_bits': str(Q_BITS),
        'q_divisor': str(Q_DIVISOR),
        'ternary_pack': '5-trit-per-byte',
        'q_pack': '2-nibbles-per-byte-int4',
        'kv_rank': str(KV_RANK),
        'yarn_factor': str(YARN_FACTOR),
        'pyramid_window': str(PYRAMID_WINDOW),
        'pyramid_min_rank': str(PYRAMID_MIN_RANK),
    }
    save_file(tensors, args.out, metadata=metadata)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f'wrote {args.out}: {len(tensors)} tensors, {size_mb:.0f} MB, {total_experts} experts', flush=True)


if __name__ == '__main__':
    main()
