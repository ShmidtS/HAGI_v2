"""Bench inference speed: int8 Q (dequant per call) vs bf16 Q (pre-dequantized)."""
import torch, torch.nn.functional as F, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsv4_generate_reduced import unpack_ternary

K = 512
INTER = 4096
KP = 384
SWIGLU_LIMIT = 10.0
REDUCED = 'dsv4_reduced'

e = torch.load(os.path.join(REDUCED, 'layer_0', 'expert_0.pt'),
               map_location='cuda', weights_only=False)

w1 = unpack_ternary(e['w1'])[:, :K].to(torch.bfloat16) * e['w1_scale'].to('cuda').to(torch.bfloat16)[:, None]
w3 = unpack_ternary(e['w3'])[:, :K].to(torch.bfloat16) * e['w3_scale'].to('cuda').to(torch.bfloat16)[:, None]
w2 = unpack_ternary(e['w2'])[:, :INTER].to(torch.bfloat16) * e['w2_scale'].to('cuda').to(torch.bfloat16)[:, None]
Q_int = e['Q'].to('cuda')          # int8
Q_scale = e['Q_scale'].to('cuda')  # [KP]
Q_bf16 = Q_int.to(torch.bfloat16) * Q_scale.to(torch.bfloat16)[None, :]


def fwd_int8(z):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        Q = Q_int.to(torch.bfloat16) * Q_scale.to(torch.bfloat16)[None, :]
        gate = (z @ w1.T).clamp(max=SWIGLU_LIMIT)
        up = (z @ w3.T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        h = F.silu(gate) * up
        yc = h @ w2.T
        return (yc @ Q.T).float()


def fwd_bf16(z):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        gate = (z @ w1.T).clamp(max=SWIGLU_LIMIT)
        up = (z @ w3.T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        h = F.silu(gate) * up
        yc = h @ w2.T
        return (yc @ Q_bf16.T).float()


for bs in (1, 8, 64, 256):
    z = torch.randn(bs, K, device='cuda', dtype=torch.bfloat16)
    for fn, name in ((fwd_int8, 'int8'), (fwd_bf16, 'bf16')):
        # warmup
        for _ in range(20):
            fn(z)
        torch.cuda.synchronize()
        n = 2000
        t0 = time.time()
        for _ in range(n):
            fn(z)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n * 1e6
        print(f'bs={bs:>4} {name:>5}: {dt:8.1f} us/call', flush=True)
