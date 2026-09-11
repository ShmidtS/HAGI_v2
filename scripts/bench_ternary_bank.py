"""Bench: ternary route-indirect bank kernel (k_h13t + k_yb) vs the current
per-expert bf16 unpack path, on one layer with T=6 (top-6), release layout.
Run: python scripts/bench_ternary_bank.py [layer]
"""
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
import triton_bank_kernels as tbk      # k_yb
from ternary_bank_kernels import k_h13t  # ternary W13

from safetensors.torch import load_file

D, I, GS = 4096, 2048, 128


def soft_lim_t(x, lim=10.0, knee=0.75):
    th = knee * lim
    tail = lim - th
    ax = x.abs()
    return torch.where(ax <= th, ax, th + tail * torch.tanh((ax - th) / tail)) * torch.sign(x)


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t = load_file(f"dsv4_release/layers/layer{layer}.safetensors", device="cuda")
    w13t, s13, b13 = t["w13t"], t["s13"], t["b13"]
    w2a, s2 = t["w2a"], t["s2"]
    # w2 scale is f32 in release; k_yb expects f16
    s2h = s2.half()

    T = 6
    torch.manual_seed(0)
    ids = torch.randperm(256, device="cuda")[:T].to(torch.int32)
    wts = torch.rand(T, device="cuda")
    z = (torch.randn(D, device="cuda") * 3.0).contiguous()

    # ---- reference: current decode path (per-expert bf16 unpack) ----
    def ref(zrow):
        out = torch.zeros(D, device="cuda")
        for tt in range(T):
            e = int(ids[tt])
            lut = torch.tensor([[((v // 3 ** j) % 3) - 1 for j in range(5)]
                                for v in range(256)], dtype=torch.int8, device="cuda")
            t1 = lut[w13t[e, :I].long()].float().reshape(I, -1)[:, :D]
            W1 = (t1 * s13[e, :I].repeat_interleave(128, dim=1).float()).to(torch.bfloat16)
            t3 = lut[w13t[e, I:].long()].float().reshape(I, -1)[:, :D]
            W3 = (t3 * s13[e, I:].repeat_interleave(128, dim=1).float()).to(torch.bfloat16)
            g = soft_lim_t(zrow @ W1.T.float() + b13[e, :I].float())
            u = soft_lim_t(zrow @ W3.T.float() + b13[e, I:].float())
            h = (F.silu(g) * u).to(torch.bfloat16)
            ti = w2a[e].to(torch.int16)
            W2 = torch.empty(D, I, device="cuda", dtype=torch.bfloat16)
            W2[:, 0::2] = ((ti & 15) - 8).to(torch.bfloat16)
            W2[:, 1::2] = ((ti >> 4) - 8).to(torch.bfloat16)
            W2 = W2 * s2h[e].repeat_interleave(GS, dim=1)  # f16 x bf16 -> f16? cast
            W2 = W2.float()
            out += wts[tt] * (h.float() @ W2.T)
        return out

    out_ref = ref(z)

    # ---- triton path ----
    h = torch.empty(T, I, device="cuda")
    y = torch.empty(T, D, device="cuda")
    out_tri = torch.zeros(D, device="cuda")

    def tri():
        out_tri.zero_()
        k_h13t[(T, I // 256)](z, ids, w13t, s13, b13, h,
                              D=D, I=I, GS=GS, BI=256, BK=128, num_warps=8)
        tbk.k_yb[(T, D // 512)](h, ids, wts, w2a, s2h, y, out_tri,
                                D=D, I=I, GS=GS, BD=512, BK=128, num_warps=8)
        return out_tri

    out_tri = tri()
    torch.cuda.synchronize()
    err = ((out_tri - out_ref).norm() / out_ref.norm()).item()
    print(f"rel err triton vs ref: {err:.4%}")

    # ---- timings ----
    for name, fn in (("ref (per-expert bf16 unpack)", ref),
                     ("triton (packed, route-indirect)", lambda _=None: tri())):
        # warmup
        for _ in range(3):
            fn(z) if name.startswith("ref") else fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N = 20
        for _ in range(N):
            fn(z) if name.startswith("ref") else fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N
        # per-layer x43 projection
        print(f"{name}: {dt*1000:.2f} ms/layer-token -> {dt*43*1000:.0f} ms/token "
              f"(~{1.0/(dt*43):.2f} tok/s MoE-bound)")


if __name__ == "__main__":
    main()
