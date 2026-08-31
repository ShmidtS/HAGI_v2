"""Unit test: route-indirect bank kernels (k_h13b/k_yb) vs torch reference."""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/c/HAGI_v2/scripts")
import triton_bank_kernels as tbk  # noqa: E402

D, I, GS = 4096, 2048, 128


def soft_lim(x, lim=10.0, knee=0.8):
    th = knee * lim
    tail = lim - th
    ax = x.abs()
    y_abs = torch.where(ax <= th, ax, th + tail * torch.tanh((ax - th) / tail))
    return y_abs * torch.sign(x)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    E = 256  # full bank dim
    w13a = torch.randint(0, 255, (E, 2 * I, D // 8), dtype=torch.uint8, device=dev)
    s13 = torch.rand(E, 2 * I, device=dev) * 0.02
    b13 = torch.randn(E, 2 * I, device=dev) * 0.5
    w2a = torch.randint(0, 255, (E, D, I // 2), dtype=torch.uint8, device=dev)
    s2 = (torch.rand(E, D, I // GS, device=dev) * 0.03).half()

    T = 6
    ids = torch.tensor([7, 255, 0, 128, 42, 191], dtype=torch.int32, device=dev)
    wts = torch.rand(T, device=dev)
    z = torch.randn(D, device=dev) * 3

    h = torch.empty(T, I, device=dev)
    y = torch.empty(T, D, device=dev)
    out = torch.zeros(D, device=dev)
    tbk.k_h13b[(T, I // 256)](z, ids, w13a, s13, b13, h, D=D, I=I, BI=256, BK=256, num_warps=8)
    tbk.k_yb[(T, D // 512)](h, ids, wts, w2a, s2, y, out, D=D, I=I, GS=GS, BD=512, BK=128, num_warps=8)

    # reference: per selected expert, torch unpack of the bank slice
    out_ref = torch.zeros(D, device=dev)
    for t in range(T):
        e = int(ids[t])
        bits = (w13a[e, :I].unsqueeze(-1) >> torch.arange(8, device=dev, dtype=torch.uint8)) & 1
        W1 = (bits.reshape(I, D).float() * 2 - 1) * s13[e, :I][:, None]
        bits3 = (w13a[e, I:].unsqueeze(-1) >> torch.arange(8, device=dev, dtype=torch.uint8)) & 1
        W3 = (bits3.reshape(I, D).float() * 2 - 1) * s13[e, I:][:, None]
        tt = w2a[e].to(torch.int16)
        W2 = torch.empty(D, I, device=dev)
        W2[:, 0::2] = (tt & 15).float() - 8
        W2[:, 1::2] = (tt >> 4).float() - 8
        W2 = W2 * s2[e].float().repeat_interleave(GS, dim=1)
        g = soft_lim(z @ W1.T + b13[e, :I])
        u = soft_lim(z @ W3.T + b13[e, I:])
        h_ref = F.silu(g) * u
        out_ref += wts[t] * (h_ref @ W2.T)

    eo = ((out - out_ref).norm() / out_ref.norm()).item()
    print(f"bank kernels fused out rel err: {eo*100:.4f}%")

    import time
    for _ in range(10):
        tbk.k_h13b[(T, I // 256)](z, ids, w13a, s13, b13, h, D=D, I=I, BI=256, BK=256, num_warps=8)
        tbk.k_yb[(T, D // 512)](h, ids, wts, w2a, s2, y, out, D=D, I=I, GS=GS, BD=512, BK=128, num_warps=8)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 300
    for _ in range(N):
        tbk.k_h13b[(T, I // 256)](z, ids, w13a, s13, b13, h, D=D, I=I, BI=256, BK=256, num_warps=8)
        tbk.k_yb[(T, D // 512)](h, ids, wts, w2a, s2, y, out, D=D, I=I, GS=GS, BD=512, BK=128, num_warps=8)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N * 1000
    print(f"6-expert bank layer decode (2 launches): {dt:.3f} ms -> 43 layers ~ {dt*43:.0f} ms/token")


if __name__ == "__main__":
    main()
