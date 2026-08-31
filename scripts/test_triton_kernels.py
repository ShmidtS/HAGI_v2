"""Unit test: triton_kernels k_h13/k_y vs the torch decode path on random
data in the EXACT checkpoint format (w1a/w3a u8 [2048,512], scales fp32,
w2a u8 [4096,1024], w2a_scale fp32 [4096,16], biases fp32)."""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/c/HAGI_v2/scripts")
import triton_kernels as tk  # noqa: E402


def soft_lim(x, lim=10.0, knee=0.8):
    th = knee * lim
    tail = lim - th
    ax = x.abs()
    y_abs = torch.where(ax <= th, ax, th + tail * torch.tanh((ax - th) / tail))
    return y_abs * torch.sign(x)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    D, I, GS = 4096, 2048, 128
    w1a = torch.randint(0, 255, (I, D // 8), dtype=torch.uint8, device=dev)
    w3a = torch.randint(0, 255, (I, D // 8), dtype=torch.uint8, device=dev)
    s1 = torch.rand(I, device=dev) * 0.02
    s3 = torch.rand(I, device=dev) * 0.02
    b1 = torch.randn(I, device=dev) * 0.5
    b3 = torch.randn(I, device=dev) * 0.5
    w2a = torch.randint(0, 255, (D, I // 2), dtype=torch.uint8, device=dev)
    s2 = torch.rand(D, I // GS, device=dev) * 0.03
    z = torch.randn(D, device=dev) * 3

    # reference (torch, exact checkpoint decode)
    bits = (w1a.unsqueeze(-1) >> torch.arange(8, device=dev, dtype=torch.uint8)) & 1
    W1 = (bits.reshape(I, D).float() * 2 - 1) * s1[:, None]
    bits3 = (w3a.unsqueeze(-1) >> torch.arange(8, device=dev, dtype=torch.uint8)) & 1
    W3 = (bits3.reshape(I, D).float() * 2 - 1) * s3[:, None]
    t = w2a.to(torch.int16)
    W2 = torch.empty(D, I, device=dev)
    W2[:, 0::2] = (t & 15).float() - 8
    W2[:, 1::2] = (t >> 4).float() - 8
    W2 = W2 * s2.repeat_interleave(GS, dim=1)
    g = soft_lim(z @ W1.T + b1)
    u = soft_lim(z @ W3.T + b3)
    h_ref = F.silu(g) * u
    y_ref = h_ref @ W2.T

    h = torch.empty(I, device=dev)
    y = torch.empty(D, device=dev)
    out = torch.zeros(D, device=dev)
    wscalar = 0.25
    tk.k_h13[(I // 256,)](z, w1a, s1, b1, w3a, s3, b3, h, D=D, I=I, BI=256, BK=256, num_warps=8)
    tk.k_y[(D // 512,)](h, w2a, s2, y, out, wscalar, D=D, I=I, GS=GS, BD=512, BK=128, num_warps=8)
    eh = ((h - h_ref).norm() / h_ref.norm()).item()
    ey = ((y - y_ref).norm() / y_ref.norm()).item()
    eo = ((out - wscalar * y_ref).norm() / (wscalar * y_ref).norm()).item()
    print(f"h rel err: {eh*100:.4f}%   y rel err: {ey*100:.4f}%   fused-acc rel err: {eo*100:.4f}%")

    # timing: 1 expert, per-expert launch pattern (6 experts/layer in prod)
    import time
    for _ in range(10):
        tk.k_h13[(I // 256,)](z, w1a, s1, b1, w3a, s3, b3, h, D=D, I=I, BI=256, BK=256, num_warps=8)
        tk.k_y[(D // 512,)](h, w2a, s2, y, out, wscalar, D=D, I=I, GS=GS, BD=512, BK=128, num_warps=8)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 500
    for _ in range(N):
        tk.k_h13[(I // 256,)](z, w1a, s1, b1, w3a, s3, b3, h, D=D, I=I, BI=256, BK=256, num_warps=8)
        tk.k_y[(D // 512,)](h, w2a, s2, y, out, wscalar, D=D, I=I, GS=GS, BD=512, BK=128, num_warps=8)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N * 1000
    print(f"1-expert k_h13+k_y: {dt:.3f} ms  -> 6 experts/layer ~ {dt*6:.2f} ms -> 43 layers ~ {dt*6*43:.0f} ms/token")


if __name__ == "__main__":
    main()
