"""Profile the LM head in isolation (reviewer prescription, round 39).

Measures, at bf16 / N=32768 rows / H=128 / V=32768 on this GPU:
  1. bare forward GEMM  [N,H] @ [H,V]  (torch.matmul)
  2. bare backward dW   [H,N] @ [N,V]
  3. the fused chunked-CE path (head.loss fwd+bwd) at real shapes
  4. a full optimizer step over head-sized params (AdamW dense)
  5. peak-FLOP share: GEMM FLOPs / measured ms vs card peak

Verdict rule (reviewer): if bare GEMM ~= fused-CE time -> GEMM-bound
(only N*V*H reduction helps); if GEMM << fused-CE -> softmax/CE kernels
are the fusion target.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hagi.config import HeadConfig  # noqa: E402
from hagi.model.head import LMHead  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402

N, H, V = 32768, 128, 32768


def bench(fn, warmup=3, iters=20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> int:
    configure_runtime()
    device = "cuda"
    torch.manual_seed(0)
    h = torch.randn(N, H, device=device, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(V, H, device=device, dtype=torch.bfloat16, requires_grad=True)
    t = torch.randint(0, V, (N,), device=device)

    # 1. bare forward GEMM
    fwd = bench(lambda: F.linear(h, w))
    # 2. backward GEMMs (dX and dW) given a dense upstream grad
    z = F.linear(h, w)
    gz = torch.randn_like(z)

    def bwd():
        z_ = F.linear(h, w)
        z_.backward(gz, retain_graph=True)

    bw = bench(bwd)

    # 3. the fused chunked-CE (fwd+bwd) through the real head
    head = LMHead(H, V, HeadConfig()).to(device)
    head = head.to(torch.bfloat16)

    def fused():
        h_ = h.detach().requires_grad_(True)
        loss, _z = head.loss(h_, t)
        loss.backward()

    fu = bench(fused)

    # 4. dense AdamW step over a head-sized param set
    opt = torch.optim.AdamW([w], lr=1e-3)
    gw = torch.randn_like(w)

    def ostep():
        w.grad = gw
        opt.step()

    os_ms = bench(ostep)

    # 5. FLOP shares
    gemm_flop = 2 * N * H * V
    fwd_tflops = gemm_flop / (fwd / 1000) / 1e12
    # both GEMM directions per step (fwd + dW) x2 for multiply-add already counted
    bwd_tflops = gemm_flop / (bw / 1000) / 1e12
    props = torch.cuda.get_device_properties(0)
    peak = getattr(props, "multi_processor_count", None)
    print(f"device: {props.name} | SMs: {peak}")
    print(f"shapes: N={N} H={H} V={V} bf16")
    print(f"1. forward GEMM:        {fwd:8.2f} ms  ({fwd_tflops:.2f} TFLOP/s)")
    print(f"2. backward GEMMs:      {bw:8.2f} ms  ({bwd_tflops:.2f} TFLOP/s)")
    print(f"3. fused-CE fwd+bwd:    {fu:8.2f} ms")
    print(f"4. AdamW step (V*H):    {os_ms:8.2f} ms")
    ratio = fu / max(fwd + bw, 1e-9)
    print(f"\nfused-CE / (fwd+bwd GEMMs) = {ratio:.2f}x")
    if ratio < 1.3:
        print("VERDICT: GEMM-bound -- only N*V*H reduction helps (NCE/V-slice).")
    else:
        print("VERDICT: softmax/CE kernels dominate -- fusion target (epilogue).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
