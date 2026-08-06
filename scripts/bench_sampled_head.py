"""A/B benchmark for the V39 sampled receiver on the active GPU.

Compares the retired per-row codebook gather with the production shared-bank
GEMM at identical N/H/V/K. Both paths include forward and backward.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=15360)
    parser.add_argument("--hidden", type=int, default=1152)
    parser.add_argument("--vocab", type=int, default=32768)
    parser.add_argument("--negatives", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA/HIP device")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    n, h, v, k = args.rows, args.hidden, args.vocab, args.negatives
    hidden = torch.randn(n, h, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(v, h, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(v, (n,), device=device)

    def clear_grads() -> None:
        hidden.grad = None
        weight.grad = None

    def per_row() -> None:
        clear_grads()
        negatives = torch.randint(v, (n, k), device=device)
        negatives = torch.where(negatives.eq(targets[:, None]), (negatives + 1) % v, negatives)
        candidates = torch.cat([targets[:, None], negatives], dim=1)
        rows = F.embedding(candidates, weight)
        logits = (hidden[:, None].to(rows.dtype) * rows).sum(dim=-1)
        (-F.log_softmax(logits.float(), dim=-1)[:, 0].mean()).backward()

    def shared_bank() -> None:
        clear_grads()
        negatives = torch.randint(v, (k,), device=device)
        target_rows = F.embedding(targets, weight)
        target_logits = (hidden.to(target_rows.dtype) * target_rows).sum(dim=-1, keepdim=True)
        negative_logits = hidden @ weight.index_select(0, negatives).t()
        negative_logits = negative_logits.masked_fill(targets[:, None].eq(negatives), float("-inf"))
        logits = torch.cat([target_logits, negative_logits], dim=1)
        (-F.log_softmax(logits.float(), dim=-1)[:, 0].mean()).backward()

    old_ms = timed(per_row, warmup=2, iterations=args.iterations)
    new_ms = timed(shared_bank, warmup=2, iterations=args.iterations)
    gathered_gib = n * (k + 1) * h * torch.tensor([], dtype=dtype).element_size() / 2**30
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"shape: N={n} H={h} V={v} K={k}")
    print(f"retired per-row gather: {old_ms:.2f} ms fwd+bwd")
    print(f"shared-bank GEMM:       {new_ms:.2f} ms fwd+bwd")
    print(f"speedup:                {old_ms / new_ms:.2f}x")
    print(f"avoided gathered tensor: {gathered_gib:.2f} GiB bf16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
