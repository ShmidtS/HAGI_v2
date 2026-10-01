"""Measure the AM-GM constants (t0, c) for the current training config.

``Hagi/Audit/Exactness.lean`` makes the step time

    T(B) = (1 + B_n/B) * (t0 + c*B)

so the optimal batch needs only two measured numbers: the fixed
per-micro-step overhead ``t0`` and the marginal per-sample cost ``c``.
This script times forward+backward at several batch sizes, fits
``T(B) = t0 + c*B`` by least squares, and reports both constants plus
the derived ``B* = sqrt(B_n*t0/c)`` and the resulting speedup.

Run: python scripts/measure_batch_law.py --config configs/ab_base.yaml
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.config import load_config  # noqa: E402
from hagi.model.factory import build_model_for_config  # noqa: E402
from hagi.train.batch_law import optimal_batch, step_time  # noqa: E402
from hagi.train.loop import cast_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ab_base.yaml")
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--token-budget", type=int, default=0,
                        help="tokens per optimizer step (0 = batch*seq_len at the first batch)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_for_config(cfg)
    # The Trainer casts to the configured precision before the first step.
    # Without this the probe times an fp32 eager model -- measured 22 s/step
    # against the Trainer's 1.35 s -- and the resulting t0 would then justify
    # a batch change for a model that never runs that way.
    cast_model(
        model,
        cfg.train.precision,
        ternary_fp32_master=bool(cfg.train.ternary_fp32_master),
    )
    model = model.to(device)
    model.train()

    vocab = int(cfg.model.vocab_size)
    seq_len = args.seq_len

    def one_step(bs: int) -> float:
        ids = torch.randint(0, vocab, (bs, seq_len), device=device)
        tgt = torch.randint(0, vocab, (bs, seq_len), device=device)
        model.zero_grad(set_to_none=True)
        out = model(ids, tgt)
        loss = out.lm_loss if out.lm_loss is not None else out.ce
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return float(loss.detach())

    timings: list[tuple[int, float]] = []
    for bs in args.batches:
        for _ in range(args.warmup):
            one_step(bs)
        samples = []
        for _ in range(args.iters):
            start = time.perf_counter()
            one_step(bs)
            samples.append(time.perf_counter() - start)
        median = statistics.median(samples)
        timings.append((bs, median))
        print(f"B={bs:>4}  {median * 1000:8.2f} ms/step  {median / bs * 1000:7.3f} ms/sample")

    # Least-squares fit of T(B) = t0 + c*B.
    n = len(timings)
    sum_b = sum(b for b, _ in timings)
    sum_t = sum(t for _, t in timings)
    sum_bb = sum(b * b for b, _ in timings)
    sum_bt = sum(b * t for b, t in timings)
    denom = n * sum_bb - sum_b * sum_b
    c = (n * sum_bt - sum_b * sum_t) / denom
    overhead = (sum_t - c * sum_b) / n

    token_budget = args.token_budget or max(b for b, _ in timings) * seq_len
    best = optimal_batch(token_budget, overhead, c) if overhead > 0 else float("nan")

    print()
    print(f"fitted t0 (fixed overhead) = {overhead * 1000:.3f} ms")
    print(f"fitted c  (per sample)     = {c * 1000:.4f} ms")
    print(f"token budget B_n           = {token_budget}")
    print(f"analytic B*                = {best:.1f}")
    print()
    print("model vs measurement:")
    for bs, measured in timings:
        print(
            f"  B={bs:>4}  measured {measured * 1000:8.2f} ms"
            f"   model {step_time(bs, token_budget, overhead, c) * 1000:8.2f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
