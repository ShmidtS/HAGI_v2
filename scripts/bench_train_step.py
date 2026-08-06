"""Wall-time benchmark of the current training path on packed corpus data."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hagi.config import load_config
from hagi.data.dataset import build_dataloader
from hagi.model.model import HAGI
from hagi.train.loop import Trainer, configure_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v41_1b.yaml")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--ce-keep-rate", type=float, default=None)
    parser.add_argument("--sampled-k", type=int, default=None)
    parser.add_argument("--in-batch-fraction", type=float, default=None)
    parser.add_argument("--ce-save-logits", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--loop-depth", type=int, default=None)
    parser.add_argument("--ffn-expansion", type=float, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--full-every", type=int, default=None)
    parser.add_argument("--exact-ce-interval", type=int, default=None)
    parser.add_argument("--exact-ce-rows", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA/HIP device")
    overrides: dict[str, object] = {"train.data.num_workers": 0, "train.checkpoint_interval": 0}
    if args.batch_size is not None:
        overrides["train.batch_size"] = args.batch_size
    if args.grad_accum is not None:
        overrides["train.grad_accum_steps"] = args.grad_accum
    if args.ce_keep_rate is not None:
        overrides["train.ce_keep_rate"] = args.ce_keep_rate
    if args.sampled_k is not None:
        overrides["model.head.sampled_softmax_k"] = args.sampled_k
    if args.in_batch_fraction is not None:
        overrides["model.head.sampled_in_batch_fraction"] = args.in_batch_fraction
    if args.ce_save_logits is not None:
        overrides["model.head.ce_save_logits"] = args.ce_save_logits
    if args.layers is not None:
        overrides["model.num_layers"] = args.layers
    if args.loop_depth is not None:
        overrides["model.loop_depth"] = args.loop_depth
    if args.ffn_expansion is not None:
        overrides["model.ffn.expansion"] = args.ffn_expansion
    if args.window is not None:
        overrides["model.sliding.window"] = args.window
    if args.full_every is not None:
        overrides["model.sliding.full_every"] = args.full_every
    if args.exact_ce_interval is not None:
        overrides["train.logging.exact_ce_interval"] = args.exact_ce_interval
    if args.exact_ce_rows is not None:
        overrides["train.logging.exact_ce_rows"] = args.exact_ce_rows
    if args.warmup_steps is not None:
        overrides["train.schedule.warmup_steps"] = args.warmup_steps
    if args.learning_rate is not None:
        overrides["train.learning_rate"] = args.learning_rate

    configure_runtime()
    torch.manual_seed(args.seed)
    cfg = load_config(args.config, **overrides)
    device = torch.device("cuda")
    model = HAGI(cfg).to(device)
    trainer = Trainer(model, cfg)
    batches = iter(build_dataloader(cfg))
    body_tokens = cfg.train.batch_size * cfg.train.grad_accum_steps * cfg.train.data.seq_len
    timings: list[float] = []

    for step in range(args.warmup + args.steps):
        microbatches = [next(batches) for _ in range(cfg.train.grad_accum_steps)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        metrics = trainer.train_step(microbatches)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= args.warmup:
            timings.append(elapsed)
            exact = f" exact_ce={metrics['exact_ce']:.4f}" if "exact_ce" in metrics else ""
            print(
                f"step={metrics['step']} ms={elapsed * 1000:.2f} "
                f"body_tok_s={body_tokens / elapsed:.0f} scored={metrics['tokens']} "
                f"nce={metrics['ce']:.4f}{exact}"
            )

    median = statistics.median(timings)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"shape: L={cfg.model.num_layers}x{cfg.model.loop_depth} H={cfg.model.hidden_size} "
        f"FFN={cfg.model.ffn.expansion:g} W={cfg.model.sliding.window}/"
        f"{cfg.model.sliding.full_every}"
    )
    print(
        f"objective: keep={cfg.train.ce_keep_rate:g} "
        f"sampled_k={cfg.model.head.sampled_softmax_k or 'exact'} "
        f"save_logits={cfg.model.head.ce_save_logits}"
    )
    print(f"median: {median * 1000:.2f} ms/step | {body_tokens / median:.0f} body tok/s")
    print(f"peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
