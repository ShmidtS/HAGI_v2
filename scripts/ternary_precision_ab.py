"""Bounded comparison of HAGI ternary precision paths.

The lanes share the same pre-cast FP32 state source, synthetic train stream,
and holdout batch. The BF16 lanes necessarily round that state during their
configured cast, so their post-cast tensors are not claimed bitwise equal.

* ``fp32`` — full FP32 compute/masters;
* ``bf16`` — legacy BF16 body masters and compute;
* ``bf16_fp32_master`` — BF16 compute with FP32 ``BitLinear`` masters.

This is a mechanism/numerics smoke, not a quality result.  It does not claim
physical ternary, INT4, INT8, or FP8 storage.  The FP32-master lane exists
to test whether small updates survive the BF16 body path; the effective
ternary matmul remains in the activation dtype.

Usage:
    python scripts/ternary_precision_ab.py --steps 4 --seed 1234
    python scripts/ternary_precision_ab.py --steps 4 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.config import Config, validate_config  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.model.ternary import BitLinear  # noqa: E402
from hagi.train.loop import Trainer  # noqa: E402

LANE_SPECS = (
    ("fp32", "fp32", False),
    ("bf16", "bf16", False),
    ("bf16_fp32_master", "bf16", True),
)


def make_config(
    *,
    seed: int,
    steps: int,
    precision: str,
    ternary_fp32_master: bool,
) -> Config:
    """Build one matched tiny HAGI configuration."""
    cfg = Config()
    model = cfg.model
    model.vocab_size = 128
    model.hidden_size = 64
    model.num_layers = 4
    model.attention.num_query_heads = 4
    model.attention.num_kv_heads = 2
    model.attention.head_dim = 16
    model.attention.max_seq_len = 32
    model.sliding.window = 0
    model.ffn.intermediate_size = 128
    model.ffn.multiple_of = 32
    model.head.ce_chunk_rows = 16
    model.init_seed = seed

    train = cfg.train
    train.batch_size = 2
    train.grad_accum_steps = 1
    train.data.seq_len = 16
    train.max_steps = steps
    train.schedule.warmup_steps = 0
    train.schedule.decay_fraction = 0.5
    train.precision = precision
    train.ternary_fp32_master = ternary_fp32_master
    train.ternary_step_cache = True
    train.grad_checkpointing = False
    train.use_muon = False
    train.learning_rate = 0.05
    validate_config(cfg)
    return cfg


def fixed_batches(
    cfg: Config,
    *,
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Create a shared train stream and a disjoint fixed holdout batch."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (cfg.train.batch_size, cfg.train.data.seq_len)

    def make_batch() -> dict[str, torch.Tensor]:
        input_ids = torch.randint(
            0,
            cfg.model.vocab_size,
            shape,
            generator=generator,
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids.to(device),
            "targets": ((input_ids + 1) % cfg.model.vocab_size).to(device),
            "loss_mask": torch.ones(shape, dtype=torch.bool, device=device),
        }

    train = [make_batch() for _ in range(steps)]
    return train, make_batch()


def model_cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot the common state on CPU before any lane-specific casting."""
    return {
        name: tensor.detach().float().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def load_exact_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Reject key/shape drift before copying any tensor into ``model``."""
    current = model.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    if missing or unexpected:
        raise ValueError(f"precision lane state mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
    mismatched = [
        name
        for name, tensor in current.items()
        if tuple(tensor.shape) != tuple(state[name].shape)
    ]
    if mismatched:
        raise ValueError(f"precision lane state shape mismatch: {mismatched[:8]}")
    model.load_state_dict(state, strict=True)


def dtype_histogram(model: torch.nn.Module) -> dict[str, int]:
    """Count parameter elements by final dtype."""
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype).removeprefix("torch.")
        counts[key] = counts.get(key, 0) + parameter.numel()
    return counts


def bitlinear_dtype_histogram(model: torch.nn.Module) -> dict[str, int]:
    """Count ternary-master elements by final dtype."""
    counts: dict[str, int] = {}
    for module in model.modules():
        if isinstance(module, BitLinear):
            key = str(module.weight.dtype).removeprefix("torch.")
            counts[key] = counts.get(key, 0) + module.weight.numel()
    return counts


def parameter_bytes(model: torch.nn.Module) -> int:
    """Return live parameter bytes, distinct from hypothetical packed storage."""
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def tensor_bytes(value: object) -> int:
    """Count tensor payload bytes recursively in optimizer state mappings."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def optimizer_state_bytes(trainer: Trainer) -> int:
    """Return live optimizer tensor bytes after materialized state exists."""
    return tensor_bytes(trainer.optimizer.state_dict())


def evaluate(model: HAGI, batch: dict[str, torch.Tensor]) -> float:
    """Evaluate exact CE on the shared holdout batch."""
    model.eval()
    with torch.no_grad():
        output = model(batch["input_ids"], batch["targets"], loss_mask=batch["loss_mask"])
    assert output.ce is not None
    return float(output.ce.detach())


def run_lane(
    name: str,
    *,
    cfg: Config,
    state: dict[str, torch.Tensor],
    train_batches: list[dict[str, torch.Tensor]],
    holdout_batch: dict[str, torch.Tensor],
    device: torch.device,
    timing_warmup: int = 0,
) -> dict[str, object]:
    """Run one precision lane with matched state, batches, and updates."""
    torch.manual_seed(int(cfg.model.init_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.model.init_seed))
    model = HAGI(cfg).to(device)
    load_exact_state(model, state)
    trainer = Trainer(model, cfg)
    # Snapshot after the lane's precision policy is applied.  Measuring from the
    # pre-cast FP32 state would mix initialization rounding into update size.
    before = {
        module_name: module.weight.detach().float().cpu().clone()
        for module_name, module in model.named_modules()
        if isinstance(module, BitLinear)
    }
    initial_ce = evaluate(model, holdout_batch)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step_times: list[float] = []
    update_flags: list[bool] = []
    executed_steps = 0
    for index, batch in enumerate(train_batches):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        metrics = trainer.train_step([batch])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if index >= timing_warmup:
            step_times.append(time.perf_counter() - started)
        update_flags.append(bool(metrics.get("update_applied", False)))
        executed_steps += 1

    final_ce = evaluate(model, holdout_batch)
    max_update = max(
        (
            float((module.weight.detach().float().cpu() - before[module_name]).abs().max())
            for module_name, module in model.named_modules()
            if isinstance(module, BitLinear)
        ),
        default=0.0,
    )
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        "name": name,
        "precision": cfg.train.precision,
        "ternary_fp32_master": cfg.train.ternary_fp32_master,
        "initial_holdout_ce": initial_ce,
        "final_holdout_ce": final_ce,
        "holdout_ce_delta": final_ce - initial_ce,
        "steps_completed": executed_steps,
        "updates_applied": sum(update_flags),
        "rejected_updates": len(update_flags) - sum(update_flags),
        "timed_steps": len(step_times),
        "mean_timed_step_ms": statistics.fmean(step_times) * 1000.0,
        "peak_vram_bytes": peak_vram,
        "parameter_bytes": parameter_bytes(model),
        "optimizer_state_bytes": optimizer_state_bytes(trainer),
        "parameter_dtype_elements": dtype_histogram(model),
        "bitlinear_master_dtype_elements": bitlinear_dtype_histogram(model),
        "max_ternary_master_update_abs": max_update,
        "params": model.param_summary(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--timing-warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ternary_precision_ab.json"),
    )
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if not 0 <= args.timing_warmup < args.steps:
        raise ValueError("timing-warmup must be in [0, steps)")

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    base_cfg = make_config(seed=args.seed, steps=args.steps, precision="fp32", ternary_fp32_master=False)
    base_state = model_cpu_state(HAGI(base_cfg))
    train_batches, holdout_batch = fixed_batches(
        base_cfg,
        steps=args.steps,
        seed=args.seed + 1,
        device=device,
    )

    lanes: list[dict[str, object]] = []
    for name, precision, fp32_master in LANE_SPECS:
        cfg = make_config(
            seed=args.seed,
            steps=args.steps,
            precision=precision,
            ternary_fp32_master=fp32_master,
        )
        lanes.append(
            run_lane(
                name,
                cfg=cfg,
                state=base_state,
                timing_warmup=args.timing_warmup,
                train_batches=train_batches,
                holdout_batch=holdout_batch,
                device=device,
            )
        )

    by_name = {str(lane["name"]): lane for lane in lanes}
    execution_completed = all(
        lane["steps_completed"] == args.steps
        and lane["rejected_updates"] == 0
        and lane["updates_applied"] == args.steps
        and math.isfinite(float(lane["initial_holdout_ce"]))
        and math.isfinite(float(lane["final_holdout_ce"]))
        for lane in lanes
    )
    report = {
        "experiment": "ternary_precision_matched_ab",
        "execution_completed": execution_completed,
        "quality_supported": False,
        "scope": "synthetic mechanism/numerics smoke; not a quality or physical-storage result",
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "timing_warmup_steps": args.timing_warmup,
        "timing_note": "bounded smoke only; backend warm-up and host effects remain",
        "common_pre_cast_state_source": True,
        "common_post_cast_state_exact": False,
        "common_train_and_holdout_batches": True,
        "physical_low_bit_kernels_implemented": False,
        "comparison": {
            "bf16_minus_fp32_final_ce": float(by_name["bf16"]["final_holdout_ce"])
            - float(by_name["fp32"]["final_holdout_ce"]),
            "fp32_master_minus_fp32_final_ce": float(by_name["bf16_fp32_master"]["final_holdout_ce"])
            - float(by_name["fp32"]["final_holdout_ce"]),
            "fp32_master_minus_bf16_final_ce": float(by_name["bf16_fp32_master"]["final_holdout_ce"])
            - float(by_name["bf16"]["final_holdout_ce"]),
        },
        "lanes": lanes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if execution_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
