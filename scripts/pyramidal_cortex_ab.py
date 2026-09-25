"""Bounded, multi-seed ablation for the directional pyramidal cortex.

The runner is a mechanism experiment, not a 27B quality result. It compares
four lanes with a common starting base and a fixed holdout batch that is not
part of the training stream:

* ``full_base`` — ordinary trainable HAGI base;
* ``lora_only`` — frozen base + existing TTT-LoRA contour;
* ``cortex_only`` — frozen base + directional Pyramidal Cortex;
* ``cortex_lora`` — frozen base + both contours.

The quality verdict is intentionally fail-closed: synthetic holdout runs
only report mechanism signals, while ``quality_supported`` requires a real
packed corpus, at least three seeds, finite measurements, and a strict
majority of wins. Physical INT2/INT4/INT8/FP8 kernels are not implemented
by this runner; its storage section is a hypothetical accounting model, not
a measured deployment format.

Usage:
    python scripts/pyramidal_cortex_ab.py --steps 12 --seed 1234
    python scripts/pyramidal_cortex_ab.py --steps 12 --num-seeds 5 --device cuda
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

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.config import Config, validate_config  # noqa: E402
from hagi.data.dataset import PackedStream, dataset_path  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.loop import Trainer  # noqa: E402

LANE_SPECS = (
    ("full_base", False, False),
    ("lora_only", False, True),
    ("cortex_only", True, False),
    ("cortex_lora", True, True),
)


def make_config(
    *,
    seed: int,
    steps: int,
    rank: int,
    cortex: bool,
    lora: bool,
    real_corpus: bool = False,
) -> Config:
    """Build one structurally complete tiny configuration."""
    cfg = Config()
    m = cfg.model
    m.vocab_size = 32768 if real_corpus else 128
    m.hidden_size = 64
    m.num_layers = 4
    m.attention.num_query_heads = 4
    m.attention.num_kv_heads = 2
    m.attention.head_dim = 16
    m.attention.max_seq_len = 32
    m.sliding.window = 0
    m.ffn.intermediate_size = 128
    m.ffn.multiple_of = 32
    m.head.ce_chunk_rows = 16
    if real_corpus:
        m.head.unigram_prior = False
        m.head.unigram_path = ""

    if cortex:
        m.cortex.enabled = True
        m.cortex.num_levels = 2
        m.cortex.rank = rank
        m.cortex.link_strides = (1,)
    if lora:
        m.adapters.enabled = True
        m.adapters.ttt_lora.enabled = True
        m.adapters.ttt_lora.rank = rank
        m.adapters.ttt_lora.alpha = 1.0

    cfg.train.batch_size = 2
    cfg.train.grad_accum_steps = 1
    cfg.train.data.seq_len = 16
    cfg.train.max_steps = steps
    cfg.train.schedule.warmup_steps = 0
    cfg.train.schedule.decay_fraction = 0.5
    cfg.train.precision = "fp32"
    cfg.train.grad_checkpointing = False
    # Keep the quantization path identical across lanes. The adaptive lanes
    # bypass the cache by contract; disabling it here avoids comparing a
    # cached base step against uncached adaptive steps in the same report.
    cfg.train.ternary_step_cache = False
    cfg.train.learning_rate = 0.05
    cfg.train.adapt.freeze_base = cortex or lora

    m.init_seed = seed
    validate_config(cfg)
    return cfg


def synthetic_batches(
    cfg: Config,
    *,
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Create disjoint fixed train batches and one holdout batch."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train: list[dict[str, torch.Tensor]] = []
    shape = (cfg.train.batch_size, cfg.train.data.seq_len)

    def make_batch() -> dict[str, torch.Tensor]:
        input_ids = torch.randint(
            0,
            cfg.model.vocab_size,
            shape,
            generator=generator,
            dtype=torch.long,
        )
        targets = (input_ids + 1) % cfg.model.vocab_size
        return {
            "input_ids": input_ids.to(device),
            "targets": targets.to(device),
            "loss_mask": torch.ones(shape, dtype=torch.bool, device=device),
        }

    for _ in range(steps):
        train.append(make_batch())
    return train, make_batch()


def real_corpus_batches(
    cfg: Config,
    *,
    source: str,
    data_dir: str,
    steps: int,
    start_offset: int,
    holdout_offset: int,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Read fixed packed windows from one real corpus source.

    Training and holdout offsets must be disjoint and the holdout window must
    start after the final training target token. This does not randomize
    document order; it makes the tiny comparison reproducible and
    memory-bounded.
    """
    if start_offset < 0 or holdout_offset < 0:
        raise ValueError("corpus offsets must be >= 0")
    stream = PackedStream(
        dataset_path(data_dir, source),
        cfg.train.data.seq_len,
        cfg.train.data.eos_token_id,
        np.random.default_rng(0),
        start_offset=start_offset,
    )
    if stream.cursor != start_offset:
        raise ValueError("start_offset is beyond a complete window in the corpus")
    train: list[dict[str, torch.Tensor]] = []
    for _ in range(steps):
        input_rows = []
        target_rows = []
        for _ in range(cfg.train.batch_size):
            window = stream.next_window()
            if window is None:
                raise ValueError("real corpus is shorter than requested training windows")
            input_rows.append(window[:-1])
            target_rows.append(window[1:])
        input_ids = torch.from_numpy(np.stack(input_rows)).to(device)
        targets = torch.from_numpy(np.stack(target_rows)).to(device)
        train.append(
            {
                "input_ids": input_ids,
                "targets": targets,
                "loss_mask": torch.ones(input_ids.shape, dtype=torch.bool, device=device),
            }
        )
    last_end = stream.consumed
    if holdout_offset <= last_end:
        raise ValueError(
            f"holdout offset {holdout_offset} overlaps train tokens ending at {last_end}"
        )
    holdout_stream = PackedStream(
        dataset_path(data_dir, source),
        cfg.train.data.seq_len,
        cfg.train.data.eos_token_id,
        np.random.default_rng(1),
        start_offset=holdout_offset,
    )
    if holdout_stream.cursor != holdout_offset:
        raise ValueError("holdout_offset is beyond a complete window in the corpus")
    holdout_input_rows = []
    holdout_target_rows = []
    for _ in range(cfg.train.batch_size):
        holdout_window = holdout_stream.next_window()
        if holdout_window is None:
            raise ValueError("real corpus is shorter than requested holdout windows")
        holdout_input_rows.append(holdout_window[:-1])
        holdout_target_rows.append(holdout_window[1:])
    holdout_ids = torch.from_numpy(np.stack(holdout_input_rows)).to(device)
    holdout_targets = torch.from_numpy(np.stack(holdout_target_rows)).to(device)
    holdout = {
        "input_ids": holdout_ids,
        "targets": holdout_targets,
        "loss_mask": torch.ones(holdout_ids.shape, dtype=torch.bool, device=device),
    }
    return train, holdout


def model_cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot every state tensor on CPU for deterministic lane setup."""
    return {
        name: tensor.detach().float().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def is_adaptive_key(name: str) -> bool:
    """Whether a state-dict key belongs to an optional adaptive contour."""
    return name.startswith("cortex.") or ".adapters." in name


def lane_state(cfg: Config, base_state: dict[str, torch.Tensor], *, seed: int) -> dict[str, torch.Tensor]:
    """Build lane state from shared base plus deterministic lane-only state."""
    torch.manual_seed(seed)
    lane = HAGI(cfg)
    native = model_cpu_state(lane)
    missing_base = set(base_state) - set(native)
    if missing_base:
        raise ValueError(f"lane is missing common base keys: {sorted(missing_base)[:8]}")
    state = dict(base_state)
    for name, tensor in native.items():
        if is_adaptive_key(name):
            state[name] = tensor
    return state


def load_exact_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load an exactly matched state and fail before partial mutation."""
    current = model.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    if missing or unexpected:
        raise ValueError(
            "lane state mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    mismatched = [
        name
        for name, tensor in current.items()
        if tuple(tensor.shape) != tuple(state[name].shape)
    ]
    if mismatched:
        raise ValueError(f"lane state shape mismatch: {mismatched[:8]}")
    model.load_state_dict(state, strict=True)


def evaluate(model: HAGI, batch: dict[str, torch.Tensor]) -> float:
    """Return exact CE on a fixed batch."""
    model.eval()
    with torch.no_grad():
        output = model(batch["input_ids"], batch["targets"], loss_mask=batch["loss_mask"])
    assert output.ce is not None
    return float(output.ce.detach())


def hypothetical_storage(counts: dict[str, int]) -> dict[str, object]:
    """Report a clearly hypothetical precision-stack accounting model."""
    return {
        "base_body_bits_per_weight": 1.584962500721156,
        "cortex_bits_per_weight": 4.0,
        "online_lora_bits_per_weight": 8.0,
        "activation_and_rls_accumulate": "fp32",
        "parameter_groups": {
            "base_body": int(counts["body"]),
            "cortex": int(counts["cortex"]),
            "online_lora": int(counts["adapter"]),
        },
        "physical_low_bit_kernels_implemented": False,
    }


def quality_supported(
    *,
    execution_completed: bool,
    num_seeds: int,
    cortex_only_deltas: list[float],
    cortex_lora_deltas: list[float],
    real_corpus: bool = False,
) -> bool:
    """Return a conservative real-corpus quality gate.

    Synthetic batches are useful for mechanism execution checks, but cannot
    support a quality verdict. This function therefore fails closed unless
    every configured seed completed on real packed data, at least three seeds
    exist, deltas are finite holdout measurements, and a strict majority of
    seeds beats the matched LoRA reference in both comparisons.
    """
    if (
        not execution_completed
        or not real_corpus
        or num_seeds < 3
        or len(cortex_only_deltas) != num_seeds
        or len(cortex_lora_deltas) != num_seeds
    ):
        return False
    if not all(math.isfinite(delta) for delta in (*cortex_only_deltas, *cortex_lora_deltas)):
        return False
    required_wins = num_seeds // 2 + 1
    return (
        statistics.fmean(cortex_only_deltas) < 0.0
        and statistics.fmean(cortex_lora_deltas) < 0.0
        and sum(delta < 0.0 for delta in cortex_only_deltas) >= required_wins
        and sum(delta < 0.0 for delta in cortex_lora_deltas) >= required_wins
    )


def run_lane(
    name: str,
    *,
    cfg: Config,
    state: dict[str, torch.Tensor],
    train_batches: list[dict[str, torch.Tensor]],
    holdout_batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, object]:
    """Train and evaluate one matched lane on the shared data contract."""
    model = HAGI(cfg).to(device)
    load_exact_state(model, state)
    trainer = Trainer(model, cfg)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer_parameters = sum(
        len(group["params"]) for group in trainer.optimizer.param_groups
    )

    initial_ce = evaluate(model, holdout_batch)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step_times: list[float] = []
    update_flags: list[bool] = []
    for batch in train_batches:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        metrics = trainer.train_step([batch])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - started)
        update_flags.append(bool(metrics.get("update_applied", False)))

    final_ce = evaluate(model, holdout_batch)
    counts = model.param_summary()
    peak_vram_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        "name": name,
        "initial_holdout_ce": initial_ce,
        "final_holdout_ce": final_ce,
        "holdout_ce_delta": final_ce - initial_ce,
        "steps_completed": len(step_times),
        "updates_applied": sum(update_flags),
        "rejected_updates": len(update_flags) - sum(update_flags),
        "mean_step_ms": statistics.fmean(step_times) * 1000.0,
        "peak_vram_bytes": peak_vram_bytes,
        "trainable_parameters": int(trainable),
        "optimizer_parameter_tensors": int(optimizer_parameters),
        "optimizer_groups": {
            "muon": bool(trainer.optimizer.muon is not None),
            "adamw": len(trainer.optimizer.adamw.param_groups),
        },
        "params": counts,
        "hypothetical_storage_model": hypothetical_storage(counts),
    }


def run_seed(
    *,
    seed: int,
    steps: int,
    rank: int,
    device: torch.device,
    real_corpus: bool,
    source: str,
    data_dir: str,
    start_offset: int,
    holdout_offset: int,
) -> dict[str, object]:
    """Run all four lanes for one seed with one fixed holdout batch."""
    # Every seed is independent; lane construction and data generation must
    # not depend on the previous seed's global RNG state.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    base_cfg = make_config(
        seed=seed,
        steps=steps,
        rank=rank,
        cortex=False,
        lora=False,
        real_corpus=real_corpus,
    )
    base_state = model_cpu_state(HAGI(base_cfg))
    if real_corpus:
        train_batches, holdout_batch = real_corpus_batches(
            base_cfg,
            source=source,
            data_dir=data_dir,
            steps=steps,
            start_offset=start_offset + seed,
            holdout_offset=holdout_offset + seed,
            device=device,
        )
    else:
        train_batches, holdout_batch = synthetic_batches(
            base_cfg,
            steps=steps,
            seed=seed + 1,
            device=device,
        )

    lanes: list[dict[str, object]] = []
    for name, cortex, lora in LANE_SPECS:
        cfg = make_config(
            seed=seed,
            steps=steps,
            rank=rank,
            cortex=cortex,
            lora=lora,
            real_corpus=real_corpus,
        )
        state = lane_state(cfg, base_state, seed=seed)
        lanes.append(
            run_lane(
                name,
                cfg=cfg,
                state=state,
                train_batches=train_batches,
                holdout_batch=holdout_batch,
                device=device,
            )
        )

    by_name = {str(lane["name"]): lane for lane in lanes}
    cortex_trainable = float(by_name["cortex_only"]["trainable_parameters"])
    cortex_lora_trainable = float(by_name["cortex_lora"]["trainable_parameters"])
    cortex_vs_lora = float(by_name["cortex_only"]["final_holdout_ce"]) - float(
        by_name["lora_only"]["final_holdout_ce"]
    )
    cortex_lora_vs_lora = float(by_name["cortex_lora"]["final_holdout_ce"]) - float(
        by_name["lora_only"]["final_holdout_ce"]
    )
    comparisons = {
        "cortex_only_minus_lora_only_ce": cortex_vs_lora,
        "cortex_plus_lora_minus_lora_only_ce": cortex_lora_vs_lora,
        "cortex_only_ce_delta_per_trainable_parameter": cortex_vs_lora / cortex_trainable,
        "cortex_plus_lora_ce_delta_per_trainable_parameter": (
            cortex_lora_vs_lora / cortex_lora_trainable
        ),
    }
    execution_completed = all(
        lane["steps_completed"] == steps
        and lane["rejected_updates"] == 0
        and math.isfinite(float(lane["initial_holdout_ce"]))
        and math.isfinite(float(lane["final_holdout_ce"]))
        for lane in lanes
    )
    return {
        "seed": seed,
        "execution_completed": execution_completed,
        "quality_signal": {
            "cortex_only_beats_lora_only": comparisons["cortex_only_minus_lora_only_ce"] < 0.0,
            "cortex_plus_lora_beats_lora_only": comparisons[
                "cortex_plus_lora_minus_lora_only_ce"
            ]
            < 0.0,
        },
        "comparisons": comparisons,
        "trainable_parameters": {
            "full_base": int(by_name["full_base"]["trainable_parameters"]),
            "lora_only": int(by_name["lora_only"]["trainable_parameters"]),
            "cortex_only": int(by_name["cortex_only"]["trainable_parameters"]),
            "cortex_lora": int(by_name["cortex_lora"]["trainable_parameters"]),
        },
        "lanes": lanes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--real-corpus", action="store_true")
    parser.add_argument("--source", default="wikipedia_ru")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--start-offset", type=int, default=100_000)
    parser.add_argument("--holdout-offset", type=int, default=10_000_000)
    parser.add_argument("--output", type=Path, default=Path("reports/pyramidal_cortex_ab.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 1 or args.rank < 1 or args.num_seeds < 1:
        raise ValueError("steps, rank, and num-seeds must be positive")

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed_reports = [
        run_seed(
            seed=args.seed + offset * 1009,
            steps=args.steps,
            rank=args.rank,
            device=device,
            real_corpus=args.real_corpus,
            source=args.source,
            data_dir=args.data_dir,
            start_offset=args.start_offset,
            holdout_offset=args.holdout_offset,
        )
        for offset in range(args.num_seeds)
    ]

    execution_completed = all(bool(report["execution_completed"]) for report in seed_reports)
    cortex_only_deltas = [
        float(report["comparisons"]["cortex_only_minus_lora_only_ce"])
        for report in seed_reports
    ]
    cortex_lora_deltas = [
        float(report["comparisons"]["cortex_plus_lora_minus_lora_only_ce"])
        for report in seed_reports
    ]
    cortex_only_wins = sum(delta < 0.0 for delta in cortex_only_deltas)
    cortex_lora_wins = sum(delta < 0.0 for delta in cortex_lora_deltas)
    quality_gate = quality_supported(
        execution_completed=execution_completed,
        num_seeds=args.num_seeds,
        cortex_only_deltas=cortex_only_deltas,
        cortex_lora_deltas=cortex_lora_deltas,
        real_corpus=args.real_corpus,
    )
    report = {
        "experiment": "pyramidal_cortex_matched_multi_seed_ab",
        "execution_completed": execution_completed,
        "quality_supported": quality_gate,
        "quality_evidence": {
            "minimum_seeds_required": 3,
            "requires_real_corpus": True,
            "synthetic_quality_verdict": "not supported; mechanism signal only",
            "cortex_only_mean_holdout_ce_delta": statistics.fmean(cortex_only_deltas),
            "cortex_lora_mean_holdout_ce_delta": statistics.fmean(cortex_lora_deltas),
            "cortex_only_wins": cortex_only_wins,
            "cortex_lora_wins": cortex_lora_wins,
        },
        "scope": (
            "real packed corpus or synthetic holdout mechanism gate; not a 27B quality result"
            if args.real_corpus
            else "synthetic holdout mechanism gate; not a 27B quality result"
        ),
        "device": str(device),
        "base_seed": args.seed,
        "num_seeds": args.num_seeds,
        "steps": args.steps,
        "rank": args.rank,
        "data": {
            "real_corpus": args.real_corpus,
            "source": args.source,
            "data_dir": args.data_dir,
            "start_offset": args.start_offset,
            "holdout_offset": args.holdout_offset,
        },
        "physical_low_bit_kernels_implemented": False,
        "seeds": seed_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if execution_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
