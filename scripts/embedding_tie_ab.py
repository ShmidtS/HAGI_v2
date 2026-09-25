"""Bounded tied-vs-untied embedding/head ablation.

The two lanes share every base tensor, token stream, optimizer setting,
seed, and update budget. The untied output projection is initialized to an
exact copy of the source codebook, so step-zero hidden states, logits, and
exact CE are identical; the only intervention is whether input and output
embeddings remain one parameter after optimization.

The runner supports fixed packed ``uint32`` streams for RU, EN, math/code,
and instruction slices. These current flat streams are not a validated
versioned acquisition artifact, so even real-stream results remain
mechanism evidence with ``quality_supported=false``.

Usage:
    python scripts/embedding_tie_ab.py --steps 4 --num-seeds 3
    python scripts/embedding_tie_ab.py --steps 4 --real-corpus --data-dir data
"""

from __future__ import annotations

import argparse
import hashlib
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
from hagi.data.dataset import PackedStream, dataset_path, load_mix  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.loop import Trainer  # noqa: E402

LANE_SPECS = (("tied", True), ("untied", False))
PROTECTED_DOMAINS = ("wikipedia_ru", "wikipedia_en", "openwebmath", "python_instruct")
MAX_MEMORY_RATIO = 1.75
MAX_THROUGHPUT_REGRESSION = 0.25
MAX_DOMAIN_CE_REGRESSION = 0.01


def make_config(
    *,
    seed: int,
    steps: int,
    tie_lm_head: bool,
    vocab_size: int = 32768,
) -> Config:
    """Build a structurally complete tiny embedding ablation configuration."""
    cfg = Config()
    model = cfg.model
    model.vocab_size = vocab_size
    model.hidden_size = 64
    model.num_layers = 4
    model.attention.num_query_heads = 4
    model.attention.num_kv_heads = 2
    model.attention.head_dim = 16
    model.attention.max_seq_len = 32
    model.sliding.window = 0
    model.embedding.tie_lm_head = tie_lm_head
    model.embedding.conv_kernel = 1
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
    train.precision = "fp32"
    train.ternary_step_cache = True
    train.grad_checkpointing = False
    train.compile_model = False
    train.use_muon = False
    train.learning_rate = 0.05
    train.ce_keep_rate = 1.0
    validate_config(cfg)
    return cfg


def _batch_fingerprint(batches: list[dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        for key in ("input_ids", "targets", "doc_ids"):
            if key not in batch:
                continue
            tensor = batch[key].detach().cpu().contiguous()
            digest.update(key.encode("ascii"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _domain_fingerprint(batches: dict[str, dict[str, torch.Tensor]]) -> str:
    return _batch_fingerprint([batches[name] for name in sorted(batches)])


def synthetic_batches(
    cfg: Config,
    *,
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    """Create shared fixed train and synthetic holdout batches."""
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
            "doc_ids": torch.zeros(shape, dtype=torch.long, device=device),
        }

    train = [make_batch() for _ in range(steps)]
    return train, {"synthetic": make_batch()}


def _read_stream_batch(
    stream: PackedStream,
    *,
    batch_size: int,
    eos_token_id: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    input_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for _ in range(batch_size):
        window = stream.next_window()
        if window is None:
            return None
        if int(window.min()) < 0 or int(window.max()) >= vocab_size:
            raise ValueError(f"packed stream {stream.path.name} contains ids outside model vocabulary")
        input_rows.append(window[:-1])
        target_rows.append(window[1:])
    input_ids = torch.from_numpy(np.stack(input_rows)).to(device)
    targets = torch.from_numpy(np.stack(target_rows)).to(device)
    is_eos = (input_ids == eos_token_id).to(torch.int32)
    doc_ids = torch.cumsum(is_eos, dim=0) - is_eos
    return {"input_ids": input_ids, "targets": targets, "doc_ids": doc_ids.to(torch.long)}


def real_corpus_batches(
    cfg: Config,
    *,
    data_dir: str,
    steps: int,
    seed: int,
    start_offset: int,
    holdout_offset: int,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]], dict[str, object]]:
    """Read deterministic fixed packed windows with disjoint protected holdouts."""
    if start_offset < 0 or holdout_offset <= start_offset:
        raise ValueError("holdout_offset must be greater than non-negative start_offset")
    mix = load_mix(data_dir)
    names = sorted(mix)
    probabilities = np.asarray([mix[name] for name in names], dtype=np.float64)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    streams = [
        PackedStream(
            dataset_path(data_dir, name),
            cfg.train.data.seq_len,
            cfg.train.data.eos_token_id,
            np.random.default_rng(seed + index),
            start_offset=start_offset,
        )
        for index, name in enumerate(names)
    ]
    for name, stream in zip(names, streams, strict=True):
        if stream.cursor != start_offset:
            raise ValueError(f"start_offset is beyond a complete window for {name}")

    train: list[dict[str, torch.Tensor]] = []
    for _ in range(steps):
        active = [index for index, stream in enumerate(streams) if not stream.exhausted]
        if not active:
            raise ValueError("real corpus is shorter than the requested training budget")
        active_probabilities = probabilities[active]
        active_probabilities = active_probabilities / active_probabilities.sum()
        selected = active[int(rng.choice(len(active), p=active_probabilities))]
        batch = _read_stream_batch(
            streams[selected],
            batch_size=cfg.train.batch_size,
            eos_token_id=cfg.train.data.eos_token_id,
            vocab_size=cfg.model.vocab_size,
            device=device,
        )
        if batch is None:
            raise ValueError(f"source {names[selected]} is shorter than one requested batch")
        train.append(batch)

    for name, stream in zip(names, streams, strict=True):
        if holdout_offset <= stream.cursor:
            raise ValueError(
                f"holdout_offset {holdout_offset} overlaps training end {stream.cursor} for {name}"
            )
    missing_domains = sorted(set(PROTECTED_DOMAINS).difference(names))
    if missing_domains:
        raise ValueError(f"protected packed domains are missing: {missing_domains}")

    holdouts: dict[str, dict[str, torch.Tensor]] = {}
    for name in PROTECTED_DOMAINS:
        stream = PackedStream(
            dataset_path(data_dir, name),
            cfg.train.data.seq_len,
            cfg.train.data.eos_token_id,
            np.random.default_rng(seed + 1),
            start_offset=holdout_offset,
        )
        if stream.cursor != holdout_offset:
            raise ValueError(f"holdout_offset is beyond a complete window for {name}")
        batch = _read_stream_batch(
            stream,
            batch_size=cfg.train.batch_size,
            eos_token_id=cfg.train.data.eos_token_id,
            vocab_size=cfg.model.vocab_size,
            device=device,
        )
        if batch is None:
            raise ValueError(f"source {name} is shorter than one holdout batch")
        holdouts[name] = batch

    metadata = {
        "real_packed_stream": True,
        "manifest_validated": False,
        "mix": {name: float(mix[name]) for name in names},
        "resolved_dataset_files": {name: str(dataset_path(data_dir, name)) for name in names},
        "start_offset": start_offset,
        "holdout_offset": holdout_offset,
        "train_end_by_source": {
            name: int(stream.cursor) for name, stream in zip(names, streams, strict=True)
        },
    }
    return train, holdouts, metadata


def model_cpu_state(model: HAGI) -> dict[str, torch.Tensor]:
    """Snapshot every state tensor on CPU before lane-specific projection setup."""
    return {
        name: tensor.detach().float().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def lane_state(
    cfg: Config,
    base_state: dict[str, torch.Tensor],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build exact lane state with an untied projection copied from the codebook."""
    torch.manual_seed(seed)
    native = model_cpu_state(HAGI(cfg))
    missing_base = sorted(set(base_state) - set(native))
    lane_only = sorted(set(native) - set(base_state))
    if missing_base:
        raise ValueError(f"lane is missing common base keys: {missing_base[:8]}")
    expected_lane_only = [] if cfg.model.embedding.tie_lm_head else ["head.projection.weight"]
    if lane_only != expected_lane_only:
        raise ValueError(f"lane-only state mismatch: expected={expected_lane_only}, actual={lane_only}")
    state = dict(base_state)
    if not cfg.model.embedding.tie_lm_head:
        codebook = state.get("encoder.embedding.weight")
        if codebook is None:
            raise ValueError("common state is missing encoder.embedding.weight")
        state["head.projection.weight"] = codebook.clone()
    return state


def load_exact_state(model: HAGI, state: dict[str, torch.Tensor]) -> None:
    """Load an exactly matched state and fail before partial mutation."""
    current = model.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    if missing or unexpected:
        raise ValueError(
            f"embedding lane state mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    mismatched = [
        name
        for name, tensor in current.items()
        if tuple(tensor.shape) != tuple(state[name].shape)
    ]
    if mismatched:
        raise ValueError(f"embedding lane state shape mismatch: {mismatched[:8]}")
    model.load_state_dict(state, strict=True)


def _tensor_digest(tensor: torch.Tensor) -> str:
    payload = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def evaluate(model: HAGI, batch: dict[str, torch.Tensor], *, trace: bool = False) -> dict[str, object]:
    """Evaluate exact native-token CE on one fixed packed batch."""
    model.eval()
    with torch.no_grad():
        output = model(
            batch["input_ids"],
            batch["targets"],
            doc_ids=batch.get("doc_ids"),
            loss_mask=torch.ones_like(batch["targets"], dtype=torch.bool),
            return_logits=trace,
        )
    assert output.ce is not None
    result: dict[str, object] = {"ce": float(output.ce)}
    if trace:
        assert output.hidden is not None and output.logits is not None
        result["hidden_sha256"] = _tensor_digest(output.hidden)
        result["logits_sha256"] = _tensor_digest(output.logits)
    return result


def unigram_ce(batch: dict[str, torch.Tensor], vocab_size: int) -> float:
    """Compute a smoothed training-free unigram reference on the same holdout."""
    targets = batch["targets"].reshape(-1).detach().cpu().to(torch.int64)
    counts = torch.bincount(targets, minlength=vocab_size).to(torch.float64)
    probabilities = (counts + 1.0) / (float(targets.numel()) + vocab_size)
    return float(-probabilities[targets].log().mean())


def parameter_bytes(model: HAGI) -> int:
    """Return unique live parameter bytes; tied weights are counted once."""
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
    """Return materialized optimizer tensor bytes after training."""
    return tensor_bytes(trainer.optimizer.state_dict())


def run_lane(
    name: str,
    *,
    cfg: Config,
    state: dict[str, torch.Tensor],
    train_batches: list[dict[str, torch.Tensor]],
    holdouts: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    timing_warmup: int,
) -> dict[str, object]:
    """Train and evaluate one tied or untied lane on the shared data contract."""
    if not 0 <= timing_warmup < len(train_batches):
        raise ValueError("timing_warmup must be in [0, number of steps)")
    torch.manual_seed(int(cfg.model.init_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.model.init_seed))
    model = HAGI(cfg).to(device)
    load_exact_state(model, state)
    trainer = Trainer(model, cfg)
    trace_name = sorted(holdouts)[0]
    initial = {
        domain: float(evaluate(model, batch)["ce"])
        for domain, batch in sorted(holdouts.items())
    }
    initial_trace = evaluate(model, holdouts[trace_name], trace=True)
    projection_exact = bool(
        cfg.model.embedding.tie_lm_head
        or torch.equal(model.head.projection.weight, model.encoder.embedding.weight)
    )
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

    final = {
        domain: float(evaluate(model, batch)["ce"])
        for domain, batch in sorted(holdouts.items())
    }
    timed = step_times[timing_warmup:]
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    params_bytes = parameter_bytes(model)
    opt_bytes = optimizer_state_bytes(trainer)
    return {
        "name": name,
        "tie_lm_head": cfg.model.embedding.tie_lm_head,
        "head_is_tied": model.head.is_tied,
        "initial_projection_exact_copy": projection_exact,
        "initial_domain_ce": initial,
        "initial_mean_ce": statistics.fmean(initial.values()),
        "initial_trace_domain": trace_name,
        "initial_trace": initial_trace,
        "final_domain_ce": final,
        "final_mean_ce": statistics.fmean(final.values()),
        "steps_completed": len(step_times),
        "updates_applied": sum(update_flags),
        "rejected_updates": len(update_flags) - sum(update_flags),
        "timing_warmup_steps": timing_warmup,
        "mean_timed_step_ms": statistics.fmean(timed) * 1000.0 if timed else 0.0,
        "parameter_bytes": params_bytes,
        "optimizer_state_bytes": opt_bytes,
        "training_state_bytes": params_bytes + opt_bytes,
        "peak_vram_bytes": peak_vram,
        "params": model.param_summary(),
    }


def run_seed(
    *,
    seed: int,
    steps: int,
    device: torch.device,
    timing_warmup: int,
    real_corpus: bool,
    data_dir: str,
    start_offset: int,
    holdout_offset: int,
) -> dict[str, object]:
    """Run both matched lanes for one seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    base_cfg = make_config(seed=seed, steps=steps, tie_lm_head=True)
    base_state = model_cpu_state(HAGI(base_cfg))
    if real_corpus:
        train_batches, holdouts, data_metadata = real_corpus_batches(
            base_cfg,
            data_dir=data_dir,
            steps=steps,
            seed=seed + 1,
            start_offset=start_offset,
            holdout_offset=holdout_offset,
            device=device,
        )
    else:
        train_batches, holdouts = synthetic_batches(
            base_cfg,
            steps=steps,
            seed=seed + 1,
            device=device,
        )
        data_metadata = {
            "real_packed_stream": False,
            "manifest_validated": False,
        }

    lanes: list[dict[str, object]] = []
    for name, tie_lm_head in LANE_SPECS:
        cfg = make_config(seed=seed, steps=steps, tie_lm_head=tie_lm_head)
        state = lane_state(cfg, base_state, seed=seed)
        lanes.append(
            run_lane(
                name,
                cfg=cfg,
                state=state,
                train_batches=train_batches,
                holdouts=holdouts,
                device=device,
                timing_warmup=timing_warmup,
            )
        )
    by_name = {str(lane["name"]): lane for lane in lanes}
    tied = by_name["tied"]
    untied = by_name["untied"]
    ce_delta = float(untied["final_mean_ce"]) - float(tied["final_mean_ce"])
    memory_ratio = float(untied["training_state_bytes"]) / float(tied["training_state_bytes"])
    throughput_regression = (
        float(untied["mean_timed_step_ms"]) / max(float(tied["mean_timed_step_ms"]), 1e-12) - 1.0
    )
    domain_deltas = {
        domain: float(untied["final_domain_ce"][domain]) - float(tied["final_domain_ce"][domain])
        for domain in sorted(holdouts)
    }
    initial_traces_exact = (
        tied["initial_trace"]["hidden_sha256"] == untied["initial_trace"]["hidden_sha256"]
        and tied["initial_trace"]["logits_sha256"] == untied["initial_trace"]["logits_sha256"]
    )
    execution_completed = all(
        lane["steps_completed"] == steps
        and lane["updates_applied"] == steps
        and lane["rejected_updates"] == 0
        and math.isfinite(float(lane["initial_mean_ce"]))
        and math.isfinite(float(lane["final_mean_ce"]))
        for lane in lanes
    )
    per_seed_gates = {
        "execution_complete": execution_completed,
        "initial_traces_exact": initial_traces_exact,
        "ce_win": ce_delta < 0.0,
        "memory_ratio_within_limit": memory_ratio <= MAX_MEMORY_RATIO,
        "throughput_regression_within_limit": throughput_regression <= MAX_THROUGHPUT_REGRESSION,
        "protected_domains_within_tolerance": all(
            delta <= MAX_DOMAIN_CE_REGRESSION for delta in domain_deltas.values()
        ),
    }
    return {
        "seed": seed,
        "execution_completed": execution_completed,
        "common_train_fingerprint": _batch_fingerprint(train_batches),
        "common_holdout_fingerprint": _domain_fingerprint(holdouts),
        "unigram_baseline_ce": {
            domain: unigram_ce(batch, base_cfg.model.vocab_size)
            for domain, batch in sorted(holdouts.items())
        },
        "comparisons": {
            "untied_minus_tied_mean_ce": ce_delta,
            "untied_over_tied_training_state_bytes": memory_ratio,
            "untied_step_time_regression_fraction": throughput_regression,
            "untied_minus_tied_domain_ce": domain_deltas,
        },
        "per_seed_gates": per_seed_gates,
        "all_per_seed_gates_passed": all(per_seed_gates.values()),
        "data": data_metadata,
        "lanes": lanes,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    """Run the pre-registered multi-seed tied/untied comparison."""
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed_reports = [
        run_seed(
            seed=args.seed + offset * 1009,
            steps=args.steps,
            device=device,
            timing_warmup=args.timing_warmup,
            real_corpus=args.real_corpus,
            data_dir=args.data_dir,
            start_offset=args.start_offset,
            holdout_offset=args.holdout_offset,
        )
        for offset in range(args.num_seeds)
    ]
    execution_completed = all(bool(report["execution_completed"]) for report in seed_reports)
    ce_deltas = [
        float(report["comparisons"]["untied_minus_tied_mean_ce"]) for report in seed_reports
    ]
    required_wins = args.num_seeds // 2 + 1
    ce_wins = sum(delta < 0.0 for delta in ce_deltas)
    overall_gate = {
        "at_least_three_seeds": args.num_seeds >= 3,
        "strict_majority_ce_wins": ce_wins >= required_wins,
        "lower_mean_final_ce": statistics.fmean(ce_deltas) < 0.0,
        "all_per_seed_gates_passed": all(
            bool(report["all_per_seed_gates_passed"]) for report in seed_reports
        ),
        "execution_completed": execution_completed,
    }
    observed_gate_passed = all(overall_gate.values())
    return {
        "experiment": "embedding_tie_head_matched_ab",
        "schema_version": 1,
        "device": str(device),
        "base_seed": args.seed,
        "num_seeds": args.num_seeds,
        "steps": args.steps,
        "timing_warmup_steps": args.timing_warmup,
        "lane_specs": [name for name, _ in LANE_SPECS],
        "real_packed_stream": args.real_corpus,
        "manifest_validated_corpus": False,
        "execution_completed": execution_completed,
        "quality_supported": False,
        "promotion_status": "research-only",
        "scope_reason": (
            "the current flat data/*.compact.bin streams are not a validated versioned "
            "acquisition artifact; even real-stream results are mechanism evidence"
        ),
        "metric_scope": "exact native-token CE; not a common-reference decoded-text CE",
        "pre_registered_gates": {
            "max_training_state_memory_ratio": MAX_MEMORY_RATIO,
            "max_step_time_regression_fraction": MAX_THROUGHPUT_REGRESSION,
            "max_domain_ce_regression_nats": MAX_DOMAIN_CE_REGRESSION,
            "required_ce_wins": required_wins,
        },
        "observed_gate_passed": observed_gate_passed,
        "promotion_eligible": False,
        "ce_wins": ce_wins,
        "mean_untied_minus_tied_ce": statistics.fmean(ce_deltas),
        "overall_gate_checks": overall_gate,
        "data_contract": {
            "data_dir": args.data_dir,
            "start_offset": args.start_offset,
            "holdout_offset": args.holdout_offset,
        },
        "seeds": seed_reports,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--timing-warmup", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--real-corpus", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--start-offset", type=int, default=100_000)
    parser.add_argument("--holdout-offset", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=Path("reports/embedding_tie_ab.json"))
    args = parser.parse_args(argv)
    if args.steps < 1 or args.num_seeds < 1:
        raise ValueError("steps and num-seeds must be positive")
    if not 0 <= args.timing_warmup < args.steps:
        raise ValueError("timing-warmup must be in [0, steps)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if bool(report["execution_completed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
