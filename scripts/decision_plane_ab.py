"""Bounded synthetic ablation for the opt-in finite-option decision plane.

The synthetic fixture validates mechanics only: data isolation, finite
updates, frozen-encoder learning, end-to-end gradients, decision metrics,
and calibration bookkeeping.  It does not support a model-quality claim;
``quality_supported`` therefore remains false for every synthetic run.
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
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.config import Config, validate_config  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.loop import Trainer  # noqa: E402

LANE_SPECS = ("majority_or_uniform", "frozen_probe", "decision_only", "end_to_end")
FIRST_TOKEN_WEIGHTS = (0.50, 0.28, 0.15, 0.07)
MAX_ECE = 0.15
MAX_ACCURACY_REGRESSION = 0.0


def make_config(*, seed: int, steps: int, num_options: int, freeze_base: bool) -> Config:
    """Build a tiny matched model config for one decision lane."""
    cfg = Config()
    model = cfg.model
    model.vocab_size = 32
    model.hidden_size = 64
    model.num_layers = 4
    model.attention.num_query_heads = 4
    model.attention.num_kv_heads = 2
    model.attention.head_dim = 16
    model.attention.max_seq_len = 32
    model.sliding.window = 0
    model.embedding.conv_kernel = 1
    model.ffn.intermediate_size = 128
    model.ffn.multiple_of = 32
    model.head.ce_chunk_rows = 16
    model.init_seed = seed
    model.decision.enabled = True
    model.decision.num_options = num_options
    model.decision.loss_weight = 1.0
    model.decision.confidence_bins = 10

    train = cfg.train
    train.batch_size = 8
    train.grad_accum_steps = 1
    train.data.seq_len = 12
    train.max_steps = steps
    train.schedule.warmup_steps = 0
    train.schedule.decay_fraction = 0.5
    train.precision = "fp32"
    train.ternary_step_cache = False
    train.grad_checkpointing = False
    train.use_muon = False
    train.learning_rate = 0.05
    train.ce_keep_rate = 1.0
    train.adapt.freeze_base = freeze_base
    validate_config(cfg)
    return cfg


def synthetic_partition(
    *,
    seed: int,
    rows: int,
    vocab_size: int,
    seq_len: int,
    num_options: int,
    domain: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build non-trivial sequence labels from the first token.

    The last token is sampled independently, so the label cannot be copied
    from the final position.  Domain 1 uses a different first-token mixture
    while preserving the label mapping.
    """
    if rows < num_options or seq_len < 3 or num_options < 2:
        raise ValueError("fixture requires rows >= options and seq_len >= 3")
    generator = torch.Generator().manual_seed(seed)
    probabilities = torch.tensor(FIRST_TOKEN_WEIGHTS, dtype=torch.float32)
    if domain == 0:
        probabilities = probabilities / probabilities.sum()
    else:
        probabilities = torch.tensor((0.40, 0.30, 0.20, 0.10))[:num_options]
        probabilities = probabilities / probabilities.sum()
    if probabilities.numel() != num_options:
        probabilities = torch.full((num_options,), 1.0 / num_options)

    first = torch.multinomial(probabilities, rows, replacement=True, generator=generator)
    while len(torch.unique(first)) < num_options:
        first = torch.multinomial(probabilities, rows, replacement=True, generator=generator)
    ids = torch.randint(0, vocab_size, (rows, seq_len), generator=generator)
    ids[:, 0] = first
    # Keep the final token independent and high-entropy.
    ids[:, -1] = torch.randint(1, vocab_size, (rows,), generator=generator)
    labels = first.clone().remainder(num_options)
    return ids, labels


def validate_fixtures(
    train_ids: torch.Tensor,
    train_labels: torch.Tensor,
    holdout_ids: torch.Tensor,
    holdout_labels: torch.Tensor,
    *,
    num_options: int,
) -> int:
    """Fail before training when fixture rows overlap or labels are invalid."""
    for name, ids, labels in (
        ("train", train_ids, train_labels),
        ("holdout", holdout_ids, holdout_labels),
    ):
        if ids.ndim != 2 or labels.ndim != 1 or ids.shape[0] != labels.shape[0]:
            raise ValueError(f"{name} ids/labels shape mismatch")
        if ids.shape[0] == 0:
            raise ValueError(f"{name} fixture is empty")
        if int(labels.min()) < 0 or int(labels.max()) >= num_options:
            raise ValueError(f"{name} contains invalid option labels")
    train_rows = {tuple(row) for row in train_ids.tolist()}
    overlap = sum(tuple(row) in train_rows for row in holdout_ids.tolist())
    if overlap:
        raise ValueError(f"train/holdout overlap contains {overlap} rows")
    return 0


def _common_state(model: HAGI) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().float().cpu().clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("decision_head.")
    }


def load_common_state(model: HAGI, state: dict[str, torch.Tensor]) -> None:
    """Load the shared base while leaving a fresh zero decision head."""
    current = model.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    expected_missing = ["decision_head.weight"]
    if missing != expected_missing or unexpected:
        raise ValueError(f"lane state mismatch: missing={missing}, unexpected={unexpected}")
    mismatched = [
        name
        for name, tensor in current.items()
        if name in state and tuple(tensor.shape) != tuple(state[name].shape)
    ]
    if mismatched:
        raise ValueError(f"lane state shape mismatch: {mismatched}")
    model.load_state_dict(state, strict=False)


def decision_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    confidence_bins: int,
) -> dict[str, object]:
    """Compute multiclass decision metrics from detached probabilities."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("decision metric shapes must be [N,K] and [N]")
    if logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("decision metrics require [N,K] with N >= 1 and K >= 2")
    if confidence_bins < 1:
        raise ValueError("confidence_bins must be >= 1")
    if int(targets.min()) < 0 or int(targets.max()) >= logits.shape[1]:
        raise ValueError("decision targets are outside the option range")
    detached_logits = logits.detach().float()
    detached_targets = targets.detach()
    probs = detached_logits.softmax(dim=-1)
    if not torch.isfinite(probs).all() or bool((probs < 0).any()):
        raise ValueError("decision probabilities must be finite and non-negative")
    if not torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]), atol=1e-5):
        raise ValueError("decision probabilities must sum to one")

    classes = detached_logits.shape[1]
    nll = F.cross_entropy(detached_logits, detached_targets, reduction="mean")
    predictions = probs.argmax(dim=-1)
    accuracy = predictions.eq(detached_targets).float().mean()
    one_hot = F.one_hot(detached_targets, num_classes=classes).float()
    brier = (probs - one_hot).square().sum(dim=-1).mean()

    f1_values: list[float] = []
    for option in range(classes):
        true_positive = ((predictions == option) & (detached_targets == option)).sum().item()
        false_positive = ((predictions == option) & (detached_targets != option)).sum().item()
        false_negative = ((predictions != option) & (detached_targets == option)).sum().item()
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)

    confidence, predictions_for_confidence = probs.max(dim=-1)
    bin_indices = torch.clamp((confidence * confidence_bins).long(), max=confidence_bins - 1)
    bins: list[dict[str, float | int]] = []
    ece = torch.zeros((), dtype=torch.float32)
    total = confidence.numel()
    for index in range(confidence_bins):
        selected = bin_indices == index
        count = int(selected.sum())
        if count == 0:
            bins.append({"index": index, "count": 0, "mean_confidence": None, "accuracy": None})
            continue
        mean_confidence = confidence[selected].mean()
        bin_accuracy = predictions_for_confidence[selected].eq(detached_targets[selected]).float().mean()
        ece = ece + selected.float().mean() * (mean_confidence - bin_accuracy).abs()
        bins.append(
            {
                "index": index,
                "count": count,
                "mean_confidence": float(mean_confidence),
                "accuracy": float(bin_accuracy),
            }
        )

    return {
        "nll": float(nll),
        "accuracy": float(accuracy),
        "macro_f1": float(statistics.fmean(f1_values)),
        "brier": float(brier),
        "ece": float(ece),
        "bins": bins,
        "rows": int(total),
    }


def majority_logits(labels: torch.Tensor, num_options: int, *, rows: int | None = None) -> torch.Tensor:
    """Build deterministic majority predictions for ``rows`` examples."""
    if rows is None:
        rows = int(labels.shape[0])
    if rows < 1:
        raise ValueError("majority baseline requires at least one row")
    counts = torch.bincount(labels, minlength=num_options)
    winner = torch.argmax(counts).item()
    # A large finite margin represents the deterministic histogram-majority
    # prediction without producing infinite NLL in the shared metric code.
    logits = torch.zeros(rows, num_options)
    logits[:, winner] = 1.0e4
    return logits


def _lm_ce(model: HAGI, batch: dict[str, torch.Tensor]) -> float:
    model.eval()
    with torch.no_grad():
        output = model(batch["input_ids"], batch["targets"])
    assert output.ce is not None
    return float(output.ce)


def _evaluate(model: HAGI, batch: dict[str, torch.Tensor], bins: int) -> tuple[dict[str, object], float]:
    model.eval()
    with torch.no_grad():
        output = model(
            batch["input_ids"],
            batch["targets"],
            decision_targets=batch["decision_targets"],
        )
    assert output.decision_logits is not None
    assert output.ce is not None
    return decision_metrics(output.decision_logits, batch["decision_targets"], confidence_bins=bins), float(output.ce)


def _train_lane(
    name: str,
    cfg: Config,
    common: dict[str, torch.Tensor],
    train_batches: list[dict[str, torch.Tensor]],
    holdout: dict[str, torch.Tensor],
    *,
    device: torch.device,
    include_lm: bool,
) -> dict[str, object]:
    torch.manual_seed(int(cfg.model.init_seed))
    model = HAGI(cfg).to(device)
    load_common_state(model, common)
    initial_metrics, initial_lm_ce = _evaluate(model, holdout, cfg.model.decision.confidence_bins)
    assert model.decision_head is not None
    head_before = model.decision_head.weight.detach().float().cpu().clone()
    trainer = Trainer(model, cfg)
    step_times: list[float] = []
    update_flags: list[bool] = []
    for batch in train_batches:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        if include_lm:
            metrics = trainer.train_step([batch])
        else:
            metrics = trainer.train_step(
                [
                    {
                        "input_ids": batch["input_ids"],
                        "decision_targets": batch["decision_targets"],
                    }
                ]
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append((time.perf_counter() - started) * 1000.0)
        update_flags.append(bool(metrics.get("update_applied", False)))
    head_update = float(
        (model.decision_head.weight.detach().float().cpu() - head_before).abs().max()
    )
    final_metrics, final_lm_ce = _evaluate(model, holdout, cfg.model.decision.confidence_bins)
    return {
        "name": name,
        "initial": initial_metrics,
        "final": final_metrics,
        "initial_lm_ce": initial_lm_ce,
        "final_lm_ce": final_lm_ce,
        "steps_completed": len(step_times),
        "updates_applied": sum(update_flags),
        "rejected_updates": len(update_flags) - sum(update_flags),
        "mean_step_ms": statistics.fmean(step_times) if step_times else 0.0,
        "decision_head_max_abs_update": head_update,
        "parameters": model.param_summary()["total"],
    }


def run_seed(
    *,
    seed: int,
    steps: int,
    num_options: int = 4,
    device: str = "cpu",
) -> dict[str, object]:
    """Run the four-lane synthetic mechanism ablation for one seed."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    torch_device = torch.device(device)
    base_cfg = make_config(seed=seed, steps=steps, num_options=num_options, freeze_base=False)
    torch.manual_seed(seed)
    base = HAGI(base_cfg)
    common = _common_state(base)
    train_ids, train_labels = synthetic_partition(
        seed=seed,
        rows=16,
        vocab_size=base_cfg.model.vocab_size,
        seq_len=base_cfg.train.data.seq_len,
        num_options=num_options,
        domain=0,
    )
    holdout_ids, holdout_labels = synthetic_partition(
        seed=seed + 1_000_003,
        rows=12,
        vocab_size=base_cfg.model.vocab_size,
        seq_len=base_cfg.train.data.seq_len,
        num_options=num_options,
        domain=1,
    )
    dropped = validate_fixtures(
        train_ids,
        train_labels,
        holdout_ids,
        holdout_labels,
        num_options=num_options,
    )
    partition_batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, train_ids.shape[0], base_cfg.train.batch_size):
        ids = train_ids[start : start + base_cfg.train.batch_size]
        labels = train_labels[start : start + base_cfg.train.batch_size]
        targets = torch.roll(ids, shifts=-1, dims=1)
        partition_batches.append(
            {
                "input_ids": ids.to(torch_device),
                "targets": targets.to(torch_device),
                "decision_targets": labels.to(torch_device),
            }
        )
    if not partition_batches:
        raise ValueError("training fixture produced no batches")
    train_batches = [partition_batches[index % len(partition_batches)] for index in range(steps)]
    holdout = {
        "input_ids": holdout_ids.to(torch_device),
        "targets": torch.roll(holdout_ids, shifts=-1, dims=1).to(torch_device),
        "decision_targets": holdout_labels.to(torch_device),
    }

    majority = {
        "name": "majority_or_uniform",
        "initial": decision_metrics(
            majority_logits(train_labels, num_options, rows=holdout_labels.shape[0]),
            holdout_labels,
            confidence_bins=base_cfg.model.decision.confidence_bins,
        ),
        "final": None,
        "initial_lm_ce": None,
        "final_lm_ce": None,
        "train_ms": 0.0,
        "steps_completed": 0,
        "updates_applied": 0,
        "rejected_updates": 0,
        "mean_step_ms": 0.0,
        "decision_head_max_abs_update": 0.0,
        "parameters": 0,
    }
    lanes = [majority]
    for name, freeze_base, include_lm in (
        ("frozen_probe", True, False),
        ("decision_only", False, False),
        ("end_to_end", False, True),
    ):
        cfg = make_config(
            seed=seed,
            steps=steps,
            num_options=num_options,
            freeze_base=freeze_base,
        )
        lanes.append(
            _train_lane(
                name,
                cfg,
                common,
                train_batches,
                holdout,
                device=torch_device,
                include_lm=include_lm,
            )
        )

    finite = all(
        math.isfinite(float(lane["initial"]["nll"]))
        and (lane["final"] is None or math.isfinite(float(lane["final"]["nll"])))
        for lane in lanes
    )
    train_lanes = [lane for lane in lanes if lane["name"] != "majority_or_uniform"]
    updates_ok = all(
        int(lane["steps_completed"]) == steps
        and int(lane["updates_applied"]) == steps
        and int(lane["rejected_updates"]) == 0
        and float(lane["decision_head_max_abs_update"]) > 0.0
        for lane in train_lanes
    )
    initial_metrics_agree = all(lane["initial"] == lanes[1]["initial"] for lane in train_lanes)
    initial_lm_agree = all(
        math.isclose(
            float(lane["initial_lm_ce"]),
            float(lanes[1]["initial_lm_ce"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for lane in train_lanes
    )
    mechanism_supported = (
        finite and dropped == 0 and updates_ok and initial_metrics_agree and initial_lm_agree
    )
    by_name = {str(lane["name"]): lane for lane in lanes}
    end_to_end = by_name["end_to_end"]["final"]
    majority = by_name["majority_or_uniform"]["initial"]
    frozen = by_name["frozen_probe"]["final"]
    assert end_to_end is not None and frozen is not None
    reference_names = ("majority_or_uniform", "frozen_probe")
    accuracy_reference_name = min(
        reference_names,
        key=lambda lane: float(
            majority["nll"] if lane == "majority_or_uniform" else frozen["nll"]
        ),
    )
    accuracy_reference = majority if accuracy_reference_name == "majority_or_uniform" else frozen
    quality_checks = {
        "execution_complete": mechanism_supported,
        "beats_majority_nll": float(end_to_end["nll"]) < float(majority["nll"]),
        "beats_frozen_probe_nll": float(end_to_end["nll"]) < float(frozen["nll"]),
        "no_accuracy_regression": float(end_to_end["accuracy"]) + MAX_ACCURACY_REGRESSION
        >= float(accuracy_reference["accuracy"]),
        "ece_within_limit": float(end_to_end["ece"]) <= MAX_ECE,
    }
    return {
        "schema_version": 2,
        "seed": seed,
        "steps": steps,
        "num_options": num_options,
        "device": str(torch_device),
        "lane_specs": list(LANE_SPECS),
        "dropped_rows": dropped,
        "train_rows": int(train_ids.shape[0]),
        "holdout_rows": int(holdout_ids.shape[0]),
        "synthetic": True,
        "mechanism_supported": mechanism_supported,
        "quality_supported": False,
        "observed_quality_gate_passed": all(quality_checks.values()),
        "quality_checks": quality_checks,
        "accuracy_reference_lane": accuracy_reference_name,
        "runtime_checks": {
            "finite_metrics": finite,
            "all_updates_applied": updates_ok,
            "common_initial_decision_metrics_exact": initial_metrics_agree,
            "common_initial_lm_ce_exact": initial_lm_agree,
        },
        "lanes": lanes,
    }


def run(*, seed: int, steps: int, num_options: int = 4, device: str = "cpu") -> dict[str, object]:
    """Compatibility entry point for one bounded synthetic seed."""
    return run_seed(seed=seed, steps=steps, num_options=num_options, device=device)


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    """Aggregate the pre-registered synthetic decision checks across seeds."""
    seed_reports = [
        run_seed(
            seed=args.seed + offset * 1009,
            steps=args.steps,
            num_options=args.num_options,
            device=args.device,
        )
        for offset in range(args.num_seeds)
    ]
    execution_completed = all(bool(report["mechanism_supported"]) for report in seed_reports)
    required_wins = args.num_seeds // 2 + 1
    majority_wins = 0
    frozen_wins = 0
    accuracy_ok = 0
    ece_ok = 0
    for report in seed_reports:
        lanes = {str(lane["name"]): lane for lane in report["lanes"]}
        end = lanes["end_to_end"]["final"]
        majority = lanes["majority_or_uniform"]["initial"]
        frozen = lanes["frozen_probe"]["final"]
        assert end is not None and frozen is not None
        majority_wins += int(float(end["nll"]) < float(majority["nll"]))
        frozen_wins += int(float(end["nll"]) < float(frozen["nll"]))
        accuracy_ok += int(bool(report["quality_checks"]["no_accuracy_regression"]))
        ece_ok += int(bool(report["quality_checks"]["ece_within_limit"]))
    overall_checks = {
        "at_least_three_seeds": args.num_seeds >= 3,
        "all_mechanisms_completed": execution_completed,
        "strict_majority_wins_vs_majority": majority_wins >= required_wins,
        "strict_majority_wins_vs_frozen_probe": frozen_wins >= required_wins,
        "all_accuracy_checks_passed": accuracy_ok == args.num_seeds,
        "all_ece_checks_passed": ece_ok == args.num_seeds,
    }
    observed_gate_passed = all(overall_checks.values())
    return {
        "experiment": "decision_plane_synthetic_matched_multi_seed",
        "schema_version": 2,
        "base_seed": args.seed,
        "num_seeds": args.num_seeds,
        "steps": args.steps,
        "num_options": args.num_options,
        "device": args.device,
        "synthetic": True,
        "execution_completed": execution_completed,
        "mechanism_supported": execution_completed,
        "quality_supported": False,
        "promotion_status": "research-only",
        "observed_quality_gate_passed": observed_gate_passed,
        "pre_registered_gates": {
            "minimum_seeds": 3,
            "required_strict_majority_wins": required_wins,
            "max_ece": MAX_ECE,
            "max_accuracy_regression": MAX_ACCURACY_REGRESSION,
        },
        "quality_evidence": {
            "wins_vs_majority": majority_wins,
            "wins_vs_frozen_probe": frozen_wins,
            "accuracy_checks_passed": accuracy_ok,
            "ece_checks_passed": ece_ok,
        },
        "overall_checks": overall_checks,
        "seeds": seed_reports,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded DecisionPlane runner CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-options", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("reports/decision_plane_ab.json"))
    args = parser.parse_args(argv)
    if args.steps < 1 or args.num_seeds < 1:
        raise ValueError("steps and num-seeds must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["execution_completed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
