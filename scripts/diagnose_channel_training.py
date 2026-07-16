"""Finite-training diagnostics for channel-correct training runs."""

from __future__ import annotations

import argparse
import json
import math

import torch


def collect_diagnostics(metrics: list[dict]) -> dict:
    keys = (
        "loss",
        "objective_loss",
        "masked_ce",
        "bpt",
        "top2_mass",
        "posterior_entropy",
        "semantic_mask_ratio",
        "physical_corruption_ratio",
        "suffix_task_ratio",
        "random_task_ratio",
        "grad_norm",
        "grad_rms",
        "correction_alignment",
    )

    def metric_is_finite(metric: dict) -> bool:
        if not all(math.isfinite(float(metric[key])) for key in keys):
            return False
        suffix_ce = float(metric["suffix_ce"])
        return math.isfinite(suffix_ce) or float(metric["suffix_task_ratio"]) == 0.0

    def endpoint(key: str, index: int):
        if not metrics or key not in metrics[index]:
            return None
        return float(metrics[index][key])

    return {
        "all_finite": all(metric_is_finite(metric) for metric in metrics),
        "max_grad_norm": max((float(metric["grad_norm"]) for metric in metrics), default=0.0),
        "max_grad_rms": max((float(metric["grad_rms"]) for metric in metrics), default=0.0),
        "loss_start": float(metrics[0]["loss"]) if metrics else None,
        "loss_end": float(metrics[-1]["loss"]) if metrics else None,
        "objective_loss_start": endpoint("objective_loss", 0),
        "objective_loss_end": endpoint("objective_loss", -1),
        "masked_ce_start": endpoint("masked_ce", 0),
        "masked_ce_end": endpoint("masked_ce", -1),
        "bpt_start": endpoint("bpt", 0),
        "bpt_end": endpoint("bpt", -1),
        "suffix_ce_start": endpoint("suffix_ce", 0),
        "suffix_ce_end": endpoint("suffix_ce", -1),
        "top2_mass_end": endpoint("top2_mass", -1),
        "posterior_entropy_end": endpoint("posterior_entropy", -1),
        "semantic_mask_ratio_end": endpoint("semantic_mask_ratio", -1),
        "physical_corruption_ratio_end": endpoint("physical_corruption_ratio", -1),
        "suffix_task_ratio_end": endpoint("suffix_task_ratio", -1),
        "random_task_ratio_end": endpoint("random_task_ratio", -1),
        "correction_alignment_start": float(metrics[0]["correction_alignment"]) if metrics else None,
        "correction_alignment_end": float(metrics[-1]["correction_alignment"]) if metrics else None,
        "steps": len(metrics),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from hagi_v4.config import load_config
    from hagi_v4.data.dataset import validate_terminal_eos
    from hagi_v4.model.hagi_v4 import HAGIv4
    from hagi_v4.train.loop import train_step
    from hagi_v4.train.optim import build_optimizer

    cfg = load_config(args.config, **{"train.max_steps": args.steps, "train.distill_enabled": False})
    cfg.model.attention.max_seq_len = min(cfg.model.attention.max_seq_len, 32)
    cfg.train.seq_len = min(cfg.train.seq_len, 32)
    cfg.train.batch_size = 1
    model = HAGIv4(cfg).to(args.device)
    optimizer = build_optimizer(model, cfg)
    metrics = []
    for step in range(args.steps):
        microbatches = []
        for micro_idx in range(cfg.train.grad_accum_steps):
            ids = torch.full(
                (1, cfg.train.seq_len),
                cfg.train.pad_token_id,
                dtype=torch.long,
                device=args.device,
            )
            content_len = max(cfg.train.seq_len - 2, 0)
            if content_len:
                content_ids = [
                    token_id
                    for token_id in range(cfg.model.vocab_size)
                    if token_id not in (cfg.train.eos_token_id, cfg.train.pad_token_id)
                ]
                if not content_ids:
                    raise ValueError("diagnostic batches require at least one non-EOS, non-pad token")
                choices = torch.randint(0, len(content_ids), (1, content_len), device=args.device)
                ids[:, :content_len] = torch.tensor(content_ids, device=args.device)[choices]
            ids[:, content_len] = cfg.train.eos_token_id
            valid_target_mask = validate_terminal_eos(
                ids,
                eos_token_id=cfg.train.eos_token_id,
                pad_token_id=cfg.train.pad_token_id,
            )
            microbatches.append(
                {
                    "input_ids": ids,
                    "targets": ids.clone(),
                    "valid_target_mask": valid_target_mask,
                    "fingerprint": (step, micro_idx),
                }
            )
        metric = train_step(model, microbatches, optimizer, cfg, step)
        metric["parameters_finite"] = all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
        metric["gradients_finite"] = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()
        )
        metrics.append(metric)
    result = collect_diagnostics(metrics)
    result["all_finite"] = result["all_finite"] and all(
        metric["parameters_finite"] and metric["gradients_finite"] for metric in metrics
    )
    print(json.dumps(result, indent=2))
    return 0 if result["all_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
