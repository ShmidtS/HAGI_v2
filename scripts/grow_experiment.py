"""grow_experiment.py — Full growing pipeline experiment.

Trains N specialist HAGI models to saturation, concatenates them,
attaches CrossExpertMixers, and runs 3 integration modes.
Records exact_CE per domain and global at every checkpoint.

Usage
-----
python scripts/grow_experiment.py --config configs/smollm2.yaml \\
    --domains ru en code math \\
    --expert-steps 30000 \\
    --integration-steps 5000 \\
    --out-dir grow_results

For a quick smoke test:
    python scripts/grow_experiment.py --config configs/smollm2.yaml \\
        --domains ru en \\
        --expert-steps 200 --integration-steps 500 \\
        --batch-size 4 --seq-len 64 \\
        --out-dir grow_smoke --device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW

# ── HAGI imports ────────────────────────────────────────────────────────────────────────────
from hagi.config import load_config
from hagi.model.model import HAGI
from hagi.train.checkpoint import save_checkpoint, load_model
from hagi.model.grow import (
    concat_experts,
    attach_cross_mixers,
    set_integration_mode,
    verify_block_diagonal,
)

log = logging.getLogger("grow_experiment")


# ────────────────────────────────────────────────────────────────────────────
# Mini training loop (simplified – no grad accum, no WSD schedule, no bf16
# casting helpers; this is for correctness / speed-of-experiment, not
# production throughput).
# ────────────────────────────────────────────────────────────────────────────

class FakeDataset:
    """Yields random token batches for smoke-testing when real data is absent."""

    def __init__(self, vocab_size: int, seq_len: int, device: torch.device) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device

    def __iter__(self):
        while True:
            ids = torch.randint(0, self.vocab_size, (1, self.seq_len + 1), device=self.device)
            yield ids[:, :-1], ids[:, 1:]


def load_domain_data(data_dir: Path, domain: str, cfg, device: torch.device):
    """Try to load a real .bin dataset for a domain, else return FakeDataset."""
    bin_path = data_dir / f"{domain}.bin"
    if bin_path.exists():
        try:
            from hagi.data.dataset import MemmapDataset
            ds = MemmapDataset(str(bin_path), cfg.train.data.seq_len + 1)
            log.info("Loaded real data: %s (%d tokens)", bin_path, len(ds) * (cfg.train.data.seq_len + 1))
            return ds
        except Exception as exc:
            log.warning("Could not load %s: %s — using fake data", bin_path, exc)
    log.info("No .bin found for domain '%s', using random fake data", domain)
    return FakeDataset(cfg.model.vocab_size, cfg.train.data.seq_len, device)


def iter_batches(dataset, batch_size: int, device: torch.device):
    """Yield (input_ids, targets) batches from a dataset or FakeDataset."""
    if isinstance(dataset, FakeDataset):
        for ids, tgt in dataset:
            yield ids.to(device), tgt.to(device)
    else:
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        while True:
            for batch in loader:
                x = batch[:, :-1].to(device)
                y = batch[:, 1:].to(device)
                yield x, y


def get_exact_ce(model: HAGI, dataset, n_eval: int, device: torch.device) -> float:
    """Estimate exact CE on n_eval tokens using chunked full-vocab head."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in iter_batches(dataset, 4, device):
            out = model(x, y)
            if out.ce is None:
                continue
            n = int(out.n_tokens or x.numel())
            total_loss += float(out.ce) * n
            total_tokens += n
            if total_tokens >= n_eval:
                break
    model.train()
    return total_loss / max(1, total_tokens)


def train_expert(
    cfg,
    domain: str,
    data_dir: Path,
    max_steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    out_dir: Path,
    log_interval: int = 100,
    plateau_window: int = 500,
    plateau_delta: float = 0.002,
) -> tuple[HAGI, Path]:
    """Train one expert to saturation on a single domain.

    Stops early if the loss plateau is detected: no improvement > plateau_delta
    over the last plateau_window steps. Always runs at least max_steps // 4 steps.

    Returns:
        (trained HAGI model, path to best checkpoint).
    """
    log.info("=== Training expert for domain '%s' (max %d steps) ===", domain, max_steps)
    model = HAGI(cfg).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    ds = load_domain_data(data_dir, domain, cfg, device)
    loss_history: list[float] = []
    best_loss = float("inf")
    best_ckpt: Path | None = None
    ckpt_dir = out_dir / f"expert_{domain}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_csv = ckpt_dir / "train_log.csv"

    with open(results_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "loss_ce", "exact_ce", "elapsed_s"])
        t0 = time.time()

        for step, (x, y) in enumerate(iter_batches(ds, batch_size, device)):
            if step >= max_steps:
                break

            model.train()
            opt.zero_grad()
            out = model(x, y)
            loss = out.loss
            if loss is None or not loss.isfinite():
                log.warning("step %d: non-finite loss, skipping", step)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ce_val = float(out.ce or loss)
            loss_history.append(ce_val)

            if step % log_interval == 0:
                elapsed = time.time() - t0
                exact_ce = get_exact_ce(model, ds, 512, device)
                writer.writerow([step, f"{ce_val:.4f}", f"{exact_ce:.4f}", f"{elapsed:.1f}"])
                fh.flush()
                log.info(
                    "[%s] step %5d | loss %.4f | exact_ce %.4f | %.1fs",
                    domain, step, ce_val, exact_ce, elapsed,
                )
                if exact_ce < best_loss:
                    best_loss = exact_ce
                    best_ckpt = save_checkpoint(model, cfg, step, ckpt_dir, keep_last=1)

            # Plateau detection: only after 1/4 of max_steps
            if step >= max_steps // 4 and len(loss_history) >= plateau_window:
                window = loss_history[-plateau_window:]
                improvement = max(window) - min(window)
                if improvement < plateau_delta:
                    log.info(
                        "[%s] Plateau detected at step %d (improvement %.4f < %.4f), stopping",
                        domain, step, improvement, plateau_delta,
                    )
                    break

    # Save final checkpoint
    final_ckpt = save_checkpoint(model, cfg, len(loss_history), ckpt_dir, keep_last=2)
    if best_ckpt is None:
        best_ckpt = final_ckpt
    log.info("[%s] Training done. best_exact_ce=%.4f, ckpt=%s", domain, best_loss, best_ckpt)
    return model, best_ckpt


def integration_experiment(
    big_model: HAGI,
    domains: list[str],
    data_dir: Path,
    mode: str,
    max_steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    out_dir: Path,
    log_interval: int = 50,
    n_eval_tokens: int = 1024,
) -> dict:
    """Run integration training in one mode, logging CE per domain."""
    log.info("=== Integration: mode='%s', steps=%d ===", mode, max_steps)
    set_integration_mode(big_model, mode)
    big_model.to(device).train()

    # Mixed dataset: round-robin over all domains
    datasets = {d: load_domain_data(data_dir, d, big_model.cfg, device) for d in domains}
    iters = {d: iter_batches(ds, batch_size, device) for d, ds in datasets.items()}

    trainable = sum(p.numel() for p in big_model.parameters() if p.requires_grad)
    log.info("Trainable params: %d", trainable)
    opt = AdamW(
        [p for p in big_model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.1,
    )

    ckpt_dir = out_dir / f"integration_{mode}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_csv = ckpt_dir / "integration_log.csv"
    domain_cols = [f"exact_ce_{d}" for d in domains]
    checkpoints_record = []

    with open(results_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "loss_ce", "exact_ce_global"] + domain_cols + ["elapsed_s"])
        t0 = time.time()

        for step in range(max_steps):
            # Round-robin domain
            domain = domains[step % len(domains)]
            x, y = next(iters[domain])
            big_model.train()
            opt.zero_grad()
            out = big_model(x, y)
            if out.loss is None or not out.loss.isfinite():
                log.warning("step %d: non-finite loss", step)
                continue
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(big_model.parameters(), 1.0)
            opt.step()

            if step % log_interval == 0:
                elapsed = time.time() - t0
                global_ce = get_exact_ce(big_model, next(iter(datasets.values())), n_eval_tokens, device)
                domain_ces = {
                    d: get_exact_ce(big_model, datasets[d], n_eval_tokens // len(domains), device)
                    for d in domains
                }
                row = [step, f"{float(out.ce or out.loss):.4f}", f"{global_ce:.4f}"]
                row += [f"{domain_ces[d]:.4f}" for d in domains]
                row.append(f"{elapsed:.1f}")
                writer.writerow(row)
                fh.flush()
                log.info(
                    "[%s] step %5d | global_ce %.4f | domain_ces %s | %.1fs",
                    mode, step, global_ce,
                    {d: f"{v:.3f}" for d, v in domain_ces.items()},
                    elapsed,
                )
                checkpoints_record.append({
                    "step": step,
                    "global_ce": global_ce,
                    "domain_ces": domain_ces,
                })

    final_ckpt = save_checkpoint(big_model, big_model.cfg, max_steps, ckpt_dir, keep_last=1)
    return {"mode": mode, "checkpoints": checkpoints_record, "final_ckpt": str(final_ckpt)}


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HAGI growing experiment")
    parser.add_argument("--config", required=True, help="Path to base YAML config")
    parser.add_argument("--domains", nargs="+", default=["ru", "en"],
                        help="Domain names (must match .bin files in data_dir)")
    parser.add_argument("--data-dir", default="data", help="Directory with domain .bin files")
    parser.add_argument("--expert-steps", type=int, default=30000,
                        help="Max training steps per expert (early stop on plateau)")
    parser.add_argument("--integration-steps", type=int, default=5000,
                        help="Steps for each integration mode")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Override batch size (0=use config)")
    parser.add_argument("--seq-len", type=int, default=0,
                        help="Override seq_len (0=use config)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--mixer-bottleneck", type=int, default=0,
                        help="CrossExpertMixer bottleneck dim (0=auto)")
    parser.add_argument("--out-dir", default="grow_results", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--skip-expert-training", action="store_true",
                        help="Skip expert training (load from --expert-ckpts)")
    parser.add_argument("--expert-ckpts", nargs="*", default=None,
                        help="Pre-trained checkpoint paths (one per domain), used with --skip-expert-training")
    parser.add_argument(
        "--integration-modes",
        nargs="+",
        default=["mixer_only", "mixer_plus_slow_body", "full"],
        choices=["mixer_only", "mixer_plus_slow_body", "full"],
        help="Which integration modes to run",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # Load base config
    cfg = load_config(args.config)
    if args.batch_size > 0:
        cfg.train.batch_size = args.batch_size
    if args.seq_len > 0:
        cfg.train.data.seq_len = args.seq_len
        cfg.model.attention.max_seq_len = max(cfg.model.attention.max_seq_len, args.seq_len)

    log.info("Config: H=%d L=%d vocab=%d", cfg.model.hidden_size, cfg.model.num_layers, cfg.model.vocab_size)
    log.info("Domains: %s", args.domains)

    # ── Phase 1: Train / load experts ─────────────────────────────────────────────
    experts: list[HAGI] = []
    expert_ckpts: list[Path] = []

    if args.skip_expert_training and args.expert_ckpts:
        for path in args.expert_ckpts:
            m = HAGI(cfg).to(device)
            steps, _ = load_model(path, m, device=str(device))
            experts.append(m)
            expert_ckpts.append(Path(path))
            log.info("Loaded expert from %s (step %d)", path, steps)
    else:
        for domain in args.domains:
            expert, ckpt = train_expert(
                cfg=cfg,
                domain=domain,
                data_dir=data_dir,
                max_steps=args.expert_steps,
                batch_size=cfg.train.batch_size,
                lr=args.lr,
                device=device,
                out_dir=out_dir,
                log_interval=args.log_interval,
            )
            experts.append(expert)
            expert_ckpts.append(ckpt)

    # ── Phase 2: Expert baselines (CE per domain) ────────────────────────────
    log.info("=== Computing expert baselines ===")
    expert_baselines: dict[str, dict[str, float]] = {}
    for domain, expert in zip(args.domains, experts):
        ces = {}
        for eval_domain in args.domains:
            ds = load_domain_data(data_dir, eval_domain, cfg, device)
            ce = get_exact_ce(expert, ds, 2048, device)
            ces[eval_domain] = ce
        expert_baselines[domain] = ces
        log.info("Expert '%s' baselines: %s", domain, {d: f"{v:.3f}" for d, v in ces.items()})

    # ── Phase 3: Concatenate + verify ────────────────────────────────────────
    log.info("=== Concatenating experts ===")
    for e in experts:
        e.to(device)
    big_model = concat_experts(experts)
    big_model.to(device)
    log.info(
        "Grown model: H=%d L=%d heads=%dq/%dkv",
        big_model.cfg.model.hidden_size,
        big_model.cfg.model.num_layers,
        big_model.cfg.model.attention.num_query_heads,
        big_model.cfg.model.attention.num_kv_heads,
    )

    # Verify block-diagonal property
    log.info("Verifying block-diagonal property...")
    deviations = verify_block_diagonal(big_model, experts)
    for k, dev in deviations.items():
        log.info("  %s deviation: %.2e", k, dev)
    assert all(d < 1e-2 for d in deviations.values()), \
        f"Block-diagonal verification failed: {deviations}"
    log.info("Block-diagonal OK")

    # Attach CrossExpertMixers
    bottleneck = args.mixer_bottleneck or None
    mixers = attach_cross_mixers(big_model, bottleneck)
    log.info("Attached %d CrossExpertMixers (bottleneck=%s)", len(mixers), bottleneck or "auto")

    # Save concat checkpoint (no training yet)
    save_checkpoint(big_model, big_model.cfg, 0, out_dir / "concat_init", keep_last=1)

    # ── Phase 4: Integration experiments ────────────────────────────────────
    all_results: dict[str, object] = {
        "expert_baselines": expert_baselines,
        "expert_ckpts": [str(p) for p in expert_ckpts],
        "n_experts": len(experts),
        "H_expert": cfg.model.hidden_size,
        "H_grown": big_model.cfg.model.hidden_size,
        "integration_results": {},
    }

    for mode in args.integration_modes:
        import copy as _copy
        model_copy = _copy.deepcopy(big_model)
        # Re-attach mixers on the fresh copy
        attach_cross_mixers(model_copy, bottleneck)
        result = integration_experiment(
            big_model=model_copy,
            domains=args.domains,
            data_dir=data_dir,
            mode=mode,
            max_steps=args.integration_steps,
            batch_size=cfg.train.batch_size,
            lr=args.lr * (0.1 if mode == "mixer_plus_slow_body" else 1.0),
            device=device,
            out_dir=out_dir,
            log_interval=args.log_interval,
        )
        all_results["integration_results"][mode] = result

    # ── Summary ────────────────────────────────────────────────────────────────────────
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Summary saved to %s", summary_path)

    # Print compact summary table
    print("\n" + "=" * 60)
    print("GROWING EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Experts: {len(experts)} × H={cfg.model.hidden_size}")
    print(f"Grown:       H={big_model.cfg.model.hidden_size}")
    print()
    print("Expert baselines (exact_CE on own domain):")
    for d in args.domains:
        print(f"  {d}: {expert_baselines[d][d]:.4f}")
    print()
    print("Integration results (global CE at final step):")
    for mode, res in all_results["integration_results"].items():
        cks = res.get("checkpoints", [])
        if cks:
            final = cks[-1]
            print(f"  {mode:30s}: global_ce={final['global_ce']:.4f}  domain_ces={final['domain_ces']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
