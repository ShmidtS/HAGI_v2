"""Targeted search: minimize val_loss AND bpt simultaneously.

Uses production ratios (d147dae) with target_params=17.5M (H=576 C=288 2/7/2).
Explores: lr, wd, muon_lr, muon_wd, mask_ratio, warmup, refinement_iters,
gradient_checkpointing, moe_int scaling, msa_top_k, batch_size, seq_len.

Score = 0.6 * normalized_val_loss + 0.4 * normalized_bpt
"""

import json, logging, random, sys, time, os, math, copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("opt")

import torch
from hagi_v4.config import HAGIv4Config, auto_configure
from hagi_v4.model.hagi_v4 import HAGIv4
from hagi_v4.train.losses import LossAggregator
from hagi_v4.train.optim import build_optimizer
from hagi_v4.model.masking import create_random_mask
from hagi_v4.research.dataset import TinyStoriesConfig, load_tinystories
from torch.utils.data import DataLoader

RESULTS_PATH = "research_results/loss_bpt_search.json"


def make_cfg(overrides: dict) -> HAGIv4Config:
    cfg = HAGIv4Config()
    m = auto_configure(17_500_000, 49154)
    cfg.model = m
    cfg.model.vocab_size = 49154
    cfg.train.max_steps = 2000
    cfg.train.distill_enabled = False
    cfg.train.tokenizer = "HuggingFaceTB/SmolLM2-135M"
    cfg.train.checkpoint_dir = "research_checkpoints"
    cfg.train.checkpoint_interval = 0
    cfg.train.use_two_phase_schedule = False
    cfg.train.log_grade_variance = False

    for k, v in overrides.items():
        if k.startswith("_"):
            continue
        parts = k.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg


SEARCH_SPACE = [
    # (name, key, values)
    ("lr", "train.learning_rate", [1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4]),
    ("muon_lr", "train.muon_lr", [0.01, 0.02, 0.03, 0.04]),
    ("wd", "train.weight_decay", [0.10, 0.20, 0.30, 0.47]),
    ("muon_wd", "train.muon_weight_decay", [0.40, 0.50, 0.60]),
    ("mask", "model.masking.mask_ratio", [0.25, 0.30, 0.35, 0.40, 0.45]),
    ("warmup", "train.warmup_steps", [50, 100, 200, 500]),
    ("iters", "model.refinement.num_iterations", [2, 3, 4]),
    ("batch", "train.batch_size", [8, 16, 32]),
    ("seqlen", "train.seq_len", [128, 256, 512]),
    ("moe_int", "model.moe.intermediate_size", [288, 384, 480]),
    ("msa_k", "model.msa.top_k", [4, 6, 8]),
    ("grad_ckpt", "model.refinement.use_gradient_checkpointing", [True, False]),
    ("gp2d_w", "model.gp2d.whiteness_weight", [0.005, 0.01, 0.02]),
    ("coh_w", "train.w_coherence", [0.0005, 0.001, 0.002]),
    ("ib_w", "train.w_ib", [0.005, 0.01, 0.02]),
]


def run_single(name: str, overrides: dict, train_ds, val_ds, device) -> dict:
    cfg = make_cfg(overrides)
    torch.manual_seed(42)

    model = HAGIv4(cfg).to(device)
    if cfg.train.precision == "bf16" and device.type == "cuda":
        model.to(torch.bfloat16)
    model.train()

    optimizer = build_optimizer(model, cfg)
    loss_agg = LossAggregator(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    step = 0
    train_losses = []
    val_losses = []
    grad_target = overrides.get("_grad_norm_target", 1.0)
    t0 = time.time()

    train_iter = iter(train_loader)
    while step < cfg.train.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(device)
        tgts = batch["targets"].to(device)
        masked_ids, mask = create_random_mask(ids, cfg.model.masking.mask_ratio, cfg.model.masking.mask_token_id)

        optimizer.zero_grad(set_to_none=True)
        out = model(masked_ids, targets=tgts, mask=mask, step=step)
        loss = loss_agg(out, tgts, mask, step=step)

        if not torch.isfinite(loss).all():
            step += 1
            continue

        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if grads:
            norm = sum(g.pow(2).sum() for g in grads).sqrt().item()
            if norm > grad_target:
                scale = grad_target / max(norm, 1e-8)
                for g in grads:
                    g.mul_(scale)
        optimizer.step()

        if step % 500 == 0:
            train_losses.append(loss.item())

        if step % 500 == 0 and step > 0:
            model.eval()
            tl, tc = 0.0, 0
            with torch.no_grad():
                for vi, vb in enumerate(val_loader):
                    if vi >= 30:
                        break
                    vi_ids = vb["input_ids"].to(device)
                    vi_tgts = vb["targets"].to(device)
                    vT = vi_ids.shape[1]
                    vmask = torch.zeros_like(vi_ids, dtype=torch.bool)
                    vmask[:, vT // 2 :] = True
                    vout = model(vi_ids, targets=vi_tgts, mask=vmask, step=0)
                    if vout.ce_loss is not None:
                        tl += vout.ce_loss.item() * vi_ids.shape[0]
                        tc += vi_ids.shape[0]
            vl = tl / max(tc, 1)
            val_losses.append(vl)
            model.train()

        step += 1

    elapsed = time.time() - t0

    # Final eval
    model.eval()
    tl, tc = 0.0, 0
    with torch.no_grad():
        for vi, vb in enumerate(val_loader):
            if vi >= 50:
                break
            vi_ids = vb["input_ids"].to(device)
            vi_tgts = vb["targets"].to(device)
            vT = vi_ids.shape[1]
            vmask = torch.zeros_like(vi_ids, dtype=torch.bool)
            vmask[:, vT // 2 :] = True
            vout = model(vi_ids, targets=vi_tgts, mask=vmask, step=0)
            if vout.ce_loss is not None:
                tl += vout.ce_loss.item() * vi_ids.shape[0]
                tc += vi_ids.shape[0]
    final_val = tl / max(tc, 1)
    final_ppl = float(torch.exp(torch.tensor(final_val)).item())
    bpt = final_val / 0.6931

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": name,
        "overrides": {k: v for k, v in overrides.items() if not k.startswith("_")},
        "final_val_loss": final_val,
        "final_val_perplexity": final_ppl,
        "bpt": bpt,
        "training_time_sec": elapsed,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "stable": True,
    }


def main():
    Path("research_results").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    ds_cfg = TinyStoriesConfig(n_stories=10000, seq_len=512, seed=42)
    train_ds, val_ds = load_tinystories(ds_cfg)
    logger.info("Dataset loaded")

    # Round 1: one-at-a-time sensitivity scan
    results = []
    baseline_overrides = {
        "train.learning_rate": 2e-4,
        "train.weight_decay": 0.25,
        "train.muon_weight_decay": 0.55,
        "model.masking.mask_ratio": 0.40,
        "train.warmup_steps": 100,
        "train.batch_size": 8,
        "train.seq_len": 512,
    }

    logger.info("=== Round 1: Sensitivity scan ===")

    # Baseline
    logger.info("Running baseline...")
    r = run_single("baseline", baseline_overrides, train_ds, val_ds, device)
    results.append(r)
    logger.info("  baseline: val=%.4f bpt=%.2f", r["final_val_loss"], r["bpt"])

    # Scan each parameter
    for param_name, key, values in SEARCH_SPACE:
        for val in values:
            current = dict(baseline_overrides)
            current[key] = val
            name = param_name + "_" + str(val)
            logger.info("Running %s = %s", name, val)
            try:
                r = run_single(name, current, train_ds, val_ds, device)
                results.append(r)
                logger.info(
                    "  %s: val=%.4f bpt=%.2f (baseline val=%.4f bpt=%.2f)",
                    name,
                    r["final_val_loss"],
                    r["bpt"],
                    results[0]["final_val_loss"],
                    results[0]["bpt"],
                )
            except Exception as e:
                logger.warning("  %s FAILED: %s", name, str(e)[:200])
                results.append({"name": name, "final_val_loss": 99, "bpt": 9999, "stable": False, "overrides": current})

            # Save after each
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2, default=str)

    # Analysis
    stable = [r for r in results if r.get("stable", True) and r["final_val_loss"] < 90]
    if not stable:
        logger.error("No stable results!")
        return

    bpts = [r["bpt"] for r in stable]

    bpt_min, bpt_max = min(bpts), max(bpts)

    for r in stable:
        bpt_n = (r["bpt"] - bpt_min) / max(bpt_max - bpt_min, 1e-8)
        r["score"] = 1.0 - bpt_n

    stable.sort(key=lambda r: r["score"], reverse=True)

    logger.info("")
    logger.info("=" * 70)
    logger.info("LOSS + BPT OPTIMIZATION RESULTS (score = 1.0 - norm_bpt)")
    logger.info("=" * 70)
    for r in stable[:15]:
        logger.info(
            "  %-25s val=%.4f bpt=%.2f score=%.3f overrides=%s",
            r["name"],
            r["final_val_loss"],
            r["bpt"],
            r["score"],
            {k: v for k, v in r.get("overrides", {}).items() if k != "train.seq_len"},
        )

    # Round 2: combine best params
    logger.info("")
    logger.info("=== Round 2: Combine best params ===")
    best = stable[:5]
    combined = dict(baseline_overrides)
    for r in best:
        for k, v in r.get("overrides", {}).items():
            if k in combined and v != combined[k]:
                # Pick the one with better score
                pass  # Keep best's overrides

    # Use top-1 overrides as base, then try combinations
    best_overrides = dict(best[0].get("overrides", baseline_overrides))

    combos = [
        ("combo_best", best_overrides),
        ("combo_best_lowspeed", {**best_overrides, "train.batch_size": 8, "train.seq_len": 512}),
        ("combo_best_hispeed", {**best_overrides, "train.batch_size": 32, "train.seq_len": 256}),
        ("combo_best_nockpt", {**best_overrides, "model.refinement.use_gradient_checkpointing": False}),
        ("combo_best_iters2", {**best_overrides, "model.refinement.num_iterations": 2}),
    ]

    for name, overrides in combos:
        logger.info("Running %s", name)
        try:
            r = run_single(name, overrides, train_ds, val_ds, device)
            results.append(r)
            logger.info("  %s: val=%.4f bpt=%.2f", name, r["final_val_loss"], r["bpt"])
        except Exception as e:
            logger.warning("  %s FAILED: %s", name, str(e)[:200])
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Final report
    stable = [r for r in results if r.get("stable", True) and r["final_val_loss"] < 90]
    bpt_min = min(r["bpt"] for r in stable)
    bpt_max = max(r["bpt"] for r in stable)
    for r in stable:
        bpt_n = (r["bpt"] - bpt_min) / max(bpt_max - bpt_min, 1e-8)
        r["score"] = 1.0 - bpt_n
    stable.sort(key=lambda r: r["score"], reverse=True)

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL TOP-10 (loss + bpt combined)")
    logger.info("=" * 70)
    for r in stable[:10]:
        logger.info("  %-25s val=%.4f bpt=%.2f score=%.3f", r["name"], r["final_val_loss"], r["bpt"], r["score"])
        logger.info("    overrides=%s", r.get("overrides", {}))

    # Save best
    if stable:
        best = stable[0]
        with open("research_results/best_loss_bpt.json", "w") as f:
            json.dump(best, f, indent=2, default=str)
        logger.info("")
        logger.info("BEST: %s val=%.4f bpt=%.2f", best["name"], best["final_val_loss"], best["bpt"])
        logger.info("Overrides: %s", best.get("overrides", {}))


if __name__ == "__main__":
    main()
