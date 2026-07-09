"""Parallel bpt optimizer — 2 workers with shared results.

bpt = bits per token = CE / ln(2) — quality metric (lower = better)
Score = 1.0 - norm_bpt  (lower bpt = higher score)

Worker A: training hyperparams (lr, wd, muon_lr, muon_wd, mask_ratio, warmup)
Worker B: architecture efficiency (iters, grad_ckpt, moe_int, msa_k, batch, seq_len)
"""

import argparse, json, logging, sys, time, os, math, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

import torch
from hagi_v4.config import HAGIv4Config, auto_configure
from hagi_v4.model.hagi_v4 import HAGIv4
from hagi_v4.train.losses import LossAggregator
from hagi_v4.train.optim import build_optimizer
from hagi_v4.model.masking import create_random_mask
from hagi_v4.research.dataset import TinyStoriesConfig, load_tinystories
from torch.utils.data import DataLoader

SHARED_PATH = "research_results/loss_bpt_all.json"
LOCK_PATH = "research_results/lock.json"

BASE_OVERRIDES = {
    "train.learning_rate": 2e-4,
    "train.weight_decay": 0.25,
    "train.muon_weight_decay": 0.55,
    "model.masking.mask_ratio": 0.40,
    "train.warmup_steps": 50,
    "train.batch_size": 16,
    "train.seq_len": 256,
}

REGIONS = {
    "A": {
        "name": "training_params",
        "params": [
            ("lr", "train.learning_rate", [1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4]),
            ("muon_lr", "train.muon_lr", [0.01, 0.02, 0.03, 0.04]),
            ("wd", "train.weight_decay", [0.10, 0.20, 0.25, 0.30, 0.40]),
            ("muon_wd", "train.muon_weight_decay", [0.40, 0.50, 0.55, 0.60]),
            ("mask", "model.masking.mask_ratio", [0.25, 0.30, 0.35, 0.40, 0.45]),
            ("warmup", "train.warmup_steps", [50, 100, 200, 500]),
        ],
    },
    "B": {
        "name": "arch_efficiency",
        "params": [
            ("iters", "model.refinement.num_iterations", [2, 3, 4]),
            ("grad_ckpt", "model.refinement.use_gradient_checkpointing", [True, False]),
            ("batch", "train.batch_size", [8, 16, 32]),
            ("seqlen", "train.seq_len", [128, 256, 512]),
            ("moe_int", "model.moe.intermediate_size", [288, 384, 480]),
            ("msa_k", "model.msa.top_k", [4, 6, 8]),
        ],
    },
    "C": {
        "name": "loss_weights",
        "params": [
            ("coh", "train.w_coherence", [0.0005, 0.001, 0.002, 0.005]),
            ("ib", "train.w_ib", [0.005, 0.01, 0.02]),
            ("whiteness", "model.gp2d.whiteness_weight", [0.005, 0.01, 0.02]),
            ("parity", "train.w_parity", [0.05, 0.10, 0.20]),
            ("extrinsic", "train.w_extrinsic_info", [0.005, 0.01, 0.02]),
        ],
    },
}


def make_cfg(overrides: dict) -> HAGIv4Config:
    cfg = HAGIv4Config()
    m = auto_configure(500_000, 49154)
    cfg.model = m
    cfg.model.vocab_size = 49154
    cfg.train.max_steps = 500
    cfg.train.distill_enabled = False
    cfg.train.tokenizer = "HuggingFaceTB/SmolLM2-135M"
    cfg.train.checkpoint_dir = "research_checkpoints"
    cfg.train.checkpoint_interval = 0
    cfg.train.use_two_phase_schedule = False
    cfg.train.log_grade_variance = False
    all_ov = {**BASE_OVERRIDES, **overrides}
    for k, v in all_ov.items():
        if k.startswith("_"):
            continue
        parts = k.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg


def load_results():
    if not Path(SHARED_PATH).exists():
        return []
    try:
        with open(SHARED_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_result(rd):
    Path("research_results").mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        try:
            with open(LOCK_PATH, "x") as f:
                f.write(str(os.getpid()))
                break
        except FileExistsError:
            time.sleep(0.5)
    try:
        existing = load_results()
        existing.append(rd)
        tmp = SHARED_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        os.replace(tmp, SHARED_PATH)
    finally:
        if Path(LOCK_PATH).exists():
            Path(LOCK_PATH).unlink()


def run_single(name, overrides, train_ds, val_ds, device):
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
    val_loader = DataLoader(val_ds, batch_size=min(cfg.train.batch_size, 16), shuffle=False)

    step = 0
    train_losses = []
    val_losses = []
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
        loss = loss_agg(out, tgts, mask)

        if not torch.isfinite(loss).all():
            step += 1
            continue

        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if grads:
            norm = sum(g.pow(2).sum() for g in grads).sqrt().item()
            if norm > 1.0:
                scale = 1.0 / max(norm, 1e-8)
                for g in grads:
                    g.mul_(scale)
        optimizer.step()

        if step % 100 == 0:
            train_losses.append(loss.item())

        if step % 250 == 0 and step > 0:
            model.eval()
            tl, tc = 0.0, 0
            with torch.no_grad():
                for vi, vb in enumerate(val_loader):
                    if vi >= 20:
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
            logger = logging.getLogger("opt")
            logger.info("[%s] step %d train=%.4f val=%.4f", name, step, loss.item(), vl)

        step += 1

    elapsed = time.time() - t0

    model.eval()
    tl, tc = 0.0, 0
    with torch.no_grad():
        for vi, vb in enumerate(val_loader):
            if vi >= 40:
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

    all_ov = {**BASE_OVERRIDES, **overrides}
    return {
        "name": name,
        "overrides": {k: v for k, v in all_ov.items() if not k.startswith("_")},
        "final_val_loss": final_val,
        "final_val_perplexity": final_ppl,
        "bpt": bpt,
        "training_time_sec": elapsed,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "stable": True,
    }


def run_worker(region, rounds=2):
    logger = logging.getLogger(region)
    device = torch.device("cuda")

    ds_cfg = TinyStoriesConfig(n_stories=10000, seq_len=256, seed=42)
    train_ds, val_ds = load_tinystories(ds_cfg)
    logger.info("Region %s: %s", region, REGIONS[region]["name"])

    for round_idx in range(rounds):
        configs = []
        for param_name, key, values in REGIONS[region]["params"]:
            for val in values:
                configs.append((f"{region}_R{round_idx}_{param_name}_{val}", {key: val}))

        for name, overrides in configs:
            done = set(r["name"] for r in load_results())
            if name in done:
                continue

            logger.info("Running %s", name)
            try:
                r = run_single(name, overrides, train_ds, val_ds, device)
                save_result(r)
                logger.info("[%s] val=%.4f bpt=%.2f", name, r["final_val_loss"], r["bpt"])
            except Exception as e:
                logger.warning("[%s] FAILED: %s", name, str(e)[:300])
                save_result(
                    {
                        "name": name,
                        "final_val_loss": 99,
                        "bpt": 9999,
                        "stable": False,
                        "overrides": {**BASE_OVERRIDES, **overrides},
                    }
                )

    # Recombine best for round 2
    if rounds > 1:
        all_r = load_results()
        stable = [
            r for r in all_r if r.get("stable", True) and r["final_val_loss"] < 90 and r["name"].startswith(region)
        ]
        if stable:
            stable.sort(key=lambda r: r["final_val_loss"])
            best = stable[0]["overrides"]
            logger.info("Best from %s: val=%.4f overrides=%s", region, stable[0]["final_val_loss"], best)

            combos = [
                (f"{region}_combo_best", best),
                (
                    f"{region}_combo_fast",
                    {
                        **best,
                        "model.refinement.use_gradient_checkpointing": False,
                        "model.refinement.num_iterations": 2,
                    },
                ),
                (f"{region}_combo_quality", {**best, "model.refinement.num_iterations": 4}),
            ]
            for name, overrides in combos:
                done = set(r["name"] for r in load_results())
                if name in done:
                    continue
                logger.info("Running %s", name)
                try:
                    r = run_single(name, overrides, train_ds, val_ds, device)
                    save_result(r)
                    logger.info("[%s] val=%.4f bpt=%.2f", name, r["final_val_loss"], r["bpt"])
                except Exception as e:
                    logger.warning("[%s] FAILED: %s", name, str(e)[:300])

    # Final report
    all_r = load_results()
    stable = [r for r in all_r if r.get("stable", True) and r["final_val_loss"] < 90]
    if stable:
        bpt_min, bpt_max = min(r["bpt"] for r in stable), max(r["bpt"] for r in stable)
        for r in stable:
            bpt_n = (r["bpt"] - bpt_min) / max(bpt_max - bpt_min, 1e-8)
            r["score"] = 1.0 - bpt_n
        stable.sort(key=lambda r: r["score"], reverse=True)

        logger.info("")
        logger.info("=== %s FINAL TOP-5 (loss+bpt) ===", region)
        for r in stable[:5]:
            logger.info(
                "  %-30s val=%.4f bpt=%.2f score=%.3f ov=%s",
                r["name"],
                r["final_val_loss"],
                r["bpt"],
                r["score"],
                {
                    k: v
                    for k, v in r.get("overrides", {}).items()
                    if k
                    in [
                        "train.learning_rate",
                        "train.weight_decay",
                        "model.masking.mask_ratio",
                        "model.refinement.num_iterations",
                        "model.refinement.use_gradient_checkpointing",
                        "train.batch_size",
                        "train.seq_len",
                        "train.w_coherence",
                        "train.w_parity",
                    ]
                },
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=["A", "B", "C"], required=True)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    Path("research_results").mkdir(parents=True, exist_ok=True)
    if not Path(SHARED_PATH).exists():
        with open(SHARED_PATH, "w") as f:
            json.dump([], f)
    run_worker(args.region, args.rounds)
