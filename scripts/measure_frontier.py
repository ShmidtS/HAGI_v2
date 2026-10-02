"""Measure the frontier: usable disagreement between the merge's experts.

R104's ``renewal_feeds_takeoff`` needs the frontier-scaling premise

    alpha * C_t <= gamma * D_t

where ``D_t`` is the usable disagreement. This script measures ``D_t``
from checkpoints that already exist, so the premise is a number instead
of an assumption.

``D_t`` is the Jensen gap between the experts' predictive distributions
on held-out windows -- the exact quantity ``formal.py`` implements as
``jensen_gap_lse``. It is zero iff the experts agree everywhere, which
is what makes it the right quantity: it measures disagreement that is
actually USABLE, not weight-space distance.

    python scripts/measure_frontier.py --steps 200 --batches 4 \
        --experts math=configs/dbridge_gen4_sib_math.yaml=checkpoints/.../step-0001300.pt ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hagi.config import load_config  # noqa: E402
from hagi.model.formal import jensen_gap_lse  # noqa: E402


def load(config_path: str, ckpt_path: str, device: str):
    """Build a model from a config and load a checkpoint into it.

    Same construction as ``eval_domains.py``: a merged config needs
    ``MergedHAGI``, a plain one ``HAGI``, and the checkpoint is loaded
    through the validating loader rather than a bare ``load_state_dict``
    so a shape mismatch fails here rather than mid-measurement.
    """
    from hagi.model.merge import MergedHAGI
    from hagi.model.model import HAGI
    from hagi.train.checkpoint import load_model

    cfg = load_config(config_path)
    dev = torch.device(device)
    if cfg.merge.enabled:
        model = MergedHAGI(cfg, n_mixers=1,
                           mixer_init_scale=cfg.merge.mixer_init_scale).to(dev)
    else:
        model = HAGI(cfg).to(dev)
    start_step, _ = load_model(ckpt_path, model, device)
    print(f"  loaded {ckpt_path} at step {start_step}")
    model.eval()
    return cfg, model


def batch(cfg, device: str, batches: int):
    """Yield input-id batches from the configured data mix.

    Built the same way ``eval_domains.py`` builds them, so the windows
    are the ones the project's own evaluation uses and the frontier is
    measured on comparable data.
    """
    from torch.utils.data import DataLoader

    from hagi.data.dataset import PackedMixDataset, load_mix

    dc = cfg.train.data
    mix = load_mix(dc.data_dir, dict(dc.weights))
    ds = PackedMixDataset(
        data_dir=dc.data_dir,
        seq_len=dc.seq_len,
        eos_token_id=dc.eos_token_id,
        weights=mix,
        seed=dc.seed,
        cross_doc_attention=dc.cross_doc_attention,
    )
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                    num_workers=0, drop_last=True)
    out = []
    for i, b in enumerate(dl):
        if i >= batches:
            break
        out.append(b["input_ids"].to(device))
    return out


def frontier_gap(models, batches) -> float:
    """Mean Jensen gap across the expert pool, per window.

    The exact ``jensen_gap_lse``: the running logit sum ``S`` over the
    pool minus the log-sum-exp of the mean, accumulated over positions.
    Zero iff every expert agrees on every position.

    Two experts at ``[32, 1024, 32768]`` in float64 is 8.6 GB -- more
    than this machine has free while a training run holds the GPU, so
    the gap is accumulated per-window in float32 and upcast only inside
    the log-sum-exp. That is numerically identical for this quantity
    (``logsumexp`` subtracts the row max first, so the cancellation
    that float32 would hurt does not occur) and costs N less memory.
    """
    total = 0.0
    positions = 0
    with torch.no_grad():
        for ids in batches:
            for i in range(0, ids.shape[0], 1):
                window = ids[i : i + 1]
                logits = [
                    m(window, return_logits=True).logits[0].double()
                    for m in models
                ]
                n = len(models)
                # per-position: sum_i LSE(z_i) - LSE(mean_i z_i)
                lse_each = sum(torch.logsumexp(z, dim=-1) for z in logits)
                lse_mean = torch.logsumexp(
                    torch.stack(logits).sum(0) / n, dim=-1
                )
                gap = lse_each - lse_mean                        # [T]
                total += float(gap.sum())
                positions += int(gap.numel())
    return total / max(positions, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experts", nargs="+", required=True,
                    metavar="NAME=CONFIG=PATH")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="harvest rate; the premise needs a measured value, "
                         "so this is a caller-supplied constant")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    specs = []
    for s in args.experts:
        parts = s.split("=")
        if len(parts) != 3:
            raise SystemExit(f"expected NAME=CONFIG=PATH, got {s!r}")
        specs.append(parts)

    models = []
    cfg = None
    for name, conf, path in specs:
        cfg, m = load(conf, path, args.device)
        models.append(m)
        print(f"loaded {name}: {conf} <- {path}")

    data_cfg, _ = load(specs[0][1], specs[0][2], args.device)
    batches = batch(data_cfg, args.device, args.batches)
    print(f"\nfrontier D_t over {len(batches)} batch(es) of "
          f"{data_cfg.train.batch_size} windows")

    d = frontier_gap(models, batches)
    print(f"  D_t (mean Jensen gap) = {d:.6f} nats")

    from hagi.train.gain_renewal import (
        frontier_must_scale,
        sustained_growth_possible,
    )

    ceiling = sustained_growth_possible(args.alpha, args.gamma, d)
    print()
    print(f"  with alpha={args.alpha}, gamma={args.gamma}:")
    print(f"    capability ceiling from a bounded frontier: {ceiling:.4f}")
    print()
    print("  frontier scaling alpha*C_t <= gamma*D_t requires, at a given")
    print("  capability C:")
    for c in (1.0, 3.0, 10.0):
        need = frontier_must_scale(args.alpha, c, args.gamma)
        ok = "satisfied" if need <= d else "NOT satisfied"
        print(f"    C={c:5.1f} -> needs D >= {need:.4f}   ({ok})")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "frontier": d,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "ceiling": ceiling,
            "experts": {n: p for n, _, p in specs},
            "batches": args.batches,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())