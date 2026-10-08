"""F3 seam (Gall-minimal): logit-mean ensemble over fiber-reconstructed experts.

ResidualSplit (R243): root + contrast_matrix[i] == expert i EXACTLY for
the factorizable tensors. The runtime seam consumes ``root.pt`` +
``fibers.pt`` from the pilot merge: per batch, each expert view is
reconstructed on the fly (state-dict overlay — no extra copies), the
logits are averaged, and exact CE of the AVERAGED distribution is
scored. This is the fiber-routing UPPER BOUND: it recovers the ensemble
Jensen gain the plain root (weight-mean) provably drops. Downgrading to
a cheap learned router (F3 mixer) can only approach this bound.

Usage:
    python scripts/growth/eval_fiber_ensemble.py \
        --config configs/latent_leaf_math.yaml \
        --root checkpoints/latent_gen1_root/root.pt \
        --fibers checkpoints/latent_gen1_root/fibers.pt \
        --batches 10
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.config import load_config  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402

DOMAINS = {
    "RU": {"wikipedia_ru": 1.0, "oscar_ru": 1.0},
    "EN": {"edu": 1.0, "slimpajama": 1.0, "wikipedia_en": 1.0},
    "MATH": {"openwebmath": 1.0},
    "CODE": {"python_instruct": 1.0},
    "HELD_MATH": {"camelmath": 1.0},
    "HELD_CHAT": {"openhermes": 1.0},
}


def expert_views(root_sd: dict, fibers: dict):
    """Yield per-expert state dicts: factorizable tensors get +contrast[i]."""
    n = len(next(iter(fibers.values())))
    for i in range(n):
        sd = dict(root_sd)
        for k, contrasts in fibers.items():
            sd[k] = root_sd[k].float() + contrasts[i].float()
        yield i, sd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--fibers", required=True)
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    configure_runtime()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    payload = torch.load(args.root, map_location="cpu", weights_only=False)
    root_sd = payload["model"]
    fib = torch.load(args.fibers, map_location="cpu", weights_only=False)["fibers"]

    model = HAGI(cfg).to(device).eval()
    views = list(expert_views(root_sd, fib))
    print(f"fiber ensemble: {len(views)} expert views over {len(fib)} tensors")

    # Precompute per-view parameter tensors on device (small model).
    view_sds = [sd for _, sd in views]

    from hagi.data.dataset import PackedMixDataset, load_mix
    from torch.utils.data import DataLoader

    dc = cfg.train.data
    results = []
    with torch.no_grad():
        for domain, weights in DOMAINS.items():
            mix = load_mix(dc.data_dir, weights)
            dataset = PackedMixDataset(
                data_dir=dc.data_dir, seq_len=dc.seq_len,
                eos_token_id=dc.eos_token_id, weights=mix, seed=dc.seed,
                cross_doc_attention=dc.cross_doc_attention,
            )
            loader = DataLoader(dataset, batch_size=cfg.train.batch_size,
                                num_workers=0, drop_last=True)
            ce_sum, n_tok, n_b = 0.0, 0, 0
            for batch in loader:
                ids = batch["input_ids"].to(device)
                targets = batch["targets"].to(device)
                probs = None
                for sd in view_sds:
                    model.load_state_dict(sd, strict=True)
                    out = model(ids, targets)
                    hidden = out.hidden.detach()
                    # exact head loss works on hidden; for the ensemble we
                    # need the distribution: use head.logits if exposed.
                    logits = model.head.logits(hidden.reshape(-1, hidden.shape[-1]))
                    lp = torch.log_softmax(logits.float(), dim=-1)
                    probs = lp.exp() if probs is None else probs + lp.exp()
                probs /= len(view_sds)
                logp = probs.clamp_min(1e-12).log()
                flat_targets = targets.reshape(-1)
                ce = torch.nn.functional.nll_loss(
                    logp, flat_targets, reduction="sum"
                )
                ce_sum += ce.item()
                n_tok += flat_targets.numel()
                n_b += 1
                if n_b >= args.batches:
                    break
            avg = ce_sum / max(1, n_tok)
            r = {"domain": domain, "exact_ce": avg,
                 "ppl": math.exp(min(avg, 20.0))}
            results.append(r)
            print(f"  {domain:10s} exact_ce={avg:.4f} ppl={r['ppl']:.2f}")
    avg_all = sum(r["exact_ce"] for r in results) / len(results)
    print(f"  AVG        exact_ce={avg_all:.4f}")
    print(json.dumps({"avg_exact_ce": avg_all, "domains": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
