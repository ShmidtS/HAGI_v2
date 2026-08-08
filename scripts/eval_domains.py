"""Evaluate a HAGI checkpoint on per-domain corpora (exact CE / perplexity).

Loads a checkpoint, then for each domain corpus builds a single-domain
dataloader and computes the exact full-alphabet cross-entropy over a fixed
number of batches. This is the honest quality metric for the growing
hypothesis: it shows whether a merged model preserves each expert's
specialized knowledge (per-domain ppl) and whether joint training recovers
overall quality.

Usage:
    python scripts/eval_domains.py --config configs/v48_merged.yaml \
        --resume checkpoints_v48_merged/step-0000020.pt \
        --batches 20

    python scripts/eval_domains.py --config configs/experts/expert_ru.yaml \
        --resume checkpoints_experts/ru/step-0000050.pt --batches 20

The domain mix is taken from the config's ``train.data.weights`` if present,
else from the DOMAINS table below (the same mapping as make_expert_configs.py).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.config import load_config  # noqa: E402
from hagi.data.dataset import build_dataloader  # noqa: E402
from hagi.model.merge import MergedHAGI  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.checkpoint import load_model  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402

# domain -> {corpus: weight} (same mapping as make_expert_configs.py)
DOMAINS: dict[str, dict[str, float]] = {
    "RU": {"wikipedia_ru": 1.0, "oscar_ru": 1.0},
    "EN": {"edu": 1.0, "slimpajama": 1.0, "wikipedia_en": 1.0},
    "MATH": {"openwebmath": 1.0},
    "CODE": {"python_instruct": 1.0},
}


def eval_domain(model: torch.nn.Module, cfg, domain: str, batches: int, device) -> dict:
    """Compute exact CE / ppl on a single domain corpus."""
    weights = DOMAINS[domain]
    # Build a single-domain dataloader by overriding the mix.
    dc = cfg.train.data
    from hagi.data.dataset import PackedMixDataset, load_mix
    from torch.utils.data import DataLoader

    root = dc.data_dir
    mix = load_mix(root, weights)
    dataset = PackedMixDataset(
        data_dir=root,
        seq_len=dc.seq_len,
        eos_token_id=dc.eos_token_id,
        weights=mix,
        seed=dc.seed,
        cross_doc_attention=dc.cross_doc_attention,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=0,
        drop_last=True,
    )
    model.eval()
    ce_sum = 0.0
    n_tokens = 0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            out = model(ids, targets)
            hidden = out.hidden.detach().reshape(-1, out.hidden.shape[-1])
            flat_targets = targets.reshape(-1)
            ce = model.head.exact_loss(hidden, flat_targets)
            ce_sum += ce.item() * flat_targets.numel()
            n_tokens += flat_targets.numel()
            n_batches += 1
            if n_batches >= batches:
                break
    avg_ce = ce_sum / max(1, n_tokens)
    return {
        "domain": domain,
        "exact_ce": avg_ce,
        "ppl": math.exp(min(avg_ce, 20.0)),
        "batches": n_batches,
        "tokens": n_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on per-domain corpora")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", required=True, help="checkpoint step-*.pt path")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    configure_runtime()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    # Build the model matching the config (merged or plain).
    if cfg.merge.enabled:
        model = MergedHAGI(cfg, n_mixers=1, mixer_init_scale=cfg.merge.mixer_init_scale).to(device)
    else:
        model = HAGI(cfg).to(device)
    start_step, _ = load_model(args.resume, model, str(device))
    print(f"loaded {args.resume} at step {start_step}")

    results = []
    for domain in DOMAINS:
        r = eval_domain(model, cfg, domain, args.batches, device)
        results.append(r)
        print(
            f"  {domain:5s} exact_ce={r['exact_ce']:.4f} ppl={r['ppl']:.2f} "
            f"({r['batches']} batches, {r['tokens']} tokens)"
        )
    avg = sum(r["exact_ce"] for r in results) / len(results)
    print(f"  AVG   exact_ce={avg:.4f} ppl={math.exp(min(avg, 20.0)):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
