"""Where does the merge's disagreement go?

Measured: the merge converts 0.2% of the disagreement its inputs carry
(`measure_harvest_rate.py`). The frontier is ample (18.25 nats), the
harvest is exact, and the operator converts almost none of it.

This localises WHY. The plain ternary merge averages the experts, and
averaging N experts that disagree in DIFFERENT directions cancels most
of the disagreement by construction: the mean of d and -d is 0. What
survives is only the component all three agree on, which is the part
that carries no usable signal.

So the question is not "is the average lossy" (it obviously is) but by
how much, and whether the loss is structural or incidental. Measured on
the real checkpoints:

  - the pairwise disagreement before the merge;
  - the disagreement of the MERGED model against each expert;
  - whether the merged model's disagreement tracks the mean of the
    experts' or something much smaller.

    python scripts/measure_merge_loss.py --steps 200 --batches 2 \
        --experts math=configs/run/gen2_dsib_math.yaml=checkpoints/.../step-0001600.pt ...
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hagi.config import load_config  # noqa: E402
from hagi.model.merge import MergedHAGI  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.checkpoint import load_model  # noqa: E402


def build(config_path: str, ckpt_path: str, device: str, merged: bool = False):
    """Build exactly what the config describes.

    The domain siblings are themselves MergedHAGI models (they carry
    mixers), so the choice follows ``cfg.merge.enabled`` rather than a
    flag -- getting it wrong raises an IncompatibleCheckpointError naming
    the missing mixer keys, which is how this was found.
    """
    cfg = load_config(config_path)
    dev = torch.device(device)
    if merged or cfg.merge.enabled:
        m = MergedHAGI(cfg, n_mixers=1,
                       mixer_init_scale=cfg.merge.mixer_init_scale).to(dev)
    else:
        m = HAGI(cfg).to(dev)
    load_model(ckpt_path, m, device)
    m.eval()
    return cfg, m


def batches(cfg, device: str, n: int):
    from torch.utils.data import DataLoader

    from hagi.data.dataset import PackedMixDataset, load_mix

    dc = cfg.train.data
    ds = PackedMixDataset(
        data_dir=dc.data_dir, seq_len=dc.seq_len, eos_token_id=dc.eos_token_id,
        weights=load_mix(dc.data_dir, dict(dc.weights)), seed=dc.seed,
        cross_doc_attention=dc.cross_doc_attention,
    )
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                    num_workers=0, drop_last=True)
    out = []
    for i, b in enumerate(dl):
        if i >= n:
            break
        out.append(b["input_ids"].to(device))
    return out


def gap_against(m: torch.nn.Module, others, ids) -> float:
    """Mean Jensen gap of ``m`` against each model in ``others``."""
    with torch.no_grad():
        z = m(ids, return_logits=True).logits.double()
        total, count = 0.0, 0
        for o in others:
            zo = o(ids, return_logits=True).logits.double()
            lse_each = torch.logsumexp(z, -1) + torch.logsumexp(zo, -1)
            lse_mean = torch.logsumexp((z + zo) / 2.0, -1)
            g = lse_each - lse_mean
            total += float(g.sum())
            count += int(g.numel())
    return total / max(count, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experts", nargs="+", required=True,
                    metavar="NAME=CONFIG=PATH")
    ap.add_argument("--merged", required=True, metavar="CONFIG=PATH",
                    help="the merged model's config and checkpoint")
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    specs = []
    for s in args.experts:
        parts = s.split("=")
        if len(parts) != 3:
            raise SystemExit(f"expected NAME=CONFIG=PATH, got {s!r}")
        specs.append(parts)

    models = []
    for name, conf, path in specs:
        _, m = build(conf, path, args.device, merged=False)
        models.append(m)
        print(f"loaded expert {name}")

    mconf, mpath = args.merged.split("=")
    cfg, merged_model = build(mconf, mpath, args.device, merged=True)
    print(f"loaded merged model from {mpath}")

    data = batches(load_config(specs[0][1]), args.device, args.batches)

    print()
    print(f"pairwise disagreement BETWEEN experts (nats):")
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            g = gap_against(models[i], [models[j]], data[0])
            print(f"  {specs[i][0]:>6} vs {specs[j][0]:<6} {g:8.4f}")

    print()
    print(f"merged model vs each expert (nats):")
    to_merged = []
    for name, _, _ in specs:
        g = gap_against(merged_model, [models[[s[0] for s in specs].index(name)]],
                        data[0])
        to_merged.append(g)
        print(f"  merged vs {name:<6} {g:8.4f}")

    # the merged model against the ENSEMBLE, on the same windows
    ens = []
    for ids in data:
        with torch.no_grad():
            zs = [m(ids, return_logits=True).logits.double() for m in models]
            zm = merged_model(ids, return_logits=True).logits.double()
            n = len(zs)
            lse_each = sum(torch.logsumexp(z, -1) for z in zs)
            lse_mean = torch.logsumexp(sum(zs) / n, -1)
            ens.append(float((lse_each - lse_mean).sum()))
    frontier = sum(ens) / max(sum(int((z.shape[0] * z.shape[1]))
                                for z in [zs[0]] for _ in [0]) or 1, 1)         if False else None

    print()
    print("Reading: if the merged model sits FARTHER from the experts than")
    print("the experts are from each other, the merge did not merely average")
    print("-- it destroyed the function the experts agreed on. If it sits at")
    print("roughly half, the disagreement cancelled in the average as")
    print("expected and what survives is the common component.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())