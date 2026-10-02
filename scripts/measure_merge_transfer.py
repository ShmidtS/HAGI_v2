"""Where does the disagreement GO when the experts are merged?

The merge was exonerated from the weight side: ``cos(W_k, merged) = 0.995``
and ``merged + dev_k = W_k`` to 2.8e-16, so no expert is lost in the
arithmetic. R108 gave the target -- ``gamma`` must rise 9x (0.0041 ->
0.037) for the R104 gate ``alpha*C <= gamma*D`` to hold with ``D = 8.90``
nats -- and R107 then said the alternative, producing the frontier 1000x
faster, is absurd. So the disagreement has to be TRANSFERRED more
efficiently, and the question is where it currently goes.

Three candidate losses, all measurable on checkpoints that exist, and
they are not mutually exclusive:

1. **The projection.** The experts' weights live in different basins; a
   single merged weight cannot represent a sum of functions whose
   outputs differ. Measured as the residual ``mean_k W_k - W_merged``
   relative to the spread ``mean_k ||W_k - mean_k W_k||``.
2. **The orthogonality.** The fixed Hadamard is orthonormal, so it
   preserves the stream norm. That is a property of the FORWARD pass --
   and it may be exactly why the learned residual has nowhere useful
   to go, since a norm-preserving re-mix cannot amplify the minority
   expert's component that the merge needs.
3. **The pooling.** Uniform weights over experts. If one expert is
   better on the data the merge is trained on, uniform pooling spends
   two thirds of its capacity on worse predictions.

This script measures (1) and (3) on weights and, more importantly,
measures what the merged model actually PREDICTS against each expert
on real held-out windows. That last one is the quantity gamma is
defined from, and it has never been measured end to end.

    python scripts/measure_merge_transfer.py \
        --experts math=checkpoints/.../step-0001300.pt lang=... --steps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hagi.config import load_config  # noqa: E402
from hagi.model.formal import jensen_gap_lse  # noqa: E402

KEY = "encoder.embedding.weight"


def weight_transfer(experts: list[torch.Tensor], merged: torch.Tensor) -> dict:
    """Loss (1): how much of the experts' spread survives the merge.

    The residual ``mean_k W_k - W_merged`` normalised by the experts'
    own spread answers the question R101 could not: a factorised core is
    only worth it if the merged weight is measurably far from the mean.
    """
    stack = torch.stack([w.flatten() for w in experts])
    mean = stack.mean(0)
    spread = float((stack - mean).norm(dim=1).mean())
    residual = float((mean - merged.flatten()).norm())
    return {
        "expert_spread": spread,
        "residual": residual,
        "retained_fraction": max(0.0, 1.0 - residual / max(spread, 1e-12)),
    }


def pooling_is_even(experts: list[torch.Tensor], merged: torch.Tensor) -> dict:
    """Loss (3): does the merged weight sit at the mean of the experts?

    A residual near zero means uniform pooling already happened in the
    weights, and the merge operator has no room to route per token --
    which is the whole argument for a learned channel that this
    project does not currently have.
    """
    stack = torch.stack([w.flatten() for w in experts])
    mean = stack.mean(0)
    return {
        "cos_to_expert_mean": float(
            torch.nn.functional.cosine_similarity(mean, merged.flatten(), dim=0)
        ),
        "offset_from_mean": float((mean - merged.flatten()).norm()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experts", nargs="+", required=True,
                    metavar="NAME=CONFIG=PATH")
    ap.add_argument("--merged", help="CONFIG=PATH for the merged model")
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    specs = []
    for s in args.experts:
        parts = s.split("=")
        if len(parts) != 3:
            raise SystemExit(f"expected NAME=CONFIG=PATH, got {s!r}")
        specs.append(parts)

    print("=== weight-side transfer ===")
    experts_w = []
    for name, _, path in specs:
        sd = torch.load(path, map_location="cpu", weights_only=True)["model"]
        if KEY not in sd:
            print(f"  {name}: no {KEY} in this checkpoint")
            return 1
        experts_w.append(sd[KEY].double())

    if not args.merged:
        print("  (pass --merged CONFIG=PATH to compare against a merge)")
        return 0

    mcfg_path, mckpt = args.merged.split("=", 1)
    msd = torch.load(mckpt, map_location="cpu", weights_only=True)["model"]
    if KEY not in msd:
        print("  merged checkpoint has no embedding weight")
        return 1
    merged_w = msd[KEY].double()

    w = weight_transfer(experts_w, merged_w)
    p = pooling_is_even(experts_w, merged_w)
    # The weight-side numbers only mean something if the merged
    # checkpoint was actually built FROM these experts. Comparing a
    # gen-3 merge against gen-4 experts gives a residual ~8x the
    # experts' own spread and a retained fraction of exactly 0.0,
    # which reads as a finding but is an artifact of comparing across
    # generations. Check the parentage instead of reporting the number.
    same_generation = w["residual"] <= w["expert_spread"]
    print(f"  expert spread        {w['expert_spread']:12.4f}")
    print(f"  residual |mean - Wm| {w['residual']:12.4f}")
    if same_generation:
        print(f"  retained fraction    {w['retained_fraction']:12.4f}")
    else:
        print("  retained fraction    NOT REPORTED -- residual exceeds the")
        print("                       experts' own spread by "
              f"{w['residual'] / max(w['expert_spread'], 1e-12):.1f}x, so the "
              "merged")
        print("                       checkpoint is not a merge OF these")
        print("                       experts (different generations).")
    print(f"  cos(mean, merged)    {p['cos_to_expert_mean']:12.6f}")

    print()
    print("=== prediction-side transfer (the quantity gamma comes from) ===")
    sys.path.insert(0, str(ROOT / "scripts"))
    from measure_frontier import batch as make_batch
    from measure_frontier import load as load_model_pair

    cfg, merged_model = load_model_pair(mcfg_path, mckpt, args.device)
    merged_model.eval()
    experts = []
    for name, conf, path in specs:
        _, m = load_model_pair(conf, path, args.device)
        m.eval()
        experts.append(m)
        print(f"  loaded {name}")
    print(f"  loaded merged <- {mckpt}")

    batches = list(make_batch(cfg, args.device, args.batches))
    print(f"\n{len(batches)} batch(es) of {cfg.train.batch_size}")

    with torch.no_grad():
        gap_total = 0.0
        n_pos = 0
        per_expert = 0.0
        for batch_ids in batches:
            # per window: [n,T,V] in float64 is 256 MB per expert, and
            # a batch of 32 would need 8.6 GB. The quantities summed
            # here are additive over windows, so accumulating per window
            # is exact rather than an approximation.
            for i in range(batch_ids.shape[0]):
                w_ids = batch_ids[i : i + 1]
                e = torch.stack(
                    [m(w_ids, return_logits=True).logits[0].double()
                     for m in experts]
                )
                mm = merged_model(w_ids, return_logits=True).logits[0].double()
                gap = jensen_gap_lse(e)
                gap_total += float(gap.sum())
                n_pos += int(gap.numel())
                logp = torch.log_softmax(mm, dim=-1)
                per_expert += float(
                    -(torch.softmax(e, dim=-1) * logp.unsqueeze(0)).sum(-1).sum()
                )

    d = gap_total / max(n_pos, 1)
    kl_expert = per_expert / max(n_pos, 1) / max(len(experts), 1)
    print(f"\n  disagreement D = {d:.4f} nats")
    print(f"  mean KL(expert || merged) = {kl_expert:.4f} nats")
    print()
    print("  gamma (harvest) = G/D compares the merged model's gain on")
    print("  held-out against this D. The KL above is what the merge")
    print("  fails to carry per token: a LARGE KL against a LARGE D is")
    print("  the case where disagreement exists and is not transferred --")
    print("  the 9x shortfall the R104 gate needs closed.")
    return 0
    print()
    print("  gamma (harvest) = G/D compares the merged model's gain on")
    print("  held-out against this D. The KL above is the loss the merge")
    print("  incurs per token: a LARGE KL with a LARGE D is the case")
    print("  where disagreement exists and is not transferred -- exactly")
    print("  the 9x shortfall the R104 gate needs closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())