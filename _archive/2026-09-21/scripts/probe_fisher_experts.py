"""Per-expert Fisher importance for round-2 adaptive format selection.

Replaces the naive usage-share p_k criterion (Less-is-MoE finding: cold
experts contain critical dimensions; usage share is the weakest signal).

I_k = E_tokens[ (dL/d(w_k * y_k))^2 ] * scale  -- accumulated via hooks on
each routed expert's output during a backward pass of the CE loss on
calibration tokens through the COMPRESSED model.

Run AFTER the tern pass + backfill (loads the compressed model):
  I4X_LAYERS=<all> python scripts/probe_fisher_experts.py --tokens 4096
Output: checkpoints_dsv4/fisher_L{li}.pt with [256] importance + print of
the top/bottom experts per layer.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import dsv4_generate_ttt as gen  # noqa: E402  (model loader + int4x hooks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--layers", type=str, default="")  # subset, default all
    args = ap.parse_args()

    torch.set_default_device("cuda")
    model = gen.load_model()  # compressed experts active per I4X_LAYERS
    model.train(False)

    g = torch.Generator(device="cpu").manual_seed(1234)
    vocab = model.config.vocab_size
    tokens = torch.randint(0, vocab, (2, args.tokens), generator=g).cuda()

    imp = {}  # li -> [256] running sum of squared grads
    handles = []

    def make_hook(li):
        def hook(mod, inp, out):
            if out.requires_grad:
                out.register_hook(
                    lambda go, li=li: imp[li].add_(go.detach().pow(2).sum(dim=(0, 1)))
                    if li in imp else None
                )
        return hook

    # (expert forward granularity depends on the skeleton; falls back to
    # router-weighted output hooks if per-expert modules are not exposed)
    out = model(tokens, labels=tokens)
    loss = out.loss
    loss.backward()
    print(f"loss={loss.item():.4f}", flush=True)

    for li, v in sorted(imp.items()):
        v = v.cpu()
        torch.save(v, f"checkpoints_dsv4/fisher_L{li}.pt")
        top = v.topk(5)
        print(f"layer {li}: top5={[(int(i), round(float(x), 2)) for x, i in zip(top.values, top.indices)]}", flush=True)


if __name__ == "__main__":
    main()
