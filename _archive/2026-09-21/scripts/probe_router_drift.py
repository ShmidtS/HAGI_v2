"""Router drift measurement: how much does ternary compression change top-6?

GEMQ finding: after 1.5-bit quantization >40% of tokens change routing.
Drift measure per layer:
  D_L = (1/N) * sum_x | top6(x_drifted) \\ top6(x_clean) | / 6
computed by running the same tokens through the ORIGINAL prefix vs the
COMPRESSED prefix (I4X_LAYERS) and comparing router top-6 sets.

Run AFTER the pass (GPU, model):
  SEQ_CH=4096 I4X_LAYERS=<all> python scripts/probe_router_drift.py --tokens 8192
Twice: once with I4X_LAYERS empty (clean reference saved), once compressed.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))


def run(tokens, compressed, out_path, seq_ch=4096):
    import dsv4_collect_seq as cs  # noqa: F401  (env-driven COMP set)
    import dsv4_generate_ttt as gen
    import gigatoken

    torch.set_default_device("cuda")
    # collector machinery: load model + skeleton hooks
    model = gen.load_model()
    router_cache = {}

    n = tokens.shape[1]
    with torch.no_grad():
        for c0 in range(0, n, seq_ch):
            chunk = tokens[:, c0 : c0 + seq_ch]
            # forward with hooks capturing router logits per layer
            gen_hooked_forward(model, chunk, router_cache)
    torch.save(router_cache, out_path)
    print(f"saved {len(router_cache)} layers -> {out_path}", flush=True)


def gen_hooked_forward(model, chunk, router_cache):
    """Minimal forward that records router top-6 per layer (uses collector's
    hook installation if available, else a direct scan of decoder layers)."""
    raise SystemExit("wired at run time against the collector's hook API - "
                     "see dsv4_collect_seq.py load_router/hook section")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    out = args.out or ("checkpoints_dsv4/router_drift_compressed.pt"
                       if os.environ.get("I4X_LAYERS") else
                       "checkpoints_dsv4/router_drift_clean.pt")
    g = torch.Generator().manual_seed(4321)
    tokens = torch.randint(0, 129000, (2, args.tokens), generator=g)
    run(tokens, compressed=bool(os.environ.get("I4X_LAYERS")), out_path=out)


if __name__ == "__main__":
    main()
