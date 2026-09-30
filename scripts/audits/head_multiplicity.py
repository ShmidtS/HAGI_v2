"""Head-multiplicity: the FGD/XTC growth trigger.

Adaptive FGD (2606.16926) grows when the merged model's error stops
improving relative to the children -- the manual proxy was agreement
between leaves. XTC (2608.22758) sharpens what "agreement" should mean
for a CONCAT assembly: a merged logit row is a *mixture*, and the
capacity is only really exhausted when the averaged leaf cannot keep
its top head separated from the runner-ups (the head becomes
ambiguous, many near-ties at the top).

Metric: on held-out data, run the mean-of-leaves "consensus" forward
(same input through every leaf, logits averaged), take the softmax
top-1 gap

    gap = p(top1) - p(top2)

and report the fraction of positions whose gap < tau ("head-ambiguous").
A leaf that still separates its head clearly has unused local capacity
-> growth (concat) is worth it. When ambiguity saturates, the leaf's
*vocabulary-level* capacity is exhausted and adding siblings is the
right move (which is what concat does).

Usage:
    python scripts/head_multiplicity.py \
        --configs configs/leafv3_s2001.yaml ... \
        --tau 0.1 --batches 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '../../..'))
for _p in (_HERE, _REPO, os.path.join(_REPO, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hagi.config import load_config  # noqa: E402
from hagi.data.dataset import dataset_path  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.checkpoint import load_payload, config_from_dict  # noqa: E402


def holdout_batches(data_dir: Path, seq: int, bs: int, windows: int):
    raw = {}
    names = ["wikipedia_ru"]
    for name in names:
        p = dataset_path(data_dir, name)
        m = np.memmap(p, dtype=np.int32, mode="r")
        raw[name] = m
    n = len(raw["wikipedia_ru"])
    for w in range(windows):
        off = n - (w + 1) * bs * seq - 1
        ids = raw["wikipedia_ru"][off:off + bs * seq].astype(np.int64)
        yield torch.from_numpy(ids).reshape(bs, seq)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True,
                    help="leaf config paths (checkpoints read from each)")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    leaves = []
    for cp_cfg in args.configs:
        cfg = load_config(cp_cfg)
        ck = Path(cfg.train.checkpoint_dir) / "step-0000800.pt"
        pl = load_payload(str(ck), "cpu")
        m = HAGI(config_from_dict(pl["config"])).to(device).to(torch.bfloat16).eval()
        m.load_state_dict(pl["model"], strict=True)
        leaves.append(m)

    bs, seq, W = args.bs, args.seq, args.batches
    amb_total, pos_total = 0, 0
    for ids in holdout_batches(Path("data"), seq, bs, W):
        x, y = ids[:, :-1].to(device), ids[:, 1:].to(device)
        # consensus logits: mean over leaves, with each leaf's own scale
        acc = None
        for m in leaves:
            out = m(x, y)
            logits = out.logits if hasattr(out, "logits") else out[0]
            acc = logits.float() if acc is None else acc + logits.float()
        logits = acc / len(leaves)
        probs = logits.softmax(-1)
        top2 = probs.topk(2, dim=-1).values
        gap = (top2[..., 0] - top2[..., 1]) < args.tau
        amb_total += int(gap.sum())
        pos_total += gap.numel()

    frac = amb_total / pos_total
    print(f"head_multiplicity: n_leaves={len(leaves)} tau={args.tau} "
          f"ambiguous={frac:.4f} ({amb_total}/{pos_total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
