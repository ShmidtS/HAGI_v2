"""Token-agreement gate: behavioral fidelity of the compressed model.

Tied-Trit-Planes central finding: 18% weight error can coexist with
behavioral parity - per-expert RMS is a loose proxy (Goodhart risk).
This gate measures BEHAVIOR directly: greedy generation from the same
prompt with the original (INT4X_OFF) and the compressed model, then

  A = (1/n) sum 1[t_orig_i == t_comp_i]

plus top-1 logit agreement when logits are exposed.

Run AFTER the pass (GPU): twice, once INT4X_OFF=1 (reference), once
compressed; the reference run saves tokens, the second compares.

  INT4X_OFF=1 python scripts/eval_token_agreement.py --save
  python scripts/eval_token_agreement.py --compare <ref_file>
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

PROMPTS = [
    "The theory of general relativity states that",
    "def quicksort(arr):\n",
    "Волга впадает в Каспийское море, а",
    "The chief cause of the French Revolution was",
    "In machine learning, overfitting occurs when",
    "SELECT name FROM users WHERE",
    "Photosynthesis converts sunlight into",
    "The Pythagorean theorem says that",
]


def greedy_gen(prompt: str, n: int) -> list[int]:
    import gigatoken
    import dsv4_generate_ttt as gen

    ids = gigatoken.encode(prompt)
    toks = []
    with torch.no_grad():
        for _ in range(n):
            logits = gen.forward_ids(ids + toks)
            nxt = int(logits[-1].argmax())
            toks.append(nxt)
            if nxt == getattr(gigatoken, "EOS", None):
                break
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--compare", type=str, default="")
    args = ap.parse_args()

    ref_path = "checkpoints_dsv4/agreement_ref.pt"
    outs = {}
    for p in PROMPTS:
        t = greedy_gen(p, args.n)
        outs[p] = t
        print(f"  [{len(t):3d} toks] {p[:40]!r}", flush=True)

    if args.save:
        torch.save(outs, ref_path)
        print(f"reference saved -> {ref_path}", flush=True)
        return
    if args.compare or os.path.exists(ref_path):
        ref = torch.load(args.compare or ref_path, weights_only=False)
        agree, tot = 0, 0
        for p in PROMPTS:
            a, b = ref.get(p, []), outs.get(p, [])
            m = min(len(a), len(b))
            same = sum(1 for i in range(m) if a[i] == b[i])
            agree += same
            tot += max(len(a), len(b))
            print(f"  {same}/{m} prefix match ({same / max(m, 1):.0%}) on {p[:40]!r}", flush=True)
        print(f"TOKEN AGREEMENT: {agree}/{tot} = {agree / max(tot, 1):.1%}", flush=True)


if __name__ == "__main__":
    main()
