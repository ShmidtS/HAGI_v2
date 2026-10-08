"""Filter backfill acts: keep only experts whose checkpoint says n_real == 0.

Usage:
  SEQ_BACKFILL=checkpoints_dsv4/seq_backfill REDUCED=dsv4_reduced \
  python scripts/filter_backfill_acts.py L

Loads acts_layer{L}.pt from the backfill dir, keeps entries for experts
whose existing checkpoint in dsv4_reduced/layer_{L} has n_real == 0
(and drops experts with no checkpoint or n_real > 0), writes the filtered
acts to checkpoints_dsv4/pod_all_tokens/acts_layer{L}.pt (canonical refit
input path) and deletes the stale checkpoint files of those experts so the
refit does not skip them.
"""

import os
import sys

import torch

BACKFILL = os.environ.get("SEQ_BACKFILL", "checkpoints_dsv4/seq_backfill")
REDUCED = os.environ.get("REDUCED", "dsv4_reduced")
POD = os.environ.get("POD", "checkpoints_dsv4/pod_all_tokens")


def main():
    L = int(sys.argv[1])
    src = os.path.join(BACKFILL, f"acts_layer{L}.pt")
    if not os.path.exists(src):
        print(f"layer {L}: no backfill acts, skip")
        return
    acts = torch.load(src, map_location="cpu", weights_only=False)
    ldir = os.path.join(REDUCED, f"layer_{L}")
    keep, rm = {}, []
    for k, rows in acts.items():
        if rows[0].shape[0] == 0:
            continue
        ep = os.path.join(ldir, f"expert_{k}.pt")
        if not os.path.exists(ep):
            continue  # not refit yet (main pass will handle)
        e = torch.load(ep, map_location="cpu", weights_only=False)
        if e.get("n_real", 0) == 0:
            keep[k] = rows
            rm.append(ep)
    if not keep:
        print(f"layer {L}: 0 dead experts with backfill rows, nothing to do")
        return
    out = os.path.join(POD, f"acts_layer{L}.pt")
    torch.save(keep, out)
    for ep in rm:
        os.remove(ep)
    tot = sum(v[0].shape[0] for v in keep.values())
    print(f"layer {L}: {len(keep)} dead experts refit-scheduled ({tot} rows), "
          f"{len(rm)} stale checkpoints removed -> {out}", flush=True)


if __name__ == "__main__":
    main()
