"""Round 52: LDT-style state refinement at inference time (zero training).

Idea (Lattice Deduction Transformer, lcrh/lattice-deduction-transformers,
applied as a hypothesis -- their absolute Sudoku numbers are a reconstruction
in progress): the trained block stack is a deduction OPERATOR; applying it
repeatedly with input reinjection refines the state,

    h_{k+1} = RMSNorm(h_k + h_0 + Stack(h_k))

instead of forcing all reasoning into one pass. For HAGI this is the
"loop depth instead of width" compute axis: the same merged gen-2 body
applied K times at eval, no new parameters, no training.

Protocol: gate (8 corpora, weighted exact CE, tail windows, 2x1024/corpus),
K = 1 (baseline = the record), 2, 3. Blind expectation (recorded before
measurement): K=2 mildly HELPS or is neutral on language modeling (the
operator was trained for one pass; LDT trains the loop). A big K=2 win
without loop training would be surprising; a clear LOSS means the trained
operator is not idempotent-refining outside its training distribution
(expected for LM blocks) -- either way the axis is measured, not guessed.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, os.path.join(_REPO, "scripts", "lora"), _REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from hagi.model.norms import RMSNorm
from hagi.train.loop import configure_runtime
from pathlib import Path

from lora_c8_deepen import build_stacked_prior  # pyright: ignore[reportMissingImports]

CORP = ["edu","python_instruct","wikipedia_en","wikipedia_ru","oscar_ru","openwebmath","tinystories","smoltalk"]
W = [0.3571,0.2232,0.0893,0.0714,0.0625,0.0893,0.0536,0.0536]


def refine_forward(model, ids, k: int) -> torch.Tensor:
    """Body applied k times with h0 reinjection + post-loop norm, then head.

    Mirrors MergedHAGI.forward's body path: encoder -> blocks (with the
    merged model's own mask/position machinery reused verbatim via the
    original forward on the FIRST pass; refinement passes re-run the block
    stack through the model's _run_blocks-equivalent path by calling the
    model's blocks directly with the same positions/mask the first pass
    used -- we reconstruct them the same way model.forward does).
    """
    B, T = ids.shape  # T is the INPUT length (targets already shifted)
    device = ids.device
    # First pass: exact stock forward (positions, doc masks, mixers) but
    # capture the pre-head hidden. Reuse the model's own internals.
    enc = model.encoder(ids)
    h = enc
    h0 = enc
    # positions like model.forward
    positions = torch.arange(T, device=device)
    mask = None  # full attention layers (W=0 in this line)
    norm = type(model.out_norm)(
        model.out_norm.weight.shape[0], model.out_norm.weight.shape[1], 1e-5
    ).to(device=device, dtype=h.dtype)
    with torch.no_grad():
        norm.weight.copy_(model.out_norm.weight.data)
    h = enc
    for k in range(k):
        x = h
        for blk in model.blocks:
            x = blk(x, positions, mask)
        if k == 0:
            h = x  # first pass: exactly the stock forward
        else:
            h = norm(h0 + h + x)  # LDT refinement passes (reinject + LN)
    # mixers + out_norm + head, exactly like MergedHAGI.forward tail
    h = model.out_norm(h)
    h = model._apply_mixers(h)
    return h


def main() -> None:
    configure_runtime()
    m = build_stacked_prior().eval()
    pl = torch.load("checkpoints/dbridge_gen2_lora_c8_d2/step-0001600.pt", map_location="cpu", weights_only=False)
    m.load_state_dict({k: v.to("cuda") for k, v in pl["model"].items()}, strict=True)

    batches = []
    for c in CORP:
        p = Path(f"data/{c}.compact.bin"); total = p.stat().st_size // 4
        with p.open("rb") as fh:
            fh.seek((total - 2_000_000) * 4)
            T = np.frombuffer(fh.read(300_000 * 4), dtype=np.uint32).astype(np.int64)
        ids = torch.from_numpy(T[:2048]).reshape(2, 1024)
        batches.append((ids[:, :-1].cuda(), ids[:, 1:].cuda()))

    for k in (1, 2, 3):
        tot = 0.0
        for (x, y), w in zip(batches, W):
            with torch.no_grad():
                h = refine_forward(m, x, k)
                flat = h.reshape(-1, h.shape[-1])
                tot += w * float(m.head.exact_loss(flat, y.reshape(-1)))
        print(f"K={k}: gate CE = {tot:.4f}", flush=True)
    print("baseline record (stock forward, K=1): 3.3835")


if __name__ == "__main__":
    main()
