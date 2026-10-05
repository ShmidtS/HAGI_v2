"""How much of the merge's communication is LEARNED?

The mixer finding has been reported for three rounds as "the channel
switched itself off": ``mixer.gain`` went 0 -> -0.0606 -> -0.0051 during
joint training. That is a true statement about the scalar. It is a
misleading statement about the CHANNEL, and this script measures why.

``HadamardMixer.forward`` is

    y = hadamard_blocks(x) + branch_scale(down(silu(gate(h)) * up(h))) * gain

At ``gain = 0`` the mixer is NOT the identity -- it is the fixed
orthonormal Hadamard re-mix over the expert axis. So there are two
channels, not one, and they behave differently:

  - the FIXED channel is parameter-free, orthonormal, and identical in
    every generation. It mixes; it cannot adapt.
  - the LEARNED channel is a rank-64 residual scaled by one learnable
    number.

Which of the two carries the communication is a measurement, not a
comment, and the answer changes the diagnosis. If the learned residual
is a few percent of the fixed path, then the merge's cross-expert
communication is almost entirely a FIXED orthogonal transform that
every generation shares -- which is a different problem from "the
channel decayed", and a different one to fix.

R104 requires gain to be PRODUCED each cycle, and R107 requires frontier
production at ``beta >= 10``. Neither can come from a transform that is
the same in every generation and is never trained.

    python scripts/measure_channel_split.py --ckpt checkpoints/mixonly_gen3/step-0000400.pt
"""

from __future__ import annotations

import argparse

import torch

from hagi.config import load_config
from hagi.model.merge import MergedHAGI, hadamard_blocks


def split(ckpt: str, config: str, seed: int = 0) -> dict:
    """The two channels' contributions on one synthetic stream.

    Uses random input rather than real tokens: the ratio being asked
    about is a property of the operator's geometry (an orthonormal
    transform against a rank-64 residual times a scalar), and a random
    stream is an unbiased sample of it. Reported alongside the
    parameter norms so the number is not mistaken for a data-dependent
    one.
    """
    cfg = load_config(config)
    model = MergedHAGI(cfg).float()
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    mixer = model.mixers[0]
    torch.manual_seed(seed)
    x = torch.randn(1, 4, cfg.model.hidden_size)

    with torch.no_grad():
        fixed = hadamard_blocks(x, mixer.n_blocks, mixer.group_sizes)
        h = mixer.norm(x)
        raw = mixer.down(torch.nn.functional.silu(mixer.gate(h)) * mixer.up(h))
        scaled = mixer.branch_scale(raw) * mixer.gain

    fixed_n = float(fixed.norm())
    raw_n = float(raw.norm())
    scaled_n = float(scaled.norm())
    return {
        "gain": float(mixer.gain.detach()),
        "fixed_norm": fixed_n,
        "learned_raw_norm": raw_n,
        "learned_scaled_norm": scaled_n,
        "raw_vs_fixed": raw_n / fixed_n,
        "scaled_vs_fixed": scaled_n / fixed_n,
        "gain_suppression": raw_n / scaled_n if scaled_n else float("inf"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/dbridge_gen7_merged.yaml")
    args = ap.parse_args()

    d = split(args.ckpt, args.config)
    print(f"mixer.gain                 {d['gain']:+.6f}")
    print(f"fixed Hadamard path norm   {d['fixed_norm']:10.4f}")
    print(f"learned residual, raw      {d['learned_raw_norm']:10.4f}"
          f"   ({100 * d['raw_vs_fixed']:.1f}% of fixed)")
    print(f"learned residual, x gain   {d['learned_scaled_norm']:10.4f}"
          f"   ({100 * d['scaled_vs_fixed']:.2f}% of fixed)")
    print()
    print(f"the scalar suppresses the learned path by {d['gain_suppression']:.1f}x")
    print()
    if d["scaled_vs_fixed"] < 0.05:
        print("VERDICT: cross-expert communication is ~entirely the FIXED")
        print("         orthogonal re-mix. The learned channel is a trace.")
        print("         Nothing here adapts between generations.")
    else:
        print("VERDICT: the learned channel is a substantial share of the")
        print("         merge; the fixed Hadamard is not carrying it alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
