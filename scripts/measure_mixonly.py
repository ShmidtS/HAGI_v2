"""Does the communication channel PAY when it is the only freedom?

The fork the growth loop is stuck at. Three rounds localised the
bottleneck to the merge operator and its trainable channel:

  - the frontier is large: D = 18.25 nats between three domain experts;
  - the harvest is exact: the realised gain +0.0365 sits far inside
    gamma*D, so nothing is being lost in the arithmetic;
  - so gamma = G/D = 0.002, and the reason is that ``mixer.gain`` went
    0 -> -0.0606 -> -0.0051: the channel switched itself OFF during
    joint training.

R105 ruled out the step size (derived eta_max = 0.0120 against a tuned
lr = 3e-4 -- 40x more room than is used). R107 then said the regime is
CAPPED, because D/C = 5.53 against a cone boundary of alpha/gamma = 50.

R105's verdict says there is no step-size problem -- there is no
mechanism that WANTS to use the room. That is the question this
script answers, and it is only answerable by an A/B with the mixer as
the ONLY trainable parameter (``merge.freeze_experts: true``), so that
no gradient pressure from the experts can move the channel:

  * gain grows in magnitude  -> the channel pays when nothing else can
    move it, and the joint-phase collapse was interference, not the
    channel's nature.  Fix joint, not the mixer.
  * gain drifts to zero      -> a parameter an optimiser is free to
    erase is not a channel.  R104 requires gain to be PRODUCED each
    cycle; a trainable scalar that vanishes is not production.  The
    replacement has to generate disagreement rather than read it.

Reading the trajectory rather than the endpoint is the whole point:
"the end value is larger" and "the value moved toward usefulness" are
different claims, and 800 steps of joint training previously produced a
path that went the wrong way before arriving at a value that only
looked fine.

    python scripts/measure_mixonly.py --config configs/mixonly_gen3.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

STEP = re.compile(r"step (\d+) \| ce=([\d.]+)")
CKPT = re.compile(r"step-(\d+)\.pt$")


def gain_of(path: Path) -> float:
    """The mixer gain from a checkpoint, or nan when it has none."""
    sd = torch.load(path, map_location="cpu", weights_only=True)["model"]
    keys = [k for k in sd if "mixers" in k and k.endswith("gain")]
    if not keys:
        return float("nan")
    return float(sd[keys[0]])


def frozen_check(log: str) -> str:
    """Did the freeze actually take? Read it from the log, not a guess.

    The checkpoint carries no record of which tensors were trainable --
    only the log line emitted at start-up does. Without this check the
    A/B is unfalsifiable: a run where the experts were also trainable
    would look identical in every gain reading, and the whole point is
    that nothing except the mixer could move.
    """
    text = Path(log).read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "freeze_experts" in line:
            return line.split("|", 2)[-1].strip()
    return "NOT FOUND -- the freeze may not have applied"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/mixonly_gen3.yaml")
    ap.add_argument("--log", default="logs/mixonly_gen3.log")
    args = ap.parse_args()

    cfg_dir = Path("checkpoints/mixonly_gen3")
    files = sorted(
        (p for p in cfg_dir.glob("step-*.pt") if CKPT.search(p.name)),
        key=lambda p: int(CKPT.search(p.name).group(1)),
    )
    if not files:
        print(f"no checkpoints under {cfg_dir}")
        return 1

    # baseline: the checkpoint the run started from
    import yaml
    src = Path(args.config).read_text(encoding="utf-8")
    init = (yaml.safe_load(src) or {}).get("train", {}).get("init_from")
    print(f"start   {init}  gain = {gain_of(Path(init)):+.6f}")
    print(f"steps   {len(files)}   log {args.log}")
    print(f"freeze  {frozen_check(args.log)}")
    print()
    print("step        gain       delta     ce")
    prev = None
    ce: dict[int, float] = {}
    if Path(args.log).exists():
        for line in Path(args.log).read_text(
            encoding="utf-8", errors="ignore").splitlines():
            m = STEP.search(line)
            if m:
                ce[int(m.group(1))] = float(m.group(2))
    for p in files:
        step = int(CKPT.search(p.name).group(1))
        g = gain_of(p)
        d = "" if prev is None else f"{g - prev:+.6f}"
        # nearest logged CE at or after this checkpoint
        near = [ce[k] for k in sorted(ce) if k >= step]
        print(f"{step:<8}  {g:+.6f}   {d:>9}   {near[0] if near else float('nan'):.4f}")
        prev = g

    last = gain_of(files[-1])
    base = gain_of(Path(init))
    print()
    print(f"gain {base:+.6f} -> {last:+.6f}   (delta {last - base:+.6f})")
    print()
    print("about the LEARNED SCALAR, not the channel:")
    print("  'the channel deactivated' overstates it. The merge's")
    print("  cross-expert path is mostly the FIXED orthonormal Hadamard")
    print("  (measure_channel_split.py: the learned branch is ~1/3 of")
    print("  the fixed path before this scalar and <1% after it), so")
    print("  what this measures is the learned CORRECTION's drift.")
    if abs(last) < abs(base):
        print()
        print("VERDICT: the learned correction shrank toward zero with the")
        print("         experts frozen -- the same direction joint")
        print("         training drives it, so the joint phase is not what")
        print("         causes it.")
    else:
        print()
        print("VERDICT: the learned correction GREW with the experts frozen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())