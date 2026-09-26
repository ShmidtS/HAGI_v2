"""Best-of-K merge selection: cheap screening, then one full run.

The idea comes from Explorative Modeling (arXiv 2607.27372), which attacks
mode-averaging by sampling K candidates per step, scoring them all, and
committing to the argmin instead of the average. Its transferable part for us
is the selection primitive, not its numbers: a merge configuration is a choice,
several choices are usually available, and scoring them all before committing
is cheaper than committing to the first one tried.

Concretely, for a candidate merge the loop considers K variations that are
cheap to enumerate and plausibly different:

  1. a subset of the available experts (all N, or the best subset by
     per-expert held-out score);
  2. ``mixer_init_scale`` in {0.0, small positive} -- at 0 the cross-expert
     mixer starts as the identity, which is what every merged run so far used;
  3. expert ordering, which changes nothing in principle but changes the
     block-diagonal layout and therefore the random init of the joint.

Screening is deliberately cheap: a short joint run with no held-out claim
attached, purely to rank candidates. Only the winner gets the full budget.
This is the ``save_mem_mode`` idea in the only form that makes sense offline:
pay for K small runs instead of K full ones.

Honest accounting of what this can and cannot do
-----------------------------------------------
The screening signal is noisy. A 300-step run ranks candidates that then get
9000 steps, so a candidate can win the screen and lose the real comparison.
That is a real limitation, not a detail: the screen is a prior, not evidence,
and the ledger records the screen scores alongside the final verdict so a
wrong pick is visible rather than hidden. With K=1 this reduces exactly to the
existing single-run behaviour, which is why the K=1 control is free.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

LOG = logging.getLogger("growth.merge_select")

# Small mixer gains worth trying. 0.0 is the default every merged run used.
MIXER_SCALES = (0.0, 0.05)


@dataclass(frozen=True)
class MergeCandidate:
    """One enumerable way to merge a set of expert checkpoints."""

    experts: tuple[str, ...]
    mixer_init_scale: float
    label: str

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_experts": len(self.experts),
            "mixer_init_scale": self.mixer_init_scale,
            "experts": list(self.experts),
        }


def enumerate_candidates(
    expert_checkpoints: list[str],
    *,
    mixer_scales: tuple[float, ...] = MIXER_SCALES,
) -> list[MergeCandidate]:
    """All-experts, across mixer scales.

    Subset search is deliberately not attempted yet. With 24 experts the subset
    space is 2^24, and a 300-step screen does not resolve a difference that
    small against 0.2 nats of held-out noise -- it would select on noise while
    appearing to select on merit. Subset selection needs a screen sharp enough
    to beat the noise, which is a measurement this project has not made.
    """
    experts = tuple(expert_checkpoints)
    return [
        MergeCandidate(experts, scale, f"all{len(experts)}_ms{scale}")
        for scale in mixer_scales
    ]


def write_candidate_config(
    base_config: Path,
    candidate: MergeCandidate,
    out_path: Path,
    *,
    screen_steps: int | None = None,
    checkpoint_dir: str | None = None,
) -> Path:
    """Render a candidate into its own config file.

    ``screen_steps`` shortens ``max_steps`` for the cheap ranking pass, and
    ``checkpoint_dir`` keeps each candidate's artefacts from colliding with the
    full run that follows.
    """
    raw = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    raw["merge"]["expert_checkpoints"] = list(candidate.experts)
    raw["merge"]["mixer_init_scale"] = candidate.mixer_init_scale
    if screen_steps:
        raw["train"]["max_steps"] = int(screen_steps)
    if checkpoint_dir:
        raw["train"]["checkpoint_dir"] = checkpoint_dir
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out_path


def screen_steps_for(budget: int, fraction: float = 0.06, floor: int = 200) -> int:
    """How many steps a screening run should get.

    6% of the full budget, at least ``floor`` steps. Too short and the ranking
    reflects init rather than merge quality; too long and K candidates cost
    more than the single full run they were meant to save.
    """
    return max(floor, int(budget * fraction))
