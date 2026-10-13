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

from hagi.train.certified_controller import CertifiedVerdict, certified_ab

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
    Screen "scores within the certification band are ties in disguise and never
    move the decision (see :func:`select_screened`).
    """
    return max(floor, int(budget * fraction))


# --- §1/§9: certified selection over the screened candidates -------------


def select_screened(
    scored: list[dict],
    *,
    incumbent_label: str | None = None,
    incumbent_ce: float | None = None,
    incumbent_n: int = 200,
    eps: float = 0.05,
    delta: float = 0.05,
    eval_n: int = 200,
) -> tuple[dict | None, dict]:
    """§1/§9: rank by screen CE (pre-filter), DECIDE by the certified gate.

    Theory (ALGORITHMS.md §1): ``selection_hurts`` forbids selecting a merge
    configuration by standalone CE -- a noisy screen ranking can strictly
    degrade the ensemble. The screen is therefore ONLY a cheap pre-filter
    that orders the candidates; the decision itself goes through the §9
    certified A/B rule (``argmax_pair_certified``): the screen winner is
    committed iff it beats the incumbent by a certified margin
    ``CE_incumbent - CE_winner > 2*eps`` with ``n >= log(2/delta)/(2 eps^2)``
    (Hoeffding premise, reusing ``certified_controller.certified_ab`` --
    one implementation of the gate, no duplication). ``ratio_dominance``
    (R86) justifies spending the single decision on the pre-filter argmin
    rather than certifying every pair: the argmax candidate dominates any
    split of the comparison budget.

    ``incumbent_ce`` is the MEASURED held-out CE of the external incumbent
    (with ``incumbent_n`` its effective sample size); when absent,
    ``incumbent_label`` names the row that proxies the incumbent among the
    screened candidates (the base configuration, i.e. mixer_init_scale 0.0
    -- "the default every merged run used"; its n is ``eval_n``).
    Each row is a dict with ``label`` and ``screen_exact_ce`` (see
    growth_supervisor.screen_candidates).

    Verdicts:
      - ACCEPT   -> the winner is returned (certified better than incumbent).
      - REJECT   -> the INCUMBENT is kept (returned as a synthetic row
        with ``label="incumbent"`` when the incumbent is external): the
        screen winner is certified WORSE; keeping the incumbent is
        certified not-worse.
      - UNDECIDED -> the INCUMBENT is kept via the documented
        not-worse fallback (same shape as certified_controller's
        growth-supervisor fallback): without certified evidence, switching
        on the noisy screen is exactly the ``selection_hurts`` path, so the
        incumbent -- by construction not worse on the shared history -- is
        kept and the undecided margin is recorded for the ledger.
      - No incumbent row (first lane / K=1): the pre-filter argmin is
        returned, there being nothing to certify against.

    Returns (winner_row_or_None, meta) where meta carries the ranking and
    the ``CertifiedVerdict`` audit trail (``verdict`` key: the
    certified_ab verdict string, ``decision`` key: what was done).
    """
    usable = [s for s in scored if s.get("screen_exact_ce") is not None]
    meta: dict = {"eps": eps, "delta": delta, "eval_n": eval_n}
    if not usable:
        meta["decision"] = "NO_SCORED_CANDIDATE"
        return None, meta
    ranked = sorted(usable, key=lambda s: s["screen_exact_ce"])
    meta["prefilter_rank"] = [s["label"] for s in ranked]
    if incumbent_ce is not None:
        incumbent = {
            "label": "incumbent", "screen_exact_ce": float(incumbent_ce),
        }
        inc_n = incumbent_n
    else:
        incumbent = next(
            (s for s in usable if incumbent_label is not None
             and s["label"] == incumbent_label),
            None,
        )
        inc_n = eval_n
    best = ranked[0]
    if incumbent is None or best["label"] == incumbent["label"]:
        meta["decision"] = "PREFILTER_ARGMIN"  # nothing to certify against
        meta["verdict"] = "NO_INCUMBENT"
        return best, meta
    cert: CertifiedVerdict = certified_ab(
        incumbent["screen_exact_ce"], inc_n,
        best["screen_exact_ce"], eval_n,
        eps=eps, delta=delta,
    )
    meta["verdict"] = cert.verdict
    meta["reason"] = cert.reason
    meta["n_required"] = cert.n_required
    if cert.accepted:
        meta["decision"] = "CERTIFIED_WIN"
        return best, meta
    # REJECT and UNDECIDED both keep the incumbent: certified-worse is a
    # hard no, and an undecided margin must not leak the forbidden
    # selection_hurts path. The fallback is documented above.
    meta["decision"] = (
        "CERTIFIED_REJECT_KEEP_INCUMBENT" if cert.rejected
        else "UNDECIDED_NOT_WORSE_FALLBACK"
    )
    return incumbent, meta
