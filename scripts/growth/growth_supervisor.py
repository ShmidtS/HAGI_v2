#!/usr/bin/env python
"""Autonomous growth supervisor: train -> merge -> evaluate -> decide.

What this replaces
------------------
The growth loop was driven by hand: run ``train.py`` for each expert, then a
joint config, then ``eval_holdout.py``, then decide. ``scripts/train_e6_experts.sh``
automated only the first step and aborted the whole sequence on any failure.

What this adds
--------------
A lane is one candidate model. The supervisor takes a lane spec (expert configs
plus a joint config), runs the phases, and records a verdict against an
incumbent in a JSONL ledger. Every phase is idempotent: an existing checkpoint
is resumed, never recomputed, so a kill at any point costs one interval rather
than the lane.

Why a supervisor and not ``src/hagi/orchestrator/``
---------------------------------------------------
That package was written for a toy scale -- hidden 8, one layer, CPU, a
Banking77 packed-CE metric, exactly three children with byte-identical configs,
and a lease/transactional owner whose 300 s TTL cannot cover a multi-hour GPU
job. Scaling it means rewriting its metric and geometry contracts. This driver
reuses the pieces that are already correct (``train.py``, ``merge_experts`` via
the joint config, ``eval_holdout.score``) and leaves the orchestrator dark.

Honesty about the decision rule
-------------------------------
The verdict is deliberately weak: the candidate must not be worse than the
incumbent by more than a fixed margin on the mean exact CE, and must not
regress on any single domain by more than a per-domain margin. Held-out exact
CE at this scale has a noise band of roughly 0.2 nats, so a tighter rule would
reject on noise rather than on evidence. This loop therefore grows the model
when growth is clear and refuses to claim growth when it is not -- it is not a
search, and it does not pretend a single lane proves recursive improvement.

Usage::

    python scripts/growth_supervisor.py --plan configs/growth_n6.yaml \\
        --device cuda --max-lanes 4
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '../../..'))
for _p in (_HERE, _REPO, os.path.join(_REPO, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# This file lives in scripts/growth/, so the repository root is three levels
# up. Using scripts/ made every --plan, checkpoint_dir and cwd resolve to a
# path that does not exist.
ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "reports" / "growth_ledger.jsonl"
LOG = logging.getLogger("growth")

# A held-out exact CE at this scale moves by ~0.2 nats between seeds, so a
# candidate is only rejected when it is clearly worse, not when it is merely
# different. Set from the spec; these are the fallbacks.
DEFAULT_TOTAL_MARGIN = 0.05
DEFAULT_DOMAIN_MARGIN = 0.25

# --- R93: anytime-valid confidence budget for the acceptance gate --------
#
# The gate above runs ONCE PER LANE, and a plan can have many lanes. A
# fixed ``domain_margin`` per comparison spends ``delta`` every time, so
# over n lanes the nominal budget is ``n * delta`` -- which is what made
# E3 in the working notes: 400 checks at delta=0.05 need a budget of 20,
# i.e. 400x more than declared.
#
# R93 (Ville's inequality) gives the schedule that does not. Spending
#
#     delta_t = delta_0 * rho^t,   delta_0 / (1 - rho) <= delta
#
# bounds the probability that ANY certificate up to T ever fails, from ONE
# global delta, uniformly in T. Since the lane index is exactly the
# comparison index t, the same formula applies verbatim here.
#
# The consequence for the gate is concrete: the FIRST lane is judged with
# the full budget and a wide margin, and later lanes are judged tighter,
# because by then the incumbent is well established and evidence has
# accumulated. Under a fixed margin the early lanes -- the ones that
# actually decide whether growth is real -- got no more statistical power
# than the fiftieth.
DEFAULT_GATE_DELTA = 0.05
DEFAULT_GATE_DECAY = 0.5

# --- §0/§9: the certified controller this driver delegates its decisions to.
#
# The verdict below used to be "not worse than the incumbent by a fixed
# margin" and candidates were screened by standalone CE. ALGORITHMS.md §1
# forbids the latter (``selection_hurts``: standalone CE ranking can
# strictly degrade the ensemble) and §9 replaces the former with the
# certified 2-eps Hoeffding rule. The driver keeps its working parts
# (lane loading, subprocess driving, resume, Windows-safe wait) and
# routes every DECISION through ``hagi.train.certified_controller``.
from hagi.train import certified_controller as cc

# Effective sample size of one held-out eval when the report does not
# record it (eval_holdout.py is driven with --batches 200).
DEFAULT_EVAL_N = 200


def anytime_margin(base: float, lane: int, total_delta: float,
                   decay: float) -> float:
    """The acceptance margin for lane ``t`` under the R93 schedule.

    Args:
        base: the margin used at ``t = 0`` (the widest, from the CLI).
        lane: the zero-based comparison index ``t``.
        total_delta: the global budget for the whole horizon.
        decay: ``rho`` in ``(0, 1)``.

    Returns:
        The margin for this lane, scaled by ``(1 - rho) rho^t``.

    Raises:
        ValueError: on a non-positive base or a ``decay`` outside
            ``(0, 1)``, rather than silently reverting to the fixed
            margin and hiding a misconfigured budget.
    """
    if base <= 0.0:
        raise ValueError("base margin must be positive")
    if not 0.0 < decay < 1.0:
        raise ValueError("decay (rho) must lie in (0, 1)")
    if total_delta <= 0.0:
        raise ValueError("total_delta must be positive")
    return base * (1.0 - decay) * (decay ** lane)


@dataclass
class Lane:
    """One candidate: N experts merged into one joint model.

    ``lane_type == "distill"`` marks a §17 distill lane: no experts are
    trained, the joint config named by ``joint_config`` is the PARENT
    whose distill config is derived by :func:`make_distill_config`.
    """

    name: str
    expert_configs: list[str]
    joint_config: str
    lane_type: str = "growth"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verdict:
    accepted: bool
    reason: str
    incumbent_mean: float | None
    candidate_mean: float | None
    deltas: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "incumbent_mean": self.incumbent_mean,
            "candidate_mean": self.candidate_mean,
            "deltas": self.deltas,
        }


def load_plan(path: Path) -> list[Lane]:
    """Read the lane list.

    The plan is a plain list so that adding a generation is a data edit, not a
    code edit. ``incumbent`` names the config whose held-out JSON the first
    lane is measured against; later lanes chain off the previous accepted one.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Accept both a bare list and a mapping with a `lanes:` key, so the plan
    # can carry top-level comments-as-data without changing the loader.
    if isinstance(raw, dict):
        raw = raw.get("lanes", raw.get("plan"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty list of lanes")
    lanes: list[Lane] = []
    for i, item in enumerate(raw):
        for key in ("name", "joint"):
            if key not in item:
                raise ValueError(f"{path}: lane {i} is missing '{key}'")
        lane_type = str(item.get("type", "growth"))
        if lane_type not in ("growth", "distill"):
            raise ValueError(f"{path}: lane {i} has unknown type {lane_type!r}")
        if lane_type == "growth" and "experts" not in item:
            raise ValueError(f"{path}: lane {i} is missing 'experts'")
        lanes.append(
            Lane(
                name=str(item["name"]),
                expert_configs=[str(p) for p in item.get("experts", [])],
                joint_config=str(item["joint"]),
                lane_type=lane_type,
                meta={
                    k: v
                    for k, v in item.items()
                    if k not in ("name", "experts", "joint", "type")
                },
            )
        )
    return lanes


def run(cmd: list[str], log_path: Path, timeout: int) -> int:
    """Run a subprocess, streaming into a log. Returns its exit code.

    The log is truncated per attempt, not appended, so a resumed lane's log
    shows the attempt that actually produced the checkpoint on disk.
    """
    LOG.info("run: %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
        )
        try:
            # Poll rather than ``proc.wait()``. On Windows a killed child
            # can leave ``wait()`` blocked on a handle that never closes, and
            # the supervisor was observed stuck in State: S for minutes after
            # the training process was long gone -- which silently ended the
            # autonomous cycle with nothing in the log saying so. A poll loop
            # with a bounded sleep cannot wedge, and the timeout is enforced
            # here rather than by an unbounded wait.
            deadline = time.monotonic() + timeout
            while True:
                try:
                    return proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    if time.monotonic() > deadline:
                        proc.kill()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            LOG.error("child did not reap after kill")
                        LOG.error("timeout after %ss: %s", timeout, " ".join(cmd))
                        return 124
        except Exception as exc:                      # pragma: no cover
            proc.kill()
            LOG.error("run failed: %s (%s)", " ".join(cmd), exc)
            return 125


def latest_checkpoint(directory: Path) -> Path | None:
    """Newest complete checkpoint in a directory, or None."""
    if not directory.is_dir():
        return None
    steps = sorted(directory.glob("step-*.pt"))
    return steps[-1] if steps else None


def log_tail(log_path: Path, lines: int = 3) -> str:
    """The last few lines of a training log, or "" if unreadable."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return "\n".join(text.rstrip().splitlines()[-lines:])


# Markers a real failure leaves behind. Their absence is what tells a
# killed process from a crash: a crash prints one of these, a kill just
# stops.
_FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "Error",
    "error:",
    "CUDA out of memory",
    "HIP error",
    "AssertionError",
    "RuntimeError",
    "ValueError",
)


_CE_LINE_RE = re.compile(r"\|\s*ce=(nan|inf|-inf|[\d.eE+-]+)", re.I)


def parse_ce_series(log_path: Path) -> list[float]:
    """The logged per-step cross-entropy series of a training log.

    The same parsing :func:`converged` does, kept as its own function so
    the ignition gate can read the trajectory without re-implementing
    the regex (and without duplicating the nan/inf handling: those
    lines are skipped here, not read as numbers).
    """
    if not log_path.exists():
        return []
    ce: list[float] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _CE_LINE_RE.search(line)
                if m:
                    try:
                        v = float(m.group(1))
                    except ValueError:
                        continue
                    # nan/inf carry no trajectory information; the
                    # ignition gate needs finite steps only.
                    if math.isfinite(v):
                        ce.append(v)
    except OSError:
        return []
    return ce


def looks_like_exception(tail: str) -> bool:
    """Does this log tail show a real failure rather than a silent stop?

    Used to tell the two apart in ``train_phase``: a child killed by
    another process leaves no traceback and exits non-zero, which looks
    identical to a trainer crash unless you check for the marker.
    """
    return any(m in tail for m in _FAILURE_MARKERS)


def checkpoint_dir_of(config: Path) -> Path:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    return ROOT / str(cfg["train"]["checkpoint_dir"])


def converged(
    log_path: Path,
    ceiling: float = 6.0,
    max_regression: float = 1.0,
) -> tuple[bool, str]:
    """Did the run finish converged, or did it degrade?

    A run can exit 0 having diverged: the gen-4 sib3 log shows CE climbing
    4.07 -> 5.61 -> 12.19 -> 29.06 over steps 1230..1330 while training
    continued to completion. Accepting that checkpoint would poison the
    merge, because a diverged expert contributes its divergence to all
    three downstream branches. Exit status alone cannot see this; only the
    loss curve can.

    Two tests, because the two observed failures look different:

    ``ceiling`` on the FINAL ce
        catches the outright blow-up (sib3, final ce 46).

    ``max_regression`` on the TAIL mean against the BEST window mean
        catches slow degradation, which a final-value ceiling misses: gen-4
        sib2 ended at ce 5.15 -- under the 6.0 ceiling -- after sliding from
        a best-window mean of 3.56 to a tail mean of 5.49. That is a
        damaged expert with no single alarming number.

    A run with too few lines to form both windows is judged on the
    ceiling alone: freshly resumed or very short runs legitimately log
    little, and refusing them would break the resume path.
    """
    if not log_path.exists():
        return True, "no log"
    ce: list[float] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                # The value may be a number OR nan/inf: a diverged run
                # reports "ce=nan", which must not be read as "no number
                # here" and therefore as convergence.
                m = _CE_LINE_RE.search(line)
                if m:
                    try:
                        ce.append(float(m.group(1)))
                    except ValueError:
                        continue
    except OSError as exc:
        LOG.warning("cannot read %s: %s", log_path, exc)
        return True, "unreadable"
    if not ce:
        return True, "no ce lines"
    last = ce[-1]
    if last != last:  # NaN
        return False, "final ce is nan"
    if last in (float("inf"), float("-inf")):
        return False, f"final ce={last}"
    if last > ceiling:
        return False, f"final ce={last:.4f} exceeds {ceiling}"

    # Degradation test over a sliding window of logged steps.
    window = max(3, min(10, len(ce) // 4))
    if len(ce) >= 2 * window:
        best = min(
            sum(ce[i : i + window]) / window
            for i in range(len(ce) - window + 1)
        )
        tail = sum(ce[-window:]) / window
        if tail - best > max_regression:
            return False, (
                f"tail mean {tail:.4f} regressed {tail - best:.4f} "
                f"beyond {max_regression} from the best window {best:.4f}"
            )
        return True, f"final ce={last:.4f}, tail {tail:.4f} vs best {best:.4f}"
    return True, f"final ce={last:.4f}"


def train_phase(
    config: Path,
    log_path: Path,
    device: str,
    attempts: int,
    max_ce: float = 6.0,
) -> Path | None:
    """Train (or resume) one config. Returns the final checkpoint, or None.

    Each attempt resumes from whatever is on disk, so a crash costs at most one
    checkpoint interval instead of the whole run. This is the crash-recovery
    mechanism: the shell chain this replaces aborted the entire expert sequence
    on the first failure.
    """
    out = checkpoint_dir_of(config)
    for attempt in range(1, attempts + 1):
        existing = latest_checkpoint(out)
        cmd = [
            sys.executable, "scripts/train.py",
            "--config", str(config),
            "--device", device,
        ]
        if existing is not None:
            # Resume whenever something is on disk, not only on a retry. Without
            # this the supervisor restarted a fully trained run from step 0,
            # spending 45 minutes to reproduce a checkpoint it already had.
            LOG.info("resume %s from %s", out, existing.name)
            cmd += ["--resume", str(existing)]
        code = run(cmd, log_path, timeout=60 * 60 * 24)
        ckpt = latest_checkpoint(out)
        if code == 0 and ckpt is not None:
            ok, why = converged(log_path, max_ce)
            if ok:
                LOG.info("train ok: %s -> %s (%s)", config.name, ckpt.name, why)
                return ckpt
            # The run exited 0 but diverged. Do NOT resume it: the next
            # attempt would continue from the diverged checkpoint. Refuse the
            # lane instead -- a diverged expert must never reach the merge.
            LOG.error(
                "diverged despite exit 0: %s -> %s (%s); refusing the checkpoint",
                config.name, ckpt.name, why,
            )
            return None
        LOG.warning("train attempt %d/%d failed (exit %s)", attempt, attempts, code)
        # A silent death is the common case on this hardware and it is NOT a
        # training problem, so it must not be reported as one. Two supervisors
        # racing for the same GPU produced exactly this: the child was killed
        # mid-run, the log simply stops, and the retry loop cannot tell that
        # apart from a crash in the trainer. Naming the distinction here is
        # what stops the next occurrence from being debugged from scratch.
        tail = log_tail(log_path, lines=3)
        if code != 0 and not looks_like_exception(tail):
            LOG.warning(
                "attempt %d exited %s with no traceback in %s -- the process "
                "was killed rather than failing. Most likely another training "
                "process holds the GPU; check for a duplicate supervisor.",
                attempt, code, log_path.name,
            )
        if ckpt is None:
            LOG.error("no checkpoint on disk after failure; giving up on %s", config)
            return None
    # Every attempt is spent. The checkpoint on disk is whatever the LAST
    # attempt left behind, which on gen-4 was a diverged one -- so it must be
    # judged, not returned. Returning it unconditionally is how a destroyed
    # expert reaches the merge.
    ok, why = converged(log_path, max_ce)
    if not ok:
        LOG.error("all attempts spent and the run diverged: %s (%s)", config.name, why)
        return None
    return latest_checkpoint(out)


def evaluate(config: Path, checkpoint: Path, out_json: Path, device: str) -> dict | None:
    """Score a checkpoint on the held-out tails. Returns the parsed report."""
    code = run(
        [
            sys.executable, "scripts/eval_holdout.py",
            "--config", str(config),
            "--resume", str(checkpoint),
            "--batches", "200",
            "--device", device,
            "--out", str(out_json),
        ],
        out_json.with_suffix(".log"),
        timeout=60 * 60 * 4,
    )
    if code != 0 or not out_json.is_file():
        LOG.error("eval failed (exit %s) for %s", code, config.name)
        return None
    return json.loads(out_json.read_text(encoding="utf-8"))


def mean_ce(report: dict) -> float | None:
    """Mean exact CE over the domains a report actually scored.

    A domain that is missing or errored is excluded rather than treated as
    zero, and if nothing scored, the lane has no number to judge on.
    """
    values = [
        float(d["exact_ce"])
        for d in report.get("domains", {}).values()
        if isinstance(d, dict) and isinstance(d.get("exact_ce"), (int, float))
    ]
    return sum(values) / len(values) if values else None


def report_n(report: dict | None) -> int:
    """Effective sample size of an eval report, for the §9 premise.

    ``eval_holdout.py`` scores a fixed number of batches; the report may
    carry it (``n_eval`` or ``batches``). The default matches the
    supervisor's own ``--batches 200`` invocation.
    """
    if not isinstance(report, dict):
        return DEFAULT_EVAL_N
    for key in ("n_eval", "batches"):
        v = report.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return DEFAULT_EVAL_N


def decide(
    candidate: dict,
    incumbent: dict | None,
    total_margin: float,
    domain_margin: float,
    lane: int = 0,
    total_delta: float = DEFAULT_GATE_DELTA,
    decay: float = DEFAULT_GATE_DECAY,
) -> Verdict:
    """Decide the lane via the §9 certified A/B rule, legacy rule as fallback.

    The DECISION routes through ``certified_controller.certified_ab``
    (ALGORITHMS.md §9): accept the candidate iff
    ``CE_incumbent - CE_candidate > 2*eps`` with ``n >= log(2/delta_t)/
    (2 eps^2)``, where ``eps`` is half the lane's anytime margin (so the
    certified boundary coincides with the historical one) and ``delta_t``
    is this lane's spend from the R93 schedule.

    A single 200-batch eval usually sits below the Hoeffding threshold
    at these eps, so the certified verdict is often UNDECIDED -- the
    comparison lacks the evidence §9 demands. In that case the verdict
    falls back to the historical not-worse-on-margin rule (including the
    per-domain regression guard), which is strictly weaker and never
    contradicts a certified verdict. A certified REJECT, however, is
    final: the fallback cannot rescue a certified regression.

    ``lane`` indexes the comparison within the run and tightens both margins
    via the R93 anytime schedule (:func:`anytime_margin`).
    """
    total_margin = anytime_margin(total_margin, lane, total_delta, decay)
    domain_margin = anytime_margin(domain_margin, lane, total_delta, decay)
    cand_mean = mean_ce(candidate)
    if cand_mean is None:
        return Verdict(False, "candidate produced no scored domain", None, None)
    if incumbent is None:
        return Verdict(
            True, "first scored lane: no incumbent to beat",
            None, cand_mean,
        )
    inc_mean = mean_ce(incumbent)
    if inc_mean is None:
        return Verdict(True, "incumbent has no scored domain", inc_mean, cand_mean)

    deltas: dict[str, float] = {}
    inc_domains = incumbent.get("domains", {})
    for name, dom in candidate.get("domains", {}).items():
        if not isinstance(dom, dict) or not isinstance(dom.get("exact_ce"), (int, float)):
            continue
        other = inc_domains.get(name)
        if isinstance(other, dict) and isinstance(other.get("exact_ce"), (int, float)):
            deltas[name] = float(dom["exact_ce"]) - float(other["exact_ce"])

    # --- §9 decision path -------------------------------------------------
    # eps = margin/2 makes the certified 2*eps band coincide with the
    # historical margin, so certification can only REJECT or CONFIRM,
    # never widen, the old acceptance boundary. When the fixed-eps rule
    # is UNDECIDED purely for sample-size, try the ADAPTIVE eps the
    # sample actually supports (n_certifiable = the largest eps whose
    # Hoeffding n fits the data): a large measured delta with small n
    # can still certify at a wider band (the anytime spirit of R93 —
    # the guarantee degrades gracefully, it does not vanish).
    delta_t = total_delta * (1.0 - decay) * (decay ** lane)
    cert = cc.certified_ab(
        inc_mean, report_n(incumbent), cand_mean, report_n(candidate),
        eps=total_margin / 2.0, delta=delta_t,
    )
    n_avail = min(report_n(incumbent), report_n(candidate))
    if not cert.accepted and not cert.rejected and n_avail > 0:
        import math as _math
        d = inc_mean - cand_mean
        # largest eps certifiable at n_avail: eps_c = sqrt(log(2/delta)/(2n))
        eps_c = (_math.sqrt(_math.log(2.0 / delta_t) / (2.0 * n_avail)) * 1.02
                if delta_t < 1 else float("inf"))
        if eps_c < _math.inf and abs(d) > 2.0 * eps_c:
            cert = cc.certified_ab(
                inc_mean, report_n(incumbent), cand_mean, report_n(candidate),
                eps=eps_c, delta=delta_t,
            )
    if cert.accepted:
        gain = inc_mean - cand_mean
        return Verdict(
            True, f"certified accept (§9): {cert.reason}"
                  + data_axis_advice(gain),
            inc_mean, cand_mean, deltas,
        )
    if cert.rejected:
        return Verdict(
            False, f"certified reject (§9): {cert.reason}",
            inc_mean, cand_mean, deltas,
        )
    # UNDECIDED: no certified evidence either way. THEORY (audit P1-1,
    # §9): an UNDECIDED comparison must NOT accept through the legacy
    # not-worse rule — that leaks the forbidden decision path. The
    # conservative theory-correct default: REJECT (the incumbent
    # stands); the legacy rule survives only as the *reason text* for
    # the retro-log.
    regressed = {k: v for k, v in deltas.items() if v > domain_margin}
    why = (
        f"domain regression: "
        + ", ".join(f"{k} {v:+.4f}" for k, v in sorted(regressed.items()))
        if regressed
        else f"mean CE {cand_mean - inc_mean:+.4f} (uncertified at n={cert.n}, "
             f"need {cert.n_required}) — incumbent stands"
    )
    return Verdict(False, why, inc_mean, cand_mean, deltas)

# --- the data axis: what to do when growth stalls (data_axis.py, §4) ----

# Below this gain the merge axis is dead -- the candidate is not worse,
# but it is not better either, so the same action will not help again.
# The value is expressed in NATS and derived from the margins above: a
# gain smaller than the gate's own resolution is indistinguishable from
# zero at this noise level (~0.2 nats on held-out exact CE).
STALL_GAIN = DEFAULT_DOMAIN_MARGIN


def data_axis_advice(gain: float, stall_threshold: float = STALL_GAIN) -> str:
    """The §4 response to a stalled merge axis, as a ledger annotation.

    ``liveness_data_axis``: the loop cannot freeze while EITHER axis is
    alive, and when the certified gain goes to zero the honest response
    is not to retry the same action but to inject fresh INDEPENDENT data,
    so that the diversity floor ``D_T > 0`` becomes strictly positive
    again.

    This is advice, not an action: the supervisor records what the state
    calls for, and a human or the next plan acts on it. Recording it
    matters because "the run kept going but nothing improved" is
    otherwise indistinguishable in the ledger from "the run is working".

    The independence requirement is the point. A same-domain corpus does
    not satisfy it -- the round-66/67 measurement showed sibling residual
    CE at ln V, i.e. pure noise, from more of the same distribution. The
    criterion is on ``inj - xi``, not on raw volume.

    Args:
        gain: ``incumbent_mean - candidate_mean``, positive when better.
        stall_threshold: the gain below which the merge axis counts as
            stalled.

    Returns:
        A short annotation, empty when growth is real.
    """
    if gain > stall_threshold:
        return ""
    return (
        f" [stalled: gain {gain:+.4f} <= {stall_threshold:.4f}; per "
        f"liveness_data_axis, retrying the same action cannot help -- "
        f"inject data from a distribution the model has NOT seen]"
    )


# --- R104: the frontier ceiling, as a stopping detector ------------------

def frontier_ceiling_advice(
    gain: float,
    frontier: float,
    alpha: float,
    gamma: float,
    stall_threshold: float = STALL_GAIN,
) -> str:
    """``bounded_frontier_no_sustained_growth`` applied to a verdict.

    R104's converse: with the gain an exact harvest of usable
    disagreement and a BOUNDED frontier ``D_t <= Dbar``, capability is
    capped forever at ``γ·Dbar/α``. So a growth loop watching a flat
    frontier is not merely not improving -- it is provably unable to,
    and no further merge of the same experts can raise the ceiling.

    That makes this a detector rather than a hope. The supervisor knows
    the measured gain of each comparison; when it also knows the frontier
    (the disagreement between the experts being merged) it can say
    whether the loop is anywhere near its ceiling.

    The frontier is passed in rather than inferred: this supervisor
    measures CE, and pretending CE spread is disagreement would be a
    category error. A caller with no frontier measurement gets no
    ceiling claim, which is the honest default.

    Args:
        gain: the measured gain of this comparison.
        frontier: the measured usable disagreement ``D_t``.
        alpha: the gate parameter.
        gamma: the harvest rate.
        stall_threshold: the gain below which the merge axis is stalled.

    Returns:
        A short annotation, empty when there is nothing to add.
    """
    if frontier <= 0.0 or alpha <= 0.0 or gamma <= 0.0:
        return ""
    from hagi.train.gain_renewal import sustained_growth_possible

    ceiling = sustained_growth_possible(alpha, gamma, frontier)
    if gain > stall_threshold:
        # Growth is real, but say whether the ceiling is close.
        return (f" [frontier {frontier:.4f} caps capability at "
                f"{ceiling:.4f}]")
    return (
        f" [stalled AND frontier-capped: D={frontier:.4f} implies "
        f"C <= {ceiling:.4f} for every future generation "
        f"(bounded_frontier_no_sustained_growth) -- the loop is at its "
        f"ceiling, so another merge cannot help; widen the frontier "
        f"(new domain, synthetic data, discovery) instead]"
    )


def append_ledger(row: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- §23: theory-phase annotations (log-only) ---------------------------

def log_phase(phase: str, lane: str, detail: str = "") -> None:
    """Emit one canonical §23 phase line so the log/ledger record which
    theory phase each stage of the cycle is in. Log-only by design."""
    line = cc.phase_log_line(phase, lane, detail)
    LOG.info("%s", line)
    return line


def theory_stop_check(lane: Lane, verdict: Verdict | None) -> str:
    """§0 step 6 as an annotation: STOP iff consensus AND inj <= xi [R80].

    The supervisor does not stop autonomously on this (the plan bounds the
    run), but the ledger records whether the two-axis stop condition is
    met: ``consensus`` is read from the verdict (a stalled or uncertified
    gain means the merge axis has converged to noise), and ``inj``/``xi``
    come from lane meta when a plan supplies them, defaulting to the
    conservative ``inj = 0`` (no fresh-data injection measured).
    """
    consensus = verdict is not None and (
        not verdict.accepted or "stalled" in verdict.reason
    )
    inj = float(lane.meta.get("inj", 0.0))
    xi = float(lane.meta.get("xi", 0.0))
    stop = cc.stop_condition(consensus, inj, xi)
    if stop:
        return ("stop_condition MET (consensus + inj<=xi, R80): an honest "
                "stop -- retrying the same action cannot help; inject data")
    return ""


def distill_leak_check(lane: Lane) -> str | None:
    """§17 distill_leak_gate as an annotation, when the lane distills.

    A lane only carries distillation measurements if its plan meta has
    ``distill_cycle_gain`` (c_k) and ``distill_slack`` (delta_k); without
    both there is nothing to gate and None is returned.
    """
    c = lane.meta.get("distill_cycle_gain")
    d = lane.meta.get("distill_slack")
    if c is None or d is None:
        return None
    continue_ = cc.leak_gate(float(c), float(d))
    log_phase(cc.GrowthPhase.DISTILL, lane.name,
              f"c_k={c} delta_k={d}")
    if continue_:
        return "distill_leak_gate: c_k - delta_k still pays; recursion may continue"
    return ("distill_leak_gate: delta >= c -- STOP the recursion (R134); "
            "do not force further distillation cycles")


# --- §10/§13: ignition / saturation / takeoff-window gates (advisory) ---
#
# These call the ALREADY-PORTED theory modules (saturation.py,
# ratio_takeoff.py, takeoff_window.py) on MEASURED inputs. The verdict
# stays non-destructive: gates advise, ``certified_ab`` decides (the
# §9 gate in decide()).
#
# Measurement model (documented, honest):
# * capability proxy  C_t = exp(-ce_t)  from the joint training log --
#   CE drop IS capability gain on the same scale, and exp keeps C
#   strictly positive as the ratio dynamics require;
# * data-field proxy  D_t = (C_{t+1} - C_t) / gamma  -- inverted from
#   the §10 dynamics C' = C + gamma*D, so D is the usable frontier
#   the step actually harvested;
# * rho_hat / beta_hat -- least-squares fit of D' = rho*D + beta*C
#   (xi = 0 form) on the trajectory, UNLESS the lane meta supplies
#   measured values (``rho``, ``beta``), which always win over the fit.
#
# Required lane-meta inputs: ``gamma`` and ``k`` (cone parameters).
# Optional: ``rho``, ``beta`` (overrides), ``xi`` (friction, default 0),
# ``alpha`` (takeoff-window gate), ``Cstar``/``sigma`` (saturation PL
# window). Anything missing that a gate needs -> that gate is skipped;
# if the REQUIRED pair is missing the whole check logs
# ``phase=IGNITE inputs_missing`` and returns None (no crash).


def _fit_frontier_dynamics(
    capability: list[float], gamma: float
) -> tuple[float, float] | None:
    """Least-squares fit of ``D' = rho*D + beta*C`` on the trajectory.

    Returns ``(rho_hat, beta_hat)`` or None when the trajectory is too
    short (needs >= 3 points, i.e. two transitions) or degenerate (all
    D equal -- nothing to regress on).
    """
    d = [(capability[t + 1] - capability[t]) / gamma
         for t in range(len(capability) - 1)]
    if len(d) < 2:
        return None
    # regress d[1:] on (d[:-1], capability[1:-1])
    xs = list(zip(d[:-1], capability[1:-1]))
    y = d[1:]
    n = float(len(y))
    sx = [sum(r[i] for r in xs) for i in (0, 1)]
    sy = sum(y)
    sxx = sum(r[0] * r[0] for r in xs)
    sxy = sum(r[0] * v for r, v in zip(xs, y))
    denom = n * sxx - sx[0] * sx[0]
    if abs(denom) < 1e-12:
        return None
    rho_hat = (n * sxy - sx[0] * sy) / denom
    # beta from the residual mean: y - rho*x0 = beta*C
    beta_hat = (sy - rho_hat * sx[0]) / sx[1] if sx[1] != 0.0 else 0.0
    return rho_hat, beta_hat


def ignition_gate_check(
    lane: Lane,
    joint_log: Path,
) -> dict[str, Any] | None:
    """§10/§13 gates on the measured joint trajectory (advisory only).

    Calls ``ratio_takeoff.bifurcation_verdict``,
    ``saturation.lifecycle_verdict`` and ``takeoff_window.takeoff_window``
    on inputs measured from the training log, and returns the advice as
    a dict for the ledger. ``None` + a logged ``phase=IGNITE
    inputs_missing`` line means the inputs are not available and the
    gates are skipped -- never a crash, never a verdict.
    """
    from hagi.train.ratio_takeoff import bifurcation_verdict
    from hagi.train.saturation import lifecycle_verdict
    from hagi.train.takeoff_window import takeoff_window

    gamma = lane.meta.get("gamma")
    k = lane.meta.get("k")
    ce = parse_ce_series(joint_log)
    if gamma is None or k is None or len(ce) < 3:
        log_phase(cc.GrowthPhase.IGNITE, lane.name, "inputs_missing")
        return None
    gamma = float(gamma)
    k = float(k)
    xi = float(lane.meta.get("xi", 0.0))

    capability = [math.exp(-x) for x in ce if math.isfinite(x)]
    if len(capability) < 3:
        log_phase(cc.GrowthPhase.IGNITE, lane.name, "inputs_missing")
        return None

    fit = _fit_frontier_dynamics(capability, gamma)
    if fit is None:
        log_phase(cc.GrowthPhase.IGNITE, lane.name, "inputs_missing")
        return None
    rho_hat, beta_hat = fit
    if "rho" in lane.meta:
        rho_hat = float(lane.meta["rho"])
    if "beta" in lane.meta:
        beta_hat = float(lane.meta["beta"])

    out: dict[str, Any] = {
        "rho_hat": round(rho_hat, 6),
        "beta_hat": round(beta_hat, 6),
        "C_final": round(capability[-1], 6),
    }

    bif = bifurcation_verdict(beta_hat, rho_hat, gamma, k,
                              C=capability[-1], xi=xi)
    out["bifurcation"] = bif.value
    advice = {
        "grow": "cone holds: continue the cycle (one-sided (1+gamma*k)^T "
                "certificate, R124)",
        "decay": "beta=0 and rho<1: frontier decays geometrically -- do NOT "
                 "cycle; wait for data injection (frontier_decay_no_growth)",
        "inject": "production below the ignition threshold: grow the data "
                  "axis (fresh independent corpora), not more cycles",
    }[bif.value]
    out["advice"] = advice

    cstar = lane.meta.get("Cstar")
    sigma = lane.meta.get("sigma")
    if cstar is not None and sigma is not None:
        lc = lifecycle_verdict(beta_hat, rho_hat, gamma, k,
                               capability[-1],
                               Cstar=float(cstar), sigma=float(sigma),
                               xi=xi)
        out["lifecycle"] = lc.value

    alpha = lane.meta.get("alpha")
    if alpha is not None:
        gains = [capability[t + 1] - capability[t]
                 for t in range(len(capability) - 1)]
        mean_gain = sum(gains) / len(gains)
        if mean_gain > 0.0:
            w = takeoff_window(float(alpha), capability[0], mean_gain)
            out["takeoff_window"] = {
                "max_successes": round(w.max_successes, 4),
                "certified_factor_bound": round(w.certified_factor_bound, 4),
            }
    log_phase(cc.GrowthPhase.IGNITE, lane.name,
              f"{bif.value} beta_hat={beta_hat:.4g} rho_hat={rho_hat:.4g}")
    return out


# --- §17 / R134: the distill cycle as a supervisor lane type --------------


def make_distill_config(parent_config: Path, out_path: Path) -> Path:
    """Derive a distill-lane config from a joint (parent) config.

    Mirrors how ``make_generation.py`` derives the distill stage, KISS:
    copy the parent YAML, then

    * ``train.checkpoint_dir``  -> ``<parent_dir>_distill``
    * ``train.init_from``       -> ``<parent_dir>/best.pt`` (the joint)
    * when the config lists ``distill.teachers``: enable the reverse-
      recursion channel (``merge.distill: true``, first listed teacher
      in ``merge.distill_teacher`` -- the training loop consumes exactly
      one) and set the disagreement token-selection quantile
      (``merge.distill_disagreement_quantile: 0.95``, Bregman-slice,
      ``disagreement_distill``).

    Pure data derivation: writing the file does NOT launch training.
    """
    raw = yaml.safe_load(parent_config.read_text(encoding="utf-8"))
    ckpt_dir = str(raw["train"]["checkpoint_dir"])
    raw["train"]["checkpoint_dir"] = f"{ckpt_dir}_distill"
    raw["train"]["init_from"] = f"{ckpt_dir}/best.pt"
    merge = raw.setdefault("merge", {})
    teachers = (raw.pop("distill", None) or {}).get("teachers")
    if teachers:
        merge["distill"] = True
        merge["distill_teacher"] = str(teachers[0])
        merge["distill_disagreement_quantile"] = 0.95
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out_path


def distill_leak_from_evals(
    lane: Lane,
    student_report: dict | None,
    teacher_report: dict | None,
    incumbent_report: dict | None,
) -> dict[str, Any] | None:
    """§17 leak gate on MEASURED evals: c_k vs delta_k.

    ``delta_k`` (the single-cycle bridge slack) is the student's CE above
    its teacher's; ``c_k`` (the cycle gain) comes from lane meta when a
    plan supplies it, else from the incumbent comparison
    ``incumbent_ce - teacher_ce``. Missing inputs -> None (logged as
    inputs_missing by the caller), never a crash.
    """
    student_ce = mean_ce(student_report) if student_report else None
    teacher_ce = mean_ce(teacher_report) if teacher_report else None
    c = lane.meta.get("distill_cycle_gain")
    if c is None and incumbent_report is not None:
        inc_ce = mean_ce(incumbent_report)
        if inc_ce is not None and teacher_ce is not None:
            c = inc_ce - teacher_ce
    if student_ce is None or teacher_ce is None or c is None:
        return None
    delta = student_ce - teacher_ce
    continue_ = cc.leak_gate(float(c), delta)
    return {
        "c_k": round(float(c), 6),
        "delta_k": round(delta, 6),
        "continue_recursion": continue_,
        "note": (
            "distill_leak_gate: c_k - delta_k still pays; recursion may continue"
            if continue_ else
            "distill_leak_gate: delta_k >= c_k -- STOP the recursion (R134); "
            "do not force further distillation cycles"
        ),
    }


def prune(keep: int, rows: list[Path]) -> None:
    """Keep the newest ``keep`` checkpoints per directory, delete the rest.

    A 241M-param joint checkpoint plus optimizer state is on the order of
    gigabytes, so an unattended loop has to bound its own disk use rather than
    discover it in the morning.
    """
    for directory in {p.parent for p in rows}:
        files = sorted(directory.glob("step-*.pt"))
        for old in files[:-keep] if keep > 0 else files:
            size_mb = old.stat().st_size / 1e6
            old.unlink()
            LOG.info("pruned %s (%.0f MB)", old.name, size_mb)


def screen_candidates(
    joint_config: Path,
    device: str,
    attempts: int,
    *,
    k: int,
) -> tuple[Path, list[dict]]:
    """Rank K merge configurations cheaply, return the winner's config path.

    Each candidate is trained for a short screening budget and scored on the
    same held-out evaluator as a full run, so the comparison is on the same
    scale as the final number. The screen is a prior, not evidence -- see
    hagi.orchestrator.merge_select -- and every screen score is returned so a
    wrong pick is visible in the ledger rather than hidden.

    With k=1 this is exactly the previous behaviour: one candidate, the base
    config unchanged, no screening overhead.
    """
    from hagi.orchestrator.merge_select import (
        enumerate_candidates,
        screen_steps_for,
        write_candidate_config,
    )

    if k <= 1:
        return joint_config, [{"label": "base", "screen_exact_ce": None}]

    raw = yaml.safe_load(joint_config.read_text(encoding="utf-8"))
    budget = int(raw["train"]["max_steps"])
    steps = screen_steps_for(budget)
    candidates = enumerate_candidates(raw["merge"]["expert_checkpoints"])[:k]
    LOG.info(
        "screening %d merge candidate(s) at %d steps each (full budget %d)",
        len(candidates), steps, budget,
    )

    scored: list[dict] = []
    for cand in candidates:
        cfg = write_candidate_config(
            joint_config,
            cand,
            joint_config.with_name(f"{joint_config.stem}__{cand.label}.yaml"),
            screen_steps=steps,
            checkpoint_dir=f"checkpoints/_screen_{cand.label}",
        )
        ckpt = train_phase(cfg, ROOT / f"logs/screen_{cand.label}.log", device, attempts)
        if ckpt is None:
            LOG.warning("screen %s produced no checkpoint; skipping", cand.label)
            continue
        report = evaluate(
            cfg, ckpt, ROOT / f"reports/screen_{cand.label}.json", device
        )
        if report is None:
            continue
        mean = mean_ce(report)
        scored.append({"label": cand.label, "screen_exact_ce": mean})
        LOG.info("screen %-18s mean CE %.4f", cand.label, mean if mean is not float("nan") else float("nan"))
        # The screen's checkpoints are disposable; keep only the winner's dir.
        for old in (ROOT / f"checkpoints/_screen_{cand.label}").glob("step-*.pt"):
            old.unlink()

    usable = [s for s in scored if s["screen_exact_ce"] is not None]
    if not usable:
        LOG.warning("no candidate scored; falling back to the base config")
        return joint_config, scored
    # §1 ``selection_hurts``: standalone CE ranking is NOT an ensemble
    # criterion and can strictly degrade the ensemble. It is kept ONLY as
    # a cheap pre-filter; the DECISION is the §9 certified gate in
    # decide(). Screen scores within the certification band are marked
    # undecided so a wrong pick stays visible in the ledger.
    best = min(usable, key=lambda s: s["screen_exact_ce"])
    sorted_usable = sorted(usable, key=lambda s: s["screen_exact_ce"])
    if len(sorted_usable) > 1:
        second = sorted_usable[1]
        cert = cc.certified_ab(
            second["screen_exact_ce"], DEFAULT_EVAL_N,
            best["screen_exact_ce"], DEFAULT_EVAL_N,
            eps=0.05, delta=0.05,
        )
        for s in usable:
            s["screen_certified"] = not cert.undecided
        if cert.undecided:
            LOG.warning(
                "screen top-2 not 2eps-separable (§9: %s) -- the pick is a "
                "prior, not evidence; the §9 gate in decide() rules",
                cert.reason,
            )
    LOG.info("screen winner: %s (%.4f)", best["label"], best["screen_exact_ce"])
    return (
        joint_config.with_name(f"{joint_config.stem}__{best['label']}.yaml"),
        scored,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True, help="YAML list of lanes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-lanes", type=int, default=0, help="0 = all lanes")
    ap.add_argument("--attempts", type=int, default=3, help="retries per train phase")
    ap.add_argument("--keep", type=int, default=2, help="checkpoints kept per run")
    ap.add_argument("--total-margin", type=float, default=DEFAULT_TOTAL_MARGIN)
    ap.add_argument("--domain-margin", type=float, default=DEFAULT_DOMAIN_MARGIN)
    ap.add_argument(
        "--gate-delta", type=float, default=DEFAULT_GATE_DELTA,
        help="R93: total error budget for ALL acceptance gates in the run. "
             "The per-gate margin is drawn from the geometric schedule "
             "delta_t = delta_0 rho^t, whose infinite sum is <= delta, so "
             "the budget does not grow with the number of lanes.",
    )
    ap.add_argument(
        "--gate-decay", type=float, default=DEFAULT_GATE_DECAY,
        help="R93: rho in (0,1). Smaller tightens later gates faster.",
    )
    ap.add_argument(
        "--merge-k", type=int, default=1,
        help="merge configurations to screen per lane (1 = current behaviour)",
    )
    ap.add_argument("--dry-run", action="store_true", help="plan only, run nothing")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    lanes = load_plan(ROOT / args.plan if not Path(args.plan).is_absolute() else Path(args.plan))
    if args.max_lanes:
        lanes = lanes[: args.max_lanes]
    LOG.info("plan: %d lane(s): %s", len(lanes), ", ".join(l.name for l in lanes))
    for lane in lanes:
        LOG.info("  %s: %d experts -> %s", lane.name, len(lane.expert_configs), lane.joint_config)
    if args.dry_run:
        return 0

    incumbent: dict | None = None
    incumbent_name: str | None = None
    accepted = 0
    # R93 comparison index: how many gates this run has already ruled on.
    # It counts every reached gate, not just accepted lanes, because a
    # rejected candidate consumes budget too -- that is the whole point of
    # the anytime schedule rather than a per-lane fixed margin.
    checks = 0

    for lane in lanes:
        started = time.time()
        LOG.info("=== lane %s (type=%s) ===", lane.name, lane.lane_type)
        checkpoints: list[Path] = []

        if lane.lane_type == "distill":
            parent = ROOT / lane.joint_config
            derived = make_distill_config(
                parent,
                parent.with_name(parent.stem + "__distill.yaml"),
            )
            log_phase(cc.GrowthPhase.DISTILL, lane.name,
                      f"config derived: {derived.name} (training NOT auto-launched "
                      "by derivation; run the lane to train)")
            ckpt = train_phase(
                derived, ROOT / f"logs/growth_{lane.name}_distill.log",
                args.device, args.attempts,
            )
            if ckpt is None:
                append_ledger({"lane": lane.name, "phase": "distill_failed",
                               "config": str(derived), "at": started})
                continue
            report = evaluate(
                derived, ckpt,
                ROOT / f"reports/growth_{lane.name}_distill.json", args.device,
            )
            teacher_report_path = lane.meta.get("teacher_report")
            teacher_report = None
            if teacher_report_path and (ROOT / teacher_report_path).is_file():
                teacher_report = json.loads(
                    (ROOT / teacher_report_path).read_text(encoding="utf-8"))
            elif incumbent is not None:
                teacher_report = incumbent
            leak = distill_leak_from_evals(lane, report, teacher_report, incumbent)
            if leak is None:
                log_phase(cc.GrowthPhase.DISTILL, lane.name, "inputs_missing")
            else:
                log_phase(cc.GrowthPhase.DISTILL, lane.name, leak["note"])
            append_ledger({
                "lane": lane.name,
                "phase": "distill_evaluated",
                "theory_phase": "DISTILL",
                "config": str(derived),
                "report": f"reports/growth_{lane.name}_distill.json",
                "elapsed_s": round(time.time() - started, 1),
                **({"distill_leak_gate": leak} if leak else {}),
            })
            prune(args.keep, [ckpt])
            continue

        for cfg_name in lane.expert_configs:
            log_phase(cc.GrowthPhase.DISCOVER, lane.name, cfg_name)
            cfg = ROOT / cfg_name
            ckpt = train_phase(cfg, ROOT / f"logs/growth_{lane.name}_{Path(cfg_name).stem}.log",
                              args.device, args.attempts)
            if ckpt is None:
                LOG.error("lane %s: expert %s failed; skipping lane", lane.name, cfg_name)
                append_ledger({"lane": lane.name, "phase": "expert_failed",
                               "config": cfg_name, "at": started})
                break
            checkpoints.append(ckpt)
        else:
            log_phase(cc.GrowthPhase.SELECT, lane.name,
                      f"merge_k={args.merge_k} (screen = cheap pre-filter only, "
                      "decision is the §9 certified gate)")
            joint = ROOT / lane.joint_config
            joint, screens = screen_candidates(
                joint, args.device, args.attempts, k=args.merge_k
            )
            log_phase(cc.GrowthPhase.CONSOLIDATE, lane.name, Path(joint).name)
            ckpt = train_phase(joint, ROOT / f"logs/growth_{lane.name}_joint.log",
                              args.device, args.attempts)
            if ckpt is None:
                LOG.error("lane %s: joint failed", lane.name)
                append_ledger({"lane": lane.name, "phase": "joint_failed", "at": started})
            else:
                log_phase(cc.GrowthPhase.MEASURE, lane.name, "eval_holdout 200 batches")
                report = evaluate(
                    joint, ckpt,
                    ROOT / f"reports/growth_{lane.name}.json", args.device,
                )
                if report is None:
                    append_ledger({"lane": lane.name, "phase": "eval_failed", "at": started})
                else:
                    log_phase(cc.GrowthPhase.GATE, lane.name, "§9 certified_ab 2eps rule")
                    verdict = decide(
                        report, incumbent, args.total_margin,
                        args.domain_margin, lane=checks,
                        total_delta=args.gate_delta, decay=args.gate_decay,
                    )
                    checks += 1
                    stop_note = theory_stop_check(lane, verdict)
                    if stop_note:
                        log_phase(cc.GrowthPhase.STOP_CONTINUE, lane.name, stop_note)
                        append_ledger({
                            "lane": lane.name, "phase": "evaluated",
                            "theory_phase": "GATE", "incumbent": incumbent_name,
                            "report": f"reports/growth_{lane.name}.json",
                            "elapsed_s": round(time.time() - started, 1),
                            **verdict.to_json(),
                            **{"stop_condition": stop_note},
                        })
                        # §0 step 6 HARD stop (audit P1-2): consensus +
                        # inj <= xi means retrying cannot help — the loop
                        # must terminate, not annotate.
                        LOG.info("stop_condition MET on lane %s — halting the growth loop", lane.name)
                        return 0
                    leak_note = distill_leak_check(lane)
                    if leak_note and "STOP" in leak_note:
                        append_ledger({
                            "lane": lane.name, "phase": "evaluated",
                            "distill_leak_gate": leak_note,
                            "report": f"reports/growth_{lane.name}.json",
                        })
                        # R134 HARD stop (audit P1-3): delta_k >= c_k —
                        # the distillation recursion no longer pays.
                        LOG.info("distill_leak_gate STOP on lane %s — halting", lane.name)
                        return 0
                    ignite = ignition_gate_check(
                        lane, ROOT / f"logs/growth_{lane.name}_joint.log"
                    )
                    append_ledger({
                        "lane": lane.name,
                        "phase": "evaluated",
                        "theory_phase": "GATE",
                        "merge_screen": screens,
                        "incumbent": incumbent_name,
                        "report": f"reports/growth_{lane.name}.json",
                        "elapsed_s": round(time.time() - started, 1),
                        **verdict.to_json(),
                        **({"stop_condition": stop_note} if stop_note else {}),
                        **({"distill_leak_gate": leak_note} if leak_note else {}),
                        **({"ignition_gate": ignite} if ignite else {}),
                    })
                    LOG.info("lane %s: accepted=%s (%s)", lane.name, verdict.accepted, verdict.reason)
                    if verdict.accepted:
                        accepted += 1
                        incumbent = report
                        incumbent_name = lane.name
                    prune(args.keep, checkpoints + [ckpt])

    LOG.info("done: %d/%d lane(s) accepted", accepted, len(lanes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
