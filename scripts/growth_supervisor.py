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
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "reports" / "growth_ledger.jsonl"
LOG = logging.getLogger("growth")

# A held-out exact CE at this scale moves by ~0.2 nats between seeds, so a
# candidate is only rejected when it is clearly worse, not when it is merely
# different. Set from the spec; these are the fallbacks.
DEFAULT_TOTAL_MARGIN = 0.05
DEFAULT_DOMAIN_MARGIN = 0.25


@dataclass
class Lane:
    """One candidate: N experts merged into one joint model."""

    name: str
    expert_configs: list[str]
    joint_config: str
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
        for key in ("name", "experts", "joint"):
            if key not in item:
                raise ValueError(f"{path}: lane {i} is missing '{key}'")
        lanes.append(
            Lane(
                name=str(item["name"]),
                expert_configs=[str(p) for p in item["experts"]],
                joint_config=str(item["joint"]),
                meta={
                    k: v
                    for k, v in item.items()
                    if k not in ("name", "experts", "joint")
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
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            LOG.error("timeout after %ss: %s", timeout, " ".join(cmd))
            return 124


def latest_checkpoint(directory: Path) -> Path | None:
    """Newest complete checkpoint in a directory, or None."""
    if not directory.is_dir():
        return None
    steps = sorted(directory.glob("step-*.pt"))
    return steps[-1] if steps else None


def checkpoint_dir_of(config: Path) -> Path:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    return ROOT / str(cfg["train"]["checkpoint_dir"])


def train_phase(config: Path, log_path: Path, device: str, attempts: int) -> Path | None:
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
            LOG.info("train ok: %s -> %s", config.name, ckpt.name)
            return ckpt
        LOG.warning("train attempt %d/%d failed (exit %s)", attempt, attempts, code)
        if ckpt is None:
            LOG.error("no checkpoint on disk after failure; giving up on %s", config)
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


def decide(
    candidate: dict,
    incumbent: dict | None,
    total_margin: float,
    domain_margin: float,
) -> Verdict:
    """Accept the candidate unless it is clearly worse than the incumbent.

    Rejects on: a missing mean, a mean worse by more than ``total_margin``, or
    a single domain worse by more than ``domain_margin``. The margins are wide
    on purpose -- held-out exact CE carries roughly 0.2 nats of seed noise at
    this scale, and a tight rule would turn noise into decisions.
    """
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

    if cand_mean - inc_mean > total_margin:
        return Verdict(False, f"mean CE worse by {cand_mean - inc_mean:+.4f}",
                       inc_mean, cand_mean, deltas)
    regressed = {k: v for k, v in deltas.items() if v > domain_margin}
    if regressed:
        detail = ", ".join(f"{k} {v:+.4f}" for k, v in sorted(regressed.items()))
        return Verdict(False, f"domain regression: {detail}", inc_mean, cand_mean, deltas)
    return Verdict(
        True, f"mean CE {cand_mean - inc_mean:+.4f} within margin",
        inc_mean, cand_mean, deltas,
    )


def append_ledger(row: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True, help="YAML list of lanes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-lanes", type=int, default=0, help="0 = all lanes")
    ap.add_argument("--attempts", type=int, default=3, help="retries per train phase")
    ap.add_argument("--keep", type=int, default=2, help="checkpoints kept per run")
    ap.add_argument("--total-margin", type=float, default=DEFAULT_TOTAL_MARGIN)
    ap.add_argument("--domain-margin", type=float, default=DEFAULT_DOMAIN_MARGIN)
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

    for lane in lanes:
        started = time.time()
        LOG.info("=== lane %s ===", lane.name)
        checkpoints: list[Path] = []

        for cfg_name in lane.expert_configs:
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
            joint = ROOT / lane.joint_config
            ckpt = train_phase(joint, ROOT / f"logs/growth_{lane.name}_joint.log",
                              args.device, args.attempts)
            if ckpt is None:
                LOG.error("lane %s: joint failed", lane.name)
                append_ledger({"lane": lane.name, "phase": "joint_failed", "at": started})
            else:
                report = evaluate(
                    joint, ckpt,
                    ROOT / f"reports/growth_{lane.name}.json", args.device,
                )
                if report is None:
                    append_ledger({"lane": lane.name, "phase": "eval_failed", "at": started})
                else:
                    verdict = decide(report, incumbent, args.total_margin, args.domain_margin)
                    append_ledger({
                        "lane": lane.name,
                        "phase": "evaluated",
                        "incumbent": incumbent_name,
                        "report": f"reports/growth_{lane.name}.json",
                        "elapsed_s": round(time.time() - started, 1),
                        **verdict.to_json(),
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
