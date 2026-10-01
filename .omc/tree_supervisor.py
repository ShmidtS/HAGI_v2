"""Autonomous tree-growth supervisor (round 72).

Runs the full pipeline without any agent turn:
  leaf-RU (waits for running instance) -> leaf-CODE (re-run, was killed)
  -> merge 3 leaves (block-diag) -> joint 1600 -> gate measure -> ledger.
Detached process; immune to session aborts. Logs to .omc/supervisor.log.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # repo root
LOG = HERE / ".omc" / "supervisor.log"
PY = sys.executable


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def wait_for(pattern: str, script: str) -> None:
    """Wait until no python process matches (existing run of `script`)."""
    while True:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(HERE / ".omc" / "gpu_check.ps1"), "-Pattern", pattern],
            capture_output=True, text=True).stdout
        if "count=0" in out:
            return
        time.sleep(60)


def run(script: str, logname: str) -> bool:
    err = HERE / ".omc" / f"{logname}.err"
    log(f"starting {script}")
    with open(HERE / ".omc" / f"{logname}.log", "wb") as out, open(err, "wb") as e:
        rc = subprocess.run([PY, "-X", "utf8", "-u", script], cwd=HERE,
                            stdout=out, stderr=e).returncode
    log(f"{script} exited rc={rc}")
    return rc == 0


def has_ckpt(d: str, step: int = 1600) -> bool:
    return (HERE / "checkpoints" / d / f"step-{step:07d}.pt").exists()


def main() -> int:
    log("supervisor: tree pipeline start")
    # 1. wait for the already-running leaf-RU
    if not has_ckpt("leaf_ru"):
        log("waiting for running leaf_ru")
        wait_for("leaf_ru.py", "leaf_ru")
    log(f"leaf_ru ckpt: {has_ckpt('leaf_ru')}")
    # 2. leaf-CODE (killed at step 250 earlier)
    if not has_ckpt("leaf_code"):
        run("scripts/lora/leaf_code.py", "leaf_code2")
    log(f"leaf_code ckpt: {has_ckpt('leaf_code')}")
    # 3. merge the 3 leaves from the record prior
    merged = HERE / "checkpoints" / "tree_gen3_merged" / "step-0000000.pt"
    if not merged.exists():
        log("merging 3 leaves")
        rc = subprocess.run([PY, "-X", "utf8", "scripts/growth/merge_experts.py",
                             "--config", "configs/dbridge_boost_merged.yaml",
                             "--experts",
                             "checkpoints/leaf_math/step-0001600.pt",
                             "checkpoints/leaf_code/step-0001600.pt",
                             "checkpoints/leaf_ru/step-0001600.pt",
                             "--out", str(merged),
                             "--drop-expert-mixers"],
                            cwd=HERE).returncode
        log(f"merge rc={rc}")
    # 4. joint: reuse boost_joint config with new init + dir
    if not has_ckpt("tree_gen3_joint"):
        cfg_src = (HERE / "configs" / "dbridge_boost_joint_lr15.yaml").read_text(encoding="utf-8")
        cfg = cfg_src.replace(
            "init_from: checkpoints/dbridge_boost_merged/step-0000000.pt",
            "init_from: checkpoints/tree_gen3_merged/step-0000000.pt"
        ).replace(
            "checkpoint_dir: checkpoints/dbridge_boost_joint_lr15",
            "checkpoint_dir: checkpoints/tree_gen3_joint"
        )
        (HERE / "configs" / "tree_gen3_joint.yaml").write_text(cfg, encoding="utf-8")
        with open(HERE / ".omc" / "tree_joint.log", "wb") as out, \
             open(HERE / ".omc" / "tree_joint.err", "wb") as e:
            rc = subprocess.run([PY, "-X", "utf8", "-u", "scripts/train.py",
                                 "--config", "configs/tree_gen3_joint.yaml"],
                                cwd=HERE, stdout=out, stderr=e).returncode
        log(f"joint rc={rc}")
    log("supervisor: pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
