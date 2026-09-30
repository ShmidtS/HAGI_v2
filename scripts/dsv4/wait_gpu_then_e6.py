"""Queue the N=6 expert chain behind whatever training is already running.

Standing policy: never two trainings at once. ``pgrep`` is unreliable on
Windows and a previous version of this script silently started a second
trainer while the merged-ternary run held the GPU, so the wait is done by
inspecting the process table for a live ``train.py`` instead.
"""
from __future__ import annotations

import subprocess
import sys
import time

import psutil


def busy() -> bool:
    for process in psutil.process_iter(["name", "cmdline"]):
        if process.info["name"] != "python.exe":
            continue
        cmdline = process.info["cmdline"] or []
        if any("scripts/train.py" in part for part in cmdline):
            return True
    return False


def main() -> int:
    while busy():
        time.sleep(20)
    return subprocess.call(["bash", "scripts/train_e6_experts.sh"])


if __name__ == "__main__":
    sys.exit(main())
