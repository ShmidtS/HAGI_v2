"""HAGI watchdog — the development loop must never stop.

Checks every 5 minutes: if no hagi_selfrewrite forever process is
alive (or the log has not grown for 2 cycles), restarts it detached.
Idempotent: only one watchdog lives at a time (pid lock file).
"""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCK = ROOT / "checkpoints/self/watchdog.lock"
LOG = ROOT / "checkpoints/self/forever.log"
STALE_S = 3 * 3600  # no log growth for 3h -> restart too


PS_DETECT = ROOT / "checkpoints/self/detect.ps1"

def _forever_alive() -> bool:
    PS_DETECT.parent.mkdir(parents=True, exist_ok=True)
    PS_DETECT.write_text(
        "$c = @(Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" "
        "| Where-Object {$_.CommandLine -match 'selfrewrite'})\n"
        "$c.Count\n")
    out = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(PS_DETECT)], capture_output=True, text=True).stdout.strip()
    try:
        return int(out.splitlines()[-1]) > 0
    except (ValueError, IndexError):
        return False


def _log_stale() -> bool:
    try:
        return time.time() - LOG.stat().st_mtime > STALE_S
    except OSError:
        return False


PS_DEDUP = ROOT / "checkpoints/self/dedup.ps1"


def _dedup_forever() -> None:
    """If >1 forever mains are alive, keep the oldest, kill the rest
    (two writers on one state file corrupt the loop)."""
    PS_DEDUP.parent.mkdir(parents=True, exist_ok=True)
    PS_DEDUP.write_text(
        "$f = @(Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" "
        "| Where-Object {$_.CommandLine -match 'selfrewrite'} "
        "| Sort-Object CreationDate)
"
        "if ($f.Count -gt 1) { $f | Select-Object -Skip 1 "
        "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force } }
")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy",
                    "Bypass", "-File", str(PS_DEDUP)], capture_output=True)


def _start_forever() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        [r"C:\Program Files\Python313\pythonw.exe", "-X", "utf8", "-u",
         "hagi_selfrewrite.py", "--forever", "--rounds", "10",
         "--workers", "32"],
        cwd=str(ROOT),
        stdout=open(LOG, "a", encoding="utf-8"),
        stderr=subprocess.STDOUT, creationflags=0x00000008)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[watchdog] restarted forever pid={p.pid}\n")
    return p.pid


def main() -> None:
    if LOCK.exists():
        try:
            old = int(LOCK.read_text())
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"(Get-Process -Id {old}).Count"],
                            capture_output=True)
            # if still alive, exit silently
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"if (Get-Process -Id {old} -ErrorAction SilentlyContinue)"
                 " {{1}} else {{0}}"], capture_output=True, text=True)
            if r.stdout.strip() == "1":
                return
        except (ValueError, OSError):
            pass
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()))
    while True:
        try:
            _dedup_forever()
            if not _forever_alive() or _log_stale():
                _start_forever()
        except Exception as e:
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(f"[watchdog] error: {e!r}\n")
        time.sleep(300)


if __name__ == "__main__":
    main()
