"""Restore unified model by replaying ALL saved domain chunks
(412 ns_*.bin) through ingest — same path as the original cycle.
Run with 1-thread OpenBLAS env (see watchdog pattern)."""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hagi.unified import Unified


def main() -> None:
    dom_dir = ROOT / "checkpoints" / "domains"
    chunks = sorted(dom_dir.glob("ns_*.bin"))
    print(f"restore: {len(chunks)} domain chunks", flush=True)
    u = Unified.load() if Unified.exists() else Unified()
    t0 = time.perf_counter()
    for i, p in enumerate(chunks, 1):
        blob = p.read_bytes()
        if len(blob) < 16:
            continue
        r = u.ingest(p.stem, blob, epochs=2, repeat=1, replay_cap=None)
        ce = r.get("holdout_ce", -1)
        print(f"[restore {i}/{len(chunks)}] {p.stem} -> CE {ce} "
              f"({r.get('seconds', '?')}s)", flush=True)
    print(f"restore done in {(time.perf_counter()-t0)/3600:.1f}h",
          flush=True)


if __name__ == "__main__":
    main()
