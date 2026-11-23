"""WAVE 245: batch size vs time-to-quality on OUR Rope model, equal
data (train_guarded, epochs=2), conflicting-free periodic+text mix."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HAGI_DEVICE", "cpu")
from hagi.lm import RopeAttentionByteLM
data = (open('README.md', 'rb').read() * 3)[:400_000]
print("batch   CE      s     CE/s-gain")
for b in (8, 16, 24, 32, 48):
    m = RopeAttentionByteLM(dim=64, hidden=128, seed=int(os.environ.get("SW_SEED","7")))
    t = time.perf_counter(); r = m.train_guarded(data, epochs=2, batch=b)
    s = time.perf_counter() - t
    print(f"{b:5}  {r['best_holdout_ce']:6.3f}  {s:6.1f}")
