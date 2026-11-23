"""WAVE 243: measure the optimal replay fraction on OUR model
(RopeAttentionByteLM), two conflicting domains, equal update
budget. Hypothesis under test (user): 1/e ~= 0.368."""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HAGI_DEVICE", "cpu")
import numpy as np, torch
from hagi.lm import RopeAttentionByteLM

K = 96
FRACS = [0.26, 0.28, 1 / math.pi, 0.32, 0.34, 0.368]
SEEDS = [7, 11, 23, 41, 73, 97, 131]
STEPS, BATCH = 300, 128


def domain(seed, kind):
    rng = np.random.default_rng(seed)
    if kind == "A":   # периодический закон
        a = np.arange(6000) * 7 % 256
        return ((a + (np.arange(6000) % 3) * 5) % 256).astype(np.uint8)
    # B: другой закон
    a = np.arange(6000) * 13 % 256
    return ((a ^ (np.arange(6000) % 7)) % 256).astype(np.uint8)


def windows(d, pos):
    pos = np.asarray(pos)
    X = np.stack([d[p:p + K][::-1] for p in pos]).astype(np.int64)
    return torch.as_tensor(X), torch.as_tensor(d[pos + K].astype(np.int64))


def ce(m, X, y):
    with torch.no_grad():
        tot = 0
        for s in range(0, len(X), 512):
            tot += float(torch.nn.functional.cross_entropy(m.logits(X[s:s+512]), y[s:s+512])) * len(X[s:s+512])
    return tot / len(X)


def run(frac, seed):
    g = torch.Generator().manual_seed(seed)
    A, B = domain(1, "A"), domain(2, "B")
    m = RopeAttentionByteLM(dim=64, hidden=128, seed=seed)
    rng = np.random.default_rng(seed)
    a_pos = np.arange(0, 3000)
    opt = torch.optim.Adam([m.emb, m.W1, m.b1, m.W2, m.b2], lr=3e-3)
    def step(X, y):
        opt.zero_grad(); torch.nn.functional.cross_entropy(m.logits(X), y).backward(); opt.step()
    for _ in range(200):
        step(*windows(A, rng.permutation(a_pos)[:BATCH]))
    # B with replay fraction
    b_pos = np.arange(0, 3000)
    n_old = int(BATCH * frac)
    for _ in range(STEPS):
        pb = rng.permutation(b_pos)[:BATCH - n_old]
        if n_old:
            po = rng.permutation(a_pos)[:n_old]
            Xb, yb = windows(B, pb); Xo, yo = windows(A, po)
            X, y = torch.cat([Xb, Xo]), torch.cat([yb, yo])
        else:
            X, y = windows(B, pb)
        step(X, y)
    testA, testB = np.arange(3500, 4500), np.arange(3500, 4500)
    return ce(m, *windows(A, testA)), ce(m, *windows(B, testB))


if __name__ == "__main__":
    t0 = time.time()
    print("frac      A_CE    B_CE   balanced")
    res = {}
    for f in FRACS:
        allr = np.array([run(f, s) for s in SEEDS])
        r = allr.mean(0); b = allr.mean(1)
        res[f] = r
        print(f"{f:5.3f}  A={r[0]:6.3f} B={r[1]:6.3f} bal={b.mean():6.3f} "
              f"±{b.std(ddof=1)/math.sqrt(len(b)):.3f}", flush=True)
    best = min(res, key=lambda f: res[f].mean())
    print(f"best fraction: {best:.3f} (1/e={1/math.e:.3f}) | {time.time()-t0:.0f}s")
