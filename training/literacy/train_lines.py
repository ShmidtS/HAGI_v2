#!/usr/bin/env python3
"""Line-aligned literacy training: every training context starts
at a line beginning, so the model always sees the full pattern.
Certified-step descent (same semantics as CertifiedStepLM)."""
import sys, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HAGI_DEVICE", "cuda")
import numpy as np
import torch
from hagi.lm import RopeAttentionByteLM

CORPUS = ROOT / "training" / "literacy" / "bits_corpus.txt"
CKPT = ROOT / "checkpoints" / "literacy_lines.pt"

def main():
    data = CORPUS.read_bytes()
    idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    # line-start positions
    starts = [0] + [i + 1 for i, b in enumerate(data) if b == 0x0A]
    starts = [s for s in starts if s + 1 < len(idx)]
    print(f"corpus {len(data)} bytes, {len(starts)} line starts")
    lm = RopeAttentionByteLM(dim=256, hidden=512, seed=1)
    print("device:", lm.device)
    t = lm.t
    opt = t.optim.Adam(lm.params(), lr=8e-4)
    rng = np.random.default_rng(7)
    n = len(starts)
    hold = set(rng.choice(n, max(16, n // 7), replace=False))
    train = [i for i in range(n) if i not in hold]
    dev = lm.device
    def batch(positions):
        X, Y = [], []
        for p in positions:
            s = starts[p]
            ctx = idx[max(0, s - 64):s + 1]   # ends AT line start+1? no:
            # use windows: predict bytes s..s+L-1 from preceding ctx
        return None
    # simpler: per line, full sequence as one sample: ctx = line[:-1],
    # targets = line[1:], with preceding 64 bytes as memory prefix
    samples = []
    for s in starts:
        e = s
        while e < len(idx) and idx[e] != 0x0A:
            e += 1
        line = idx[s:e + 1]
        if len(line) < 4:
            continue
        prefix = idx[max(0, s - 64):s]
        samples.append((prefix, line))
    print(f"line samples: {len(samples)}")
    hold_s = set(rng.choice(len(samples), max(8, len(samples)//7), replace=False))
    tr = [s for i, s in enumerate(samples) if i not in hold_s]
    va = [s for i, s in enumerate(samples) if i in hold_s]

    def vce():
        tot = cnt = 0
        with t.no_grad():
            for prefix, line in va:
                ids = torch.as_tensor(np.concatenate([prefix, line]),
                                      dtype=torch.long, device=dev)
                L = len(line) - 1
                logits = lm.full_logits(ids[:-1].unsqueeze(0))[0, -L:, :]
                tgt = torch.as_tensor(line[1:], dtype=torch.long, device=dev)
                tot += float(t.nn.functional.cross_entropy(
                    logits, tgt, reduction='sum'))
                cnt += L
        return tot / max(1, cnt)

    best = vce()
    print(f"init holdout CE {best:.4f}")
    t0 = time.time()
    for ep in range(150):
        order = rng.permutation(len(tr))
        for bs in range(0, len(tr), 32):
            sel = order[bs:bs + 32]
            ids, tgts, lens = [], [], []
            for i in sel:
                prefix, line = tr[i]
                ids.append(np.concatenate([prefix, line]))
                lens.append(len(line) - 1)
            maxlen = max(len(x) for x in ids)
            X = np.full((len(ids), maxlen - 1), 32)
            Y = np.full((len(ids), maxlen - 1), 32)
            for j, (x, L) in enumerate(zip(ids, lens)):
                x_in, y_out = x[:-1], x[1:]
                X[j, -len(x_in):] = x_in
                Y[j, -len(y_out):] = y_out
                # mask prompt positions with -100
            Xt = torch.as_tensor(X, dtype=torch.long, device=dev)
            Yt = torch.as_tensor(Y, dtype=torch.long, device=dev)
            loss_mask = torch.zeros_like(Yt, dtype=torch.bool)
            for j, L in enumerate(lens):
                loss_mask[j, -L:] = True
            logits = lm.full_logits(Xt)
            loss = t.nn.functional.cross_entropy(
                logits.view(-1, 256), Yt.view(-1), reduction='none')
            loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 20 == 0 or ep == 199:
            v = vce()
            print(f"ep {ep}: holdout CE {v:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if v < best:
                best = v
                torch.save({nm: getattr(lm, nm).detach().cpu()
                            for nm in ("emb", "q", "W1", "b1", "W2", "b2")}
                           | {"dim": lm.dim, "hidden": lm.hidden,
                              "theta0": lm.theta0, "lam": lm.lam}, CKPT)
    print(f"best CE {best:.4f}; saved {CKPT}")
    # literacy check (argmax)
    from hagi.lm import load_rope
    lm2 = load_rope(CKPT)
    lm2.device = lm.device
    for nm in lm2.params():
        nm.data = nm.data.to(lm2.device)
    ok = 0
    # WAVE 181: format matches the generated corpus (4-bit tables)
    tests = [("0101 XOR 0011 = ", "0110"), ("0101 AND 0011 = ", "0001"),
             ("1100 XOR 1010 = ", "0110"), ("NOT 0101 = ", "1010"),
             ("6 XOR 3 = ", "5"), ("CHR 97 ", "HEX 61")]
    for tt, want in tests:
        out = lm2.generate(tt, n_bytes=len(want)+2, temp=0.1)
        gen = out[len(tt):].strip()
        hit = gen.startswith(want)
        ok += hit
        print(f"  {tt!r} -> {gen!r} want {want!r} {'OK' if hit else 'MISS'}")
    print(f"LITERACY: {ok}/{len(tests)}")

if __name__ == "__main__":
    main()
