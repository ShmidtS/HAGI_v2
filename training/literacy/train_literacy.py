#!/usr/bin/env python3
"""Train RopeAttentionByteLM on the literacy corpus (GPU-ready)."""
import sys, time, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HAGI_DEVICE", "cuda")

import torch
from hagi.lm import RopeAttentionByteLM

CORPUS = ROOT / "training" / "literacy" / "bits_corpus.txt"
CKPT = ROOT / "checkpoints" / "literacy_v3_bits.pt"

def main():
    data = CORPUS.read_bytes()
    print(f"corpus: {len(data)} bytes; device cuda={torch.cuda.is_available()}")
    lm = RopeAttentionByteLM(dim=256, hidden=512, theta0=1.0, lam=0.05, seed=1)
    print("model device:", lm.device)
    t0 = time.time()
    r = lm.train_guarded(data, epochs=200)
    print(f"train {time.time()-t0:.0f}s | best CE {r['best_holdout_ce']:.4f}")
    CKPT.parent.mkdir(exist_ok=True)
    torch.save({n: getattr(lm, n).detach().cpu()
                for n in ("emb", "q", "W1", "b1", "W2", "b2")}
               | {"dim": lm.dim, "hidden": lm.hidden,
                  "theta0": getattr(lm, "theta0", 0.0),
                  "lam": getattr(lm, "lam", 0.0)}, CKPT)
    print("saved:", CKPT)

    # генерация: продолжение булевой таблицы — проверка грамотности
    lm.model if hasattr(lm, 'model') else None
    try:
        out = lm.generate("0 XOR 0 = ", n_bytes=8)
        print("генерация:", out)
    except Exception as e:
        print("generate недоступен:", e)

if __name__ == "__main__":
    main()
