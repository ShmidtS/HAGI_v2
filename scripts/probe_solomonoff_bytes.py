"""Score the released self-play learner on REAL utf-8 text, not on ids mod 256.

The first attempt fed it `compacted_token_id % 256`, which invents a byte
frequency distribution out of an unrelated token space. That is not text, and
the resulting unigram floor was an artifact of the approximation. This version
reads actual bytes from the tail of the raw corpus the project's BPE was
trained on.
"""
import importlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
sys.path.insert(0, str(Path.home() / "solomonoff"))

import numpy as np
import torch
import torch.nn.functional as F

import solomonoff_model.model as M

CFG = json.loads((Path.home() / "solomonoff" / "config.json").read_text())["learner"]
WEIGHTS = Path.home() / "solomonoff" / "learner_1M.pth"
N_CTX, WINDOWS = 1024, 16


def real_bytes(n: int = N_CTX * WINDOWS + 4096) -> np.ndarray:
    src = Path.cwd() / "data" / "bpe_corpus.txt"
    with src.open("rb") as fh:
        fh.seek(300_000_000)  # tail region; no training run reads this offset
        blob = fh.read(n)
    return np.frombuffer(blob, dtype=np.uint8).astype(np.int64)


def bits_per_byte(model, arr: np.ndarray, device: str) -> float:
    tot = n = 0
    with torch.no_grad():
        for w in range(WINDOWS):
            ch = arr[w * N_CTX:(w + 1) * N_CTX]
            if len(ch) < 2:
                break
            ids = torch.from_numpy(np.ascontiguousarray(ch)).long().unsqueeze(0).to(device)
            # Their model masks every row that does not begin with the output
            # prefix byte 'O', so a raw byte stream needs that prefix to be a
            # valid output row at all.
            ids = torch.cat(
                [torch.full((1, 1), ord("O"), dtype=ids.dtype, device=device), ids], dim=1
            )
            lg = model(ids)[0]
            tot += F.cross_entropy(
                lg[:, :-1].reshape(-1, lg.shape[-1]).float(),
                ids[:, 1:].reshape(-1),
                reduction="sum",
            ).item()
            n += ids.shape[1] - 2
    return tot / n / math.log(2)


def unigram_floor(arr: np.ndarray) -> float:
    """Bits/byte achievable from raw byte frequencies alone -- no order.

    This is the control that decides the question. A byte unigram model is
    exactly what you get from text statistics, so a self-play learner that
    cannot beat it transferred no order-dependence and the whole "use it as a
    leaf" proposal dies here.
    """
    fit, test = arr[: len(arr) // 2], arr[len(arr) // 2:]
    cnt = np.bincount(fit, minlength=256).astype(np.float64)
    q = (cnt + 1.0) / (cnt.sum() + 256.0)
    return float(-np.log2(q[test]).mean())


def main() -> int:
    arr = real_bytes()
    print(f"real utf-8 bytes: {len(arr)}  distinct byte values: {len(np.unique(arr))}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model = M.ProgramLanguageModel(**CFG)
    missing, unexpected = model.load_state_dict(blob["learner_state_dict"], strict=False)
    print(f"checkpoint round={blob.get('round')} missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(dev).eval()

    torch.manual_seed(0)
    fresh = M.ProgramLanguageModel(**CFG).to(dev).eval()

    rnd = np.random.default_rng(0).integers(0, 256, size=len(arr)).astype(np.int64)

    print()
    print(f"{'input':<26} {'released':>9} {'untrained':>10}   bits/byte")
    for tag, a in (("real utf-8 text", arr), ("random bytes", rnd)):
        r = bits_per_byte(model, a, dev)
        u = bits_per_byte(fresh, a, dev)
        print(f"{tag:<26} {r:>9.4f} {u:>10.4f}")

    # The floor a model with NO order-dependence can reach on these bytes.
    floor = unigram_floor(arr)
    rel = bits_per_byte(model, arr, dev)
    print()
    print(f"unigram byte floor (no order information): {floor:.4f}")
    print(f"released learner on real text:             {rel:.4f}")
    print(f"margin over the unigram floor:             {floor - rel:+.4f} bits/byte")
    print(f"uniform over 256 would be:                  8.0000")
    verdict = "ORDER-DEPENDENT STRUCTURE TRANSFERRED" if rel < floor else "NO STRUCTURE BEYOND BYTE FREQUENCY"
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def unigram_floor(arr: np.ndarray) -> float:
    fit, test = arr[: len(arr) // 2], arr[len(arr) // 2:]
    cnt = np.bincount(fit, minlength=256).astype(np.float64)
    q = (cnt + 1.0) / (cnt.sum() + 256.0)
    return float(-np.log2(q[test]).mean())
