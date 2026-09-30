"""Make a durable, NONZERO copy of the rank-8 LoRA fixture.

Why: the pytest-tmp fixture has A=0 and B=0 on disk, so delta = B*(A^T x) is
identically zero and a per-request `scale` cannot be shown to move anything. The
contour tool writes orthonormal A in memory at run time (run9 log: "wrote
orthonormal lora_a into 368 pairs"); this script bakes the same A, plus a modest
random B, into a file so the HTTP lever can be tested end to end.

Layout: GGUF ne is [in_dim, rank] for lora_a, and gguf-py's memmap view is the
transpose, (rank, in_dim). Orthonormal ROWS of that view means A^T A = I_rank.
data_offset is an absolute file offset (verified: last tensor's end == file size).
"""
import os
import shutil
import sys

import numpy as np
from gguf import GGUFReader

SRC = r"C:/Users/shmid/AppData/Local/Temp/pytest-of-shmid/pytest-187/lora0/bonsai_lora_rank8.gguf"
DST = r"C:/HAGI_v2/data/lora_fixtures/bonsai_lora_rank8_nonzero.gguf"

SIGMA_B = 0.05
SEED = 12345


def main():
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    print("copying 233 MiB ...", flush=True)
    shutil.copyfile(SRC, DST)

    r = GGUFReader(SRC)
    ts = sorted(r.tensors, key=lambda t: t.name)
    rng = np.random.default_rng(SEED)
    n_a = n_b = 0
    worst_ortho = 0.0
    max_b = 0.0

    with open(DST, "r+b") as fh:
        for t in ts:
            view = np.asarray(t.data).shape      # (rank, dim) per gguf-py
            rank, dim = view
            off, nb = int(t.data_offset), int(t.n_bytes)
            assert nb == rank * dim * 4, (t.name, nb, rank, dim)
            if t.name.endswith(".lora_a"):
                # QR of a tall gaussian -> orthonormal columns of (dim, rank)
                g = rng.standard_normal((dim, rank)).astype(np.float32)
                q, _ = np.linalg.qr(g)
                m = np.ascontiguousarray(q.T.astype(np.float32))
                worst_ortho = max(worst_ortho,
                                  float(np.abs(m @ m.T - np.eye(rank)).max()))
                arr = m
                n_a += 1
            else:
                arr = (rng.standard_normal((rank, dim)) * SIGMA_B).astype(np.float32)
                max_b = max(max_b, float(np.abs(arr).max()))
                n_b += 1
            mm = np.memmap(fh, dtype="<f4", mode="r+", offset=off, shape=view)
            mm[:] = arr
        fh.flush()
    del r

    print(f"A пар: {n_a}   B пар: {n_b}")
    print(f"худшее |A A^T - I| = {worst_ortho:.3e}   (ортонормальность подтверждена)")
    print(f"max|B| = {max_b:.4f}  (sigma = {SIGMA_B:.3f})")
    print("размер копии:", os.path.getsize(DST), "байт ->", DST)

    # read back through a fresh reader: proves the bytes landed, not just the call
    chk = GGUFReader(DST)
    za = zb = 0
    for t in chk.tensors:
        a = np.asarray(t.data)
        if t.name.endswith(".lora_a"):
            gram = a @ a.T
            if float(np.abs(gram - np.eye(gram.shape[0])).max()) > 1e-3:
                za += 1
        else:
            if float(np.abs(a).max()) == 0.0:
                zb += 1
    print(f"обратное чтение: A не ортонормальны в {za} случаях, B нулевые в {zb}")
    return 0 if (za == 0 and zb == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
