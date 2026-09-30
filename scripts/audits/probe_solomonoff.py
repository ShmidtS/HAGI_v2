"""Does the released self-play learner carry structure that transfers to our text?

Claim under test (arXiv 2609.30063): a learner trained ONLY on programs, with
no natural data, predicts natural data better than chance, and pre-training on
its output accelerates pre-training on natural data.

Why this is testable here rather than an appeal to the paper:
  * the released learner is byte-level (`vocab_size: 256`), so its alphabet is
    disjoint from our 32768-token one -- no shared table of facts can leak in;
  * we score it on OUR held-out text, not on the paper's corpora.

The model is built from the AUTHORS' OWN `model.py` (scoring/src/framework in
the paper's repository) rather than a reimplementation: a hand-rolled
equivalent silently disagreed with their RoPE and head layout and produced
nonsense losses, which is exactly the failure this indirection avoids.

Usage:
    python scripts/probe_solomonoff.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path.home() / "solomonoff"


def load_their_model(their_model_py: Path, cfg_path: Path, weights: Path, device: str):
    # The authors' module uses a relative import, so it has to be loaded as a
    # package member rather than a loose file.
    # The package root is the PARENT of the package directory: putting the
    # package itself on sys.path makes `import <pkg>` fail.
    pkg_dir = their_model_py.parent.parent
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '../../..'))
for _p in (_HERE, _REPO, os.path.join(_REPO, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
    for name in list(sys.modules):
        if name.startswith("solomonoff_model"):
            del sys.modules[name]
    mod = importlib.import_module("solomonoff_model.model")
    sys.modules["their_model"] = mod
    cfg = json.loads(cfg_path.read_text())["learner"]
    model = mod.ProgramLanguageModel(**cfg)
    blob = torch.load(weights, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(blob["learner_state_dict"], strict=False)
    # wte is their embedding; the paper ties the read-out to nothing, so a
    # missing lm_head would silently make every logit identical.
    if any("lm_head" in k for k in missing):
        raise SystemExit(f"checkpoint does not fill lm_head: {missing[:5]}")
    return model.to(device).eval(), blob.get("round"), len(missing), len(unexpected)


def bits_per_byte(model, data: np.ndarray, device: str, n_ctx: int = 1024,
                  windows: int = 24) -> float:
    tot, n = 0.0, 0
    with torch.no_grad():
        for w in range(windows):
            chunk = data[w * n_ctx:(w + 1) * n_ctx]
            if len(chunk) < 2:
                break
            ids = torch.from_numpy(np.ascontiguousarray(chunk)).long().unsqueeze(0).to(device)
            # Their model masks any row that does not start with the output
            # prefix byte 'O' (it assumes a program, not free text), which
            # returns -inf for every logit. Prepending 'O' is what makes a
            # raw byte stream a valid OUTPUT row for them.
            ids = torch.cat([torch.full((1, 1), ord("O"), dtype=ids.dtype,
                                        device=ids.device), ids], dim=1)
            logits = model(ids)[0]
            tot += F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                ids[:, 1:].reshape(-1), reduction="sum",
            ).item()
            n += ids.shape[1] - 2  # skip the artificial prefix pair
    return tot / n / math.log(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-py", default=str(HERE / "solomonoff_model" / "model.py"))
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--weights", default=str(HERE / "learner_1M.pth"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-bytes", type=int, default=24 * 1024)
    args = ap.parse_args()

    model, rnd, n_missing, n_unexpected = load_their_model(
        Path(args.model_py), Path(args.config), Path(args.weights), args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint round={rnd}  params={n_params:,}  "
          f"d_model={model.d_model} heads={model.n_heads} layers={model.n_layers} "
          f"kv_heads={model.n_kv_heads} vocab={model.vocab_size}")
    print(f"load_state_dict: {n_missing} missing, {n_unexpected} unexpected")

    from hagi.data.dataset import dataset_path

    # Real bytes from the untouched tail of wikipedia_ru. The compacted stream
    # stores 32768-space ids, so the ids are mapped back to source ids and then
    # taken modulo 256: an approximation of true bytes, and labelled as such.
    path = dataset_path(Path.cwd() / "data", "wikipedia_ru")
    raw = np.memmap(path, dtype=np.int32, mode="r")
    tail = np.asarray(raw[-400_000:]).astype(np.int64)
    mfile = Path.cwd() / "data" / "vocab_map.npz"
    if mfile.exists():
        new_to_old = np.load(mfile)["new_to_old"]
        tail = new_to_old[np.clip(tail, 0, len(new_to_old) - 1)]
    real = (tail % 256)[: args.n_bytes].astype(np.int64)
    print(f"text source: {path.name} tail, {len(real)} bytes "
          f"(ids->bytes via vocab_map + mod 256: an APPROXIMATION of real text)")

    rnd_arr = np.random.default_rng(0).integers(0, 256, size=len(real)).astype(np.int64)
    cycle = np.tile(np.arange(256, dtype=np.int64), len(real) // 256 + 1)[: len(real)]

    print()
    print(f"{'data':<32} {'bits/byte':>10}   (8.0 = uniform over 256)")
    for tag, arr in (("random bytes", rnd_arr), ("uniform cycle 0..255", cycle),
                     ("real wikipedia_ru tail", real)):
        print(f"{tag:<32} {bits_per_byte(model, arr, args.device):>10.4f}")

    cfg = json.loads(Path(args.config).read_text())["learner"]
    torch.manual_seed(0)
    fresh = sys.modules["their_model"].ProgramLanguageModel(**cfg).to(args.device).eval()
    print()
    for tag, arr in (("untrained, real text", real), ("untrained, random bytes", rnd_arr)):
        print(f"{tag:<32} {bits_per_byte(fresh, arr, args.device):>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
