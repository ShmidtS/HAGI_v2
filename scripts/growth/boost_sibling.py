"""Round 54: AdaHAGI — boosting siblings trained on the parent's error tail.

The idea (own proposal, grounded in our own measurements):
- growth saturated because siblings differ only by seed -> ensemble
  disagreement G -> 0 (GenCycle: G_inf = D/(1-rho) with D ~ 0);
- GapLaw: ensemble gain is proportional to expert disagreement;
- therefore a new leaf should train on the parent's ERROR TAIL --
  gradient boosting in growth space. Each generation focuses on what
  the ensemble still gets wrong, restoring the diversity source D.

Phase A (score+extract): one inference pass of the parent over probe
windows (one 2k window per 1M-token chunk, ~5 min GPU); per-chunk mean
CE selects the hardest 30% of each corpus; those chunks are byte-copied
into data/boost_<corp>.compact.bin with data/boost_mix.json (original
corpus weights renormalized).

Phase B (train): a standard sibling (H=384, init_from gen1 joint) on
the boosted stream -- directly comparable to the seed-only gen2 sibs
whose solo CE and merge behaviour are known.

BLIND PREDICTION (recorded before launch, .omc/attempts/boost_round54.md):
solo CE of the boost-sib is WORSE by 0.05-0.15 nat (harder data), but
disagreement with the parent pool rises >=30%, and a full 3-boost-sib
merge+joint beats seed-gen2 (3.4484) by >=0.05 nat. Solo CE > 4.2 would
mean the weighting is too aggressive.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, os.path.join(_REPO, "scripts", "lora"), _REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from hagi.train.loop import configure_runtime
from lora_gen2_joint_c8 import build_lora_prior  # pyright: ignore[reportMissingImports]
from pathlib import Path

CORP = ["edu", "python_instruct", "wikipedia_en", "wikipedia_ru", "oscar_ru", "openwebmath", "tinystories", "smoltalk"]
W = [0.3571, 0.2232, 0.0893, 0.0714, 0.0625, 0.0893, 0.0536, 0.0536]
CHUNK = 1_000_000        # tokens per scored chunk
PROBE = 2048             # probe window per chunk
TOP_FRAC = 0.30          # hardest fraction kept

def score_and_extract() -> None:
    configure_runtime()
    # parent = the RECORD: clamp-8 base + trained r16 adapters + deepen pass.
    # build from the BASE ckpt, then overlay the trained adapter state.
    parent = build_lora_prior(
        "configs/dbridge_gen2_merged_had_c8.yaml",
        "checkpoints/dbridge_gen2_merged_had_c8/step-0001600.pt", 16,
    ).eval().to(torch.bfloat16)
    plora = torch.load("checkpoints/dbridge_gen2_lora_c8_d2/step-0001600.pt", map_location="cpu", weights_only=False)
    parent.load_state_dict({k: v.to("cuda") for k, v in plora["model"].items()}, strict=True)

    mix_out = {}
    for corp, w in zip(CORP, W):
        path = Path(f"data/{corp}.compact.bin")
        n_tokens = path.stat().st_size // 4
        n_chunks = n_tokens // CHUNK
        scores = np.full(n_chunks, np.inf, dtype=np.float32)
        # probe from the chunk START (training slices stream sequentially;
        # the boost sibling will start from offset 0 and consume ~52M tokens,
        # so score the FIRST ~60 chunks that it can actually reach)
        reachable = min(n_chunks, 64)
        with path.open("rb") as fh:
            for c in range(reachable):
                fh.seek(c * CHUNK * 4)
                raw = np.frombuffer(fh.read((PROBE + 1) * 4), dtype=np.uint32).astype(np.int64)
                ids = torch.from_numpy(raw).cuda()
                x, y = ids[None, :-1], ids[None, 1:]
                with torch.no_grad():
                    o = parent(x, y)
                    flat = o.hidden.reshape(-1, o.hidden.shape[-1])
                    scores[c] = float(parent.head.exact_loss(flat, y.reshape(-1)))
        n_keep = max(1, int(reachable * TOP_FRAC))
        keep = np.argsort(scores[:reachable])[-n_keep:]  # HARDEST (highest CE), not easiest
        keep.sort()
        # extract the kept chunks (byte copy) into a boosted stream
        out = Path(f"data/boost_{corp}.compact.bin")
        with path.open("rb") as src, out.open("wb") as dst:
            for c in keep:
                src.seek(c * CHUNK * 4)
                dst.write(src.read(CHUNK * 4))
        mix_out[f"boost_{corp}"] = w
        print(f"{corp}: reachable {reachable} keep {n_keep} chunks "
              f"(CE {scores[keep].mean():.3f} kept vs {scores[:reachable].mean():.3f} all)",
              flush=True)
    tot = sum(mix_out.values())
    mix_out = {k: v / tot for k, v in mix_out.items()}
    Path("data/boost_mix.json").write_text(json.dumps({"sources": [
        {"name": k, "ratio": v} for k, v in mix_out.items()]}))
    print("wrote data/boost_mix.json", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["score", "train"], required=True)
    ap.add_argument("--sib", type=int, default=1)
    args = ap.parse_args()
    if args.phase == "score":
        score_and_extract()
    else:
        from hagi.config import load_config
        from hagi.data.dataset import build_dataloader
        from hagi.train.loop import train

        configure_runtime()
        cfg = load_config("configs/dbridge_gen2_sib1.yaml")
        # weights override REPLACES data/mix.json entirely (load_mix semantics):
        # the boosted sources are the sibling's whole world.
        cfg.train.data.weights = {
            s["name"]: float(s["ratio"])
            for s in json.loads(Path("data/boost_mix.json").read_text())["sources"]
        }
        cfg.train.data.seed = 15411 + (args.sib - 1) * 111
        cfg.train.checkpoint_dir = f"checkpoints/dbridge_boost_sib{args.sib}"
        cfg.model.init_seed = 12101 + (args.sib - 1)
        cfg.train.init_from = "checkpoints/dbridge_gen1_joint/step-0001600.pt"
        model = None
        from hagi.model.factory import build_model_for_config
        from hagi.train.checkpoint import load_payload
        cfg.train.zero_init_proj = False  # guard: prior ckpts predate the round-40 rule
        pl = load_payload(cfg.train.init_from, "cpu")
        model = build_model_for_config(cfg)
        model.load_state_dict(pl["model"], strict=True)
        model = model.to("cuda")
        dl = build_dataloader(cfg, "data", start_offset=0)
        last = None
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from train import setup_logging
        log_path = setup_logging(cfg.train.checkpoint_dir)
        print(f"logging to {log_path}", flush=True)
        for metrics in train(model, dl, cfg, start_step=0):
            last = metrics
        print("done:", last)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
