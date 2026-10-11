"""Formula-exact logit twoGap (Lean Hagi/Ensemble/GapLaw.lean:99).

twoGap = ½·log Σ_u Σ_v p_u·p_v·cosh(d_u − d_v),  p = softmax(midpoint).

The double sum FACTORIZES exactly (the cosh form kills the sinh part
by (u,v)-symmetry, GapLaw.lean twoGap_cosh):
    Σ_u Σ_v p_u p_v·exp(d_u − d_v) = (Σ_u p_u e^{d_u})·(Σ_v p_v e^{−d_v})
so the exact per-token value costs O(V) via logsumexp — no O(V²).

This is the THEORY quantity (Jensen gap on logits; twoGap_ce_identity:
merge gap == (CE_1+CE_2)/2 − CE_merge). The runtime merge gate
(pilot_latent_merge) uses a documented WEIGHT-level proxy; this script
measures the formula-exact logit gap for comparison per generation.

Usage:
    python scripts/growth/measure_logit_twogap.py --gen 42 --batches 2 \
        --domain math --device cpu
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.config import load_config  # noqa: E402
from hagi.data.dataset import PackedMixDataset, load_mix  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.checkpoint import load_model  # noqa: E402

DOMAIN_MIX = {
    "math": {"openwebmath": 1.0},
    "code": {"python_instruct": 1.0},
    "ru": {"wikipedia_ru": 0.5, "oscar_ru": 0.5},
    "enedu": {"edu": 1.0},
    "enweb": {"slimpajama": 1.0},
    "enwiki": {"wikipedia_en": 1.0},
}


@torch.no_grad()
def measure(
    gen: int,
    domains: list[str],
    config_dir: str,
    batches: int = 2,
    device: str = "cpu",
    domain: str = "math",
    seed: int = 15111,
) -> dict:
    pairs = []
    for d in domains:
        ck = Path(f"checkpoints/gen{gen}_leaf_{d}/best.pt")
        cfg = Path(config_dir) / f"leaf_{d}.yaml"
        if ck.exists() and cfg.exists():
            pairs.append((d, str(cfg), str(ck)))
    if len(pairs) < 2:
        raise SystemExit(f"need >=2 leaf checkpoints, found {len(pairs)}")

    cfg0 = load_config(pairs[0][1])
    dc = cfg0.train.data
    mix = load_mix(dc.data_dir, DOMAIN_MIX.get(domain, {"edu": 1.0}))
    # Split-half needs DISJOINT samples: with a single-source mix the
    # first window always starts at offset 0 regardless of the rng seed,
    # so seeds alone give identical batches. Derive per-seed read offsets
    # instead (each half reads a different slice of the corpus).
    offset = (seed % 1000) * 8192
    dataset = PackedMixDataset(
        data_dir=dc.data_dir, seq_len=dc.seq_len,
        eos_token_id=dc.eos_token_id, weights=mix, seed=seed,
        cross_doc_attention=dc.cross_doc_attention,
        start_offsets={n: offset for n in mix})
    loader = DataLoader(dataset, batch_size=cfg0.train.batch_size,
                        num_workers=0, drop_last=True)

    models = []
    for _l, cfg_path, ck in pairs:
        cfg = load_config(cfg_path)
        m = HAGI(cfg).to(device).eval()
        load_model(ck, m, device=device)
        models.append(m)

    # Memory discipline: logits are full-vocab float32 tensors; holding
    # all N at once costs N * B*T*V*4 bytes (tens of GB on CPU). Instead
    # process ONE pair at a time -- peak = 2 logits, not N. Same data
    # (fixed seed) per pair, so values are comparable across pairs.
    def pair_gaps(i: int, j: int) -> tuple[list[float], int]:
        out: list[float] = []
        tok = 0
        for bi, batch in enumerate(loader):
            if bi >= batches:
                break
            ids = batch["input_ids"].to(device)
            tok += ids.numel()
            d_i = models[i](ids, return_logits=True).logits.float()
            d_j = models[j](ids, return_logits=True).logits.float()
            logp = torch.log_softmax((d_i + d_j) / 2, dim=-1)
            # factorized exact: ΣΣ p_u p_v e^{d_u−d_v} = A·B  (cosh form)
            A = torch.logsumexp(logp + d_i, dim=-1)
            B = torch.logsumexp(logp - d_j, dim=-1)
            out.append((0.5 * (A + B)).mean().item())  # per-token twoGap
            del logp, A, B, d_i, d_j
        return out, tok

    res: dict[str, float] = {}
    tok_total = 0
    for i, j in itertools.combinations(range(len(models)), 2):
        gaps, tok_total = pair_gaps(i, j)
        key = f"{pairs[i][0]}~{pairs[j][0]}"
        res[key] = round(sum(gaps) / len(gaps), 6) if gaps else float("nan")

    # free the N models before returning -- the caller may call measure()
    # again (split-half bracket) in the same process
    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vals = list(res.values())
    return {"gen": gen, "domain": domain, "tokens": tok_total,
            "pairs": res, "ensemble_mean_twoGap": round(sum(vals) / len(vals), 6)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gen", type=int, required=True)
    p.add_argument("--domains", nargs="+",
                   default=["math", "code", "ru", "enedu", "enweb", "enwiki"])
    p.add_argument("--config-dir", default=None)
    p.add_argument("--domain", default="math")
    p.add_argument("--batches", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=15111,
                   help="dataset seed; split-half bracket uses two seeds")
    a = p.parse_args()
    cdir = a.config_dir or f"checkpoints/gen{a.gen}/configs"
    out = measure(a.gen, a.domains, cdir, a.batches, a.device, a.domain,
                  a.seed)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
