"""§17 disagreement-slice distillation stage (standalone, non-invasive).

Theory: R104 gain renewal + §17 Bregman-slice — growth lives on the
HIGH-DISAGREEMENT tokens (mean pairwise JS > batch q95); consensus
tokens carry ~zero gradient for the growth channel. Measured logit
twoGap between leaves is ~3.4 nats/token (measure_logit_twogap.py),
while the ladder harvests ~0.05/generation via plain averaging — the
disagreement_slice_loss is the theory's mechanism to harvest it.

Protocol (R222 kl_contraction_budget): student = merged root,
teachers = the N leaves of the SAME generation (frozen). On-policy
batches from the root_ft mix; loss = CE + lambda * KL(p̄_teachers ‖
p_student) restricted to disagreement tokens. Candidate is then
judged by the same §9 certified A/B as every other arm.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[2] / "src"
for _p in (_SRC,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hagi.config import load_config  # noqa: E402
from hagi.data.dataset import PackedMixDataset, load_mix  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.checkpoint import load_model, save_checkpoint  # noqa: E402
from hagi.train.disagreement_distill import disagreement_slice_loss  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gen", type=int, required=True,
                   help="generation whose leaves act as teachers")
    p.add_argument("--student", required=True,
                   help="merged root.pt (or any ckpt) to distill INTO")
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("--config", required=True,
                   help="root_ft config (mix + geometry for student)")
    p.add_argument("--domains", nargs="+",
                   default=["math", "code", "ru", "enedu", "enweb", "enwiki"])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lam", type=float, default=1.0,
                   help="weight of the disagreement-slice KL term")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    configure_runtime()
    device = torch.device(args.device)

    cfg = load_config(args.config)
    student = HAGI(cfg).to(device)
    step0, _ = load_model(args.student, student, str(device))
    student.train()

    teachers = []
    for d in args.domains:
        ck = ROOT_CK / f"gen{args.gen}_leaf_{d}" / "best.pt"
        lc = Path(args.config).parent / f"leaf_{d}.yaml"
        if not (ck.exists() and lc.exists()):
            continue
        tcfg = load_config(str(lc))
        t = HAGI(tcfg).to(device).eval()
        load_model(str(ck), t, str(device))
        for pp in t.parameters():
            pp.requires_grad_(False)
        teachers.append(t)
    if len(teachers) < 2:
        raise SystemExit("need >= 2 teacher leaves")
    print(f"[distill] student={args.student} teachers={len(teachers)}")

    dc = cfg.train.data
    mix = load_mix(dc.data_dir, load_mix_weights(args.config))
    ds = PackedMixDataset(
        data_dir=dc.data_dir, seq_len=dc.seq_len,
        eos_token_id=dc.eos_token_id, weights=mix, seed=dc.seed,
        cross_doc_attention=dc.cross_doc_attention)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size,
                        num_workers=0, drop_last=True)

    opt = torch.optim.AdamW(
        (pp for pp in student.parameters() if pp.requires_grad),
        lr=args.lr, weight_decay=0.01)

    hist = []
    it = iter(loader)
    for s in range(args.steps):
        batch = next(it)
        ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        with torch.no_grad():
            t_logits = [t(ids, return_logits=True).logits.float()
                        for t in teachers]
        out = student(ids, targets, return_logits=True)
        hidden = out.hidden.detach().reshape(-1, out.hidden.shape[-1])
        flat_t = targets.reshape(-1)
        ce = student.head.exact_loss(hidden, flat_t)
        # JS-mask on a row SUBSAMPLE (default 8k rows): full-batch fp64
        # mean pairwise JS over 15 teacher pairs x 131k rows x V=32768
        # costs minutes per step; the q95 threshold estimated on 8k
        # rows is statistically equivalent (and the slice KL trains on
        # the same subsample — the gradient is an unbiased estimate of
        # the full-batch slice gradient).
        v = out.logits.shape[-1]
        rows_s = out.logits.reshape(-1, v)
        rows_t = [lg.reshape(-1, v) for lg in t_logits]
        n_rows = rows_s.shape[0]
        idx = torch.randperm(n_rows, device=device)[: min(2048, n_rows)]
        sl = disagreement_slice_loss(rows_s[idx], [t[idx] for t in rows_t])
        loss = ce + args.lam * sl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        opt.step()
        hist.append(float(ce))
        if (s + 1) % 25 == 0:
            print(f"  step {s+1}/{args.steps} ce={hist[-1]:.4f} "
                  f"slice={float(sl):.4f}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(student, cfg, step0 + args.steps, out_dir)
    print(json.dumps({"out": str(out_dir), "ce_first": hist[0],
                      "ce_last": hist[-1],
                      "ce_gain": hist[0] - hist[-1]}))
    return 0


ROOT_CK = Path(__file__).resolve().parents[2] / "checkpoints"


def load_mix_weights(cfg_path: str) -> dict:
    import yaml
    raw = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    return raw.get("train", {}).get("data", {}).get("mix",
                                                    {"edu": 0.35})


if __name__ == "__main__":
    raise SystemExit(main())
