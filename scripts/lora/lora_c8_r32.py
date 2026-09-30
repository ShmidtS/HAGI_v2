"""Round 59: rank-32 deepen on the tables.

The record (d2, 3.3835) = clamp-8 base + r16 adapters + 3200 steps.
Round-58 spectra: kappa~0.002 -> rank-16 far from spectral
saturation. This run widens the adapters to r32 (old 16 components
carried over, new 16 zero-initialized = step-0 identical to the
record) and trains 1600 more steps on a fresh seed.

Blind prediction (.omc/attempts/rank32_round59_pred.md): gain in
[0.004, 0.015]; <0.004 closes the rank axis on the tables too.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '../..'))
for _p in (_HERE, _REPO, os.path.join(_REPO, 'src'), os.path.join(_REPO, 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from hagi.config import load_config
from hagi.data.dataset import build_dataloader
from hagi.model.factory import build_model_for_config
from hagi.train.checkpoint import config_from_dict, load_payload
from hagi.train.loop import configure_runtime, train
from train import setup_logging

from lora_leaf_ab import _LoRALookup, _LoRAProjection  # noqa: E402

PRIOR = "checkpoints/dbridge_gen2_merged_had_c8/step-0001600.pt"
LORA_D2 = "checkpoints/dbridge_gen2_lora_c8_d2/step-0001600.pt"
BASE_CFG = "configs/dbridge_gen2_merged_had_c8.yaml"
RANK = 32


def widen(v: torch.Tensor, old_rank: int, rank: int) -> torch.Tensor:
    """Zero-extend the rank axis (first or last, wherever dim==old_rank)."""
    if v.shape[0] == old_rank:
        out = v.new_zeros((rank,) + tuple(v.shape[1:]))
        out[:old_rank] = v
        return out
    if v.shape[-1] == old_rank:
        out = v.new_zeros(tuple(v.shape[:-1]) + (rank,))
        out[..., :old_rank] = v
        return out
    raise ValueError(f"cannot locate rank axis {old_rank} in {tuple(v.shape)}")


def build_wide_prior() -> torch.nn.Module:
    configure_runtime()
    pl = load_payload(PRIOR, "cpu")
    model = build_model_for_config(config_from_dict(pl["config"]))
    model.load_state_dict(pl["model"], strict=True)
    emb = model.encoder.embedding.weight.data.clone()
    model.encoder.embedding = _LoRALookup(emb, RANK)
    head_w = model.head.projection.weight.data.clone()
    model.head.projection = _LoRAProjection(head_w, RANK)

    plora = torch.load(LORA_D2, map_location="cpu", weights_only=False)
    sd = plora["model"]
    wide = {}
    for k, v in sd.items():
        if ".lora.A" in k or ".lora.B" in k:
            wide[k] = widen(v.float(), 16, RANK)
        else:
            wide[k] = v
    model.load_state_dict(wide, strict=True)
    return model.to("cuda").to(torch.bfloat16)


def main() -> int:
    model = build_wide_prior()
    model.encoder.embedding.lora.B.data = model.encoder.embedding.lora.B.data.float()
    model.head.projection.lora.B.data = model.head.projection.lora.B.data.float()
    cfg = load_config(BASE_CFG)
    cfg.train.checkpoint_dir = "checkpoints/dbridge_gen2_lora_c8_r32"
    cfg.train.max_steps = 1600
    cfg.train.data.seed = 14159  # fresh, unseen by 12801/13837
    log_path = setup_logging(cfg.train.checkpoint_dir)
    print(f"logging to {log_path}", flush=True)
    dataloader = build_dataloader(cfg, cfg.train.data.data_dir, start_offset=0)
    last = None
    for metrics in train(model, dataloader, cfg, start_step=0):
        last = metrics
    print("done:", last)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
