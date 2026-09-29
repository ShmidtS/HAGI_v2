"""Round 36: TableLoRA on the gen-2 joint (rank channel per DesignOpt).

Freezes the gen-2 joint (H=1152, gate 3.4484) and trains rank-16 table
deltas + body, per the ComputeBudget argmax (rank channel had the best
measured marginal value; movement channel decayed -0.35 -> -0.092).

Reuses scripts/lora_leaf_ab.py's _LoRALookup/_LoRAProjection classes;
the only difference is model construction through the factory so a
MergedHAGI payload (mixers.*) loads correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from hagi.config import load_config
from hagi.data.dataset import build_dataloader
from hagi.model.factory import build_model_for_config
from hagi.train.checkpoint import config_from_dict, load_payload
from hagi.train.loop import configure_runtime, train

from lora_leaf_ab import _LoRALookup, _LoRAProjection  # noqa: E402


def build_lora_prior(joint_cfg_path: str, prior_ckpt: str, rank: int,
                     device: str = "cuda"):
    configure_runtime()
    cfg = load_config(joint_cfg_path)
    pl = load_payload(prior_ckpt, "cpu")
    model = build_model_for_config(config_from_dict(pl["config"]))
    model.load_state_dict(pl["model"], strict=True)
    emb = model.encoder.embedding.weight.data.clone()
    model.encoder.embedding = _LoRALookup(emb, rank)
    head_w = model.head.projection.weight.data.clone()
    model.head.projection = _LoRAProjection(head_w, rank)
    return model.to(device).to(torch.bfloat16)


def main() -> int:
    model = build_lora_prior(
        "configs/dbridge_gen2_joint.yaml",
        "checkpoints/dbridge_gen2_joint/step-0001600.pt", 16)
    # LoRA B in fp32 (bf16-frozen-gain law, round 35c)
    model.encoder.embedding.lora.B.data = model.encoder.embedding.lora.B.data.float()
    model.head.projection.lora.B.data = model.head.projection.lora.B.data.float()
    cfg = load_config("configs/dbridge_gen2_joint.yaml")
    cfg.train.checkpoint_dir = "checkpoints/dbridge_gen2_lora"
    cfg.train.max_steps = 1600
    cfg.train.data.seed = 12801
    dataloader = build_dataloader(cfg, cfg.train.data.data_dir, start_offset=0)
    last = None
    for metrics in train(model, dataloader, cfg, start_step=0):
        last = metrics
    print("done:", last)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
