"""Round 50: rank-channel deepening — continue the r16 adapters of the
ABSOLUTE RECORD (dbridge_gen2_lora_c8, gate 3.3928) for another 1600
steps on a fresh data seed.

Blind prediction (recorded in .omc/attempts/cycle_decay_round50.md BEFORE
launch): gain in [0.02, 0.03] nat. Verdict rule: gain < 0.5*prediction
-> rank channel exhausted at r16+3200 steps, switch channel; gain >=
prediction -> continue deepening (r=32 next).

Same protocol as lora_gen2_joint_c8 (bf16 body, LoRA B in fp32 — the
bf16-frozen-gain law, round 35c), fresh data seed 13837.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "../../.."))
for _p in (_HERE, _REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from hagi.config import load_config
from hagi.data.dataset import build_dataloader
from hagi.model.factory import build_model_for_config
from hagi.train.checkpoint import config_from_dict, load_payload
from hagi.train.loop import configure_runtime, train

from lora_gen2_joint_c8 import _LoRALookup, _LoRAProjection  # noqa: E402

BASE_CFG = "configs/dbridge_gen2_merged_had_c8.yaml"
PRIOR = "checkpoints/dbridge_gen2_merged_had_c8/step-0001600.pt"
LORA_CKPT = "checkpoints/dbridge_gen2_lora_c8/step-0001600.pt"


def build_stacked_prior() -> torch.nn.Module:
    """Clamp-8 base + TRAINED r16 adapters (the record) — step 0 of this
    run is bit-identical to the record checkpoint."""
    configure_runtime()
    pl = load_payload(PRIOR, "cpu")
    model = build_model_for_config(config_from_dict(pl["config"]))
    model.load_state_dict(pl["model"], strict=True)
    emb = model.encoder.embedding.weight.data.clone()
    model.encoder.embedding = _LoRALookup(emb, 16)
    head_w = model.head.projection.weight.data.clone()
    model.head.projection = _LoRAProjection(head_w, 16)

    plora = load_payload(LORA_CKPT, "cpu")
    model.load_state_dict(plora["model"], strict=True)
    return model.to("cuda").to(torch.bfloat16)


def main() -> int:
    model = build_stacked_prior()
    # LoRA B in fp32 (bf16-frozen-gain law, round 35c)
    model.encoder.embedding.lora.B.data = model.encoder.embedding.lora.B.data.float()
    model.head.projection.lora.B.data = model.head.projection.lora.B.data.float()
    cfg = load_config(BASE_CFG)
    cfg.train.checkpoint_dir = "checkpoints/dbridge_gen2_lora_c8_d2"
    cfg.train.max_steps = 1600
    cfg.train.data.seed = 13837  # fresh seed, unseen by r16 pass (12801)
    dataloader = build_dataloader(cfg, cfg.train.data.data_dir, start_offset=0)
    last = None
    for metrics in train(model, dataloader, cfg, start_step=0):
        last = metrics
    print("done:", last)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
