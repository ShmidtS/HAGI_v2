"""Gen7 rank channel: TableLoRA r=16 on the gen6 joint (ComputeBudget argmax).

The gen7 growth gate (scripts/growth/gen7_gate.py, §AU) returned GROW:
G_F = 0.575 nats above eps_g = 0.182 -- free energy is high -- with
R_repr = 0.051 well above eps_r = 0.0008, so the LoRA/TTT cell is NOT
licensed and the mechanism is picked INSIDE grow by ComputeBudget
argmax DeltaG_F/DeltaC. The rank channel is the cheapest mechanism
with a measured marginal (0.196 nat at r=16, era-2 ledgers; the joint
cycle's 0.23-0.35 costs three sibling trainings + a merge + a joint
run): this script spends that cheap channel first, then the next
gate pass decides whether the expensive merge cycle is still needed.

Reuses lora_leaf_ab.py's _LoRALookup/_LoRAProjection (the classes are
generation-agnostic: they wrap whatever tables the checkpoint carries;
gen6_joint is a MergedHAGI with wide [V, 3456] tables, built through
the factory so mixers.* load correctly).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (ROOT / "src", ROOT / "scripts", ROOT / "scripts" / "lora"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402
from lora_leaf_ab import _LoRALookup, _LoRAProjection  # noqa: E402

from hagi.config import load_config  # noqa: E402
from hagi.data.dataset import build_dataloader  # noqa: E402
from hagi.model.factory import build_model_for_config  # noqa: E402
from hagi.train.checkpoint import config_from_dict, load_payload  # noqa: E402
from hagi.train.loop import configure_runtime, train  # noqa: E402

JOINT_CFG = "configs/dbridge_gen6_joint.yaml"
JOINT_CKPT = "checkpoints/dbridge_gen6_joint/step-0001300.pt"
RANK = 16


def main() -> int:
    configure_runtime()
    cfg = load_config(JOINT_CFG)
    pl = load_payload(JOINT_CKPT, "cpu")
    model = build_model_for_config(config_from_dict(pl["config"]))
    model.load_state_dict(pl["model"], strict=True)
    emb = model.encoder.embedding.weight.data.clone()
    model.encoder.embedding = _LoRALookup(emb, RANK)
    head_w = model.head.projection.weight.data.clone()
    model.head.projection = _LoRAProjection(head_w, RANK)
    model = model.to("cuda").to(torch.bfloat16)
    # LoRA B in fp32 (bf16-frozen-gain law, round 35c)
    model.encoder.embedding.lora.B.data = model.encoder.embedding.lora.B.data.float()
    model.head.projection.lora.B.data = model.head.projection.lora.B.data.float()

    cfg.train.checkpoint_dir = "checkpoints/dbridge_gen7_lora"
    cfg.train.max_steps = 800
    cfg.train.data.seed = 14101
    dataloader = build_dataloader(cfg, cfg.train.data.data_dir, start_offset=0)
    last = None
    for metrics in train(model, dataloader, cfg, start_step=0):
        last = metrics
    print("done:", last)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
