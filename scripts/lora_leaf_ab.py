"""Round-22 A/B: low-rank table element vs full tables (Element.lean).

Arm A (reference, round 19): full-table leaves from the shared E0
  (init_from, everything trainable) -- measured flat3 = 5.875.
Arm B (this experiment): E0 FROZEN tables + TableLoRA rank-r deltas
  (embedding + head) + the body fully trainable. The seed variance
  enters through r*(V+H) delta params + the body instead of 2VH.

Measured question: how much of the leaf's specialization actually
needs full-rank table freedom? The round-19 spectra said the per-seed
residual is ~35% of the delta -- if rank 16-32 captures the useful
part, Arm B matches Arm A at 2.5x+ table compression.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hagi.config import load_config  # noqa: E402
from hagi.model.adaptive import freeze_base_in_place  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.model.table_lora import TableLoRA  # noqa: E402
from hagi.train.checkpoint import config_from_dict, load_payload  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402


def build_lora_leaf(leaf_cfg_path: str, prior_ckpt: str, rank: int,
                    device: str = "cuda") -> HAGI:
    """A leaf whose embedding/head are TableLoRA over the FROZEN prior."""
    configure_runtime()
    cfg = load_config(leaf_cfg_path)
    pl = load_payload(prior_ckpt, "cpu")
    prior_sd = pl["model"]
    model = HAGI(config_from_dict(pl["config"])).to("cpu")
    model.load_state_dict(prior_sd, strict=True)
    # replace the tables with frozen-base low-rank residuals
    emb = model.encoder.embedding.weight.data.clone()
    model.encoder.embedding = _LoRALookup(emb, rank)
    head_w = model.head.projection.weight.data.clone()
    model.head.projection = _LoRAProjection(head_w, rank)
    return model.to(device)


class _LoRALookup(torch.nn.Module):
    """Embedding lookup that never materializes the effective table.

    out = W0[ids] + (A[ids] @ B): the frozen base is a cheap index
    gather and only the [T, r] x [r, H] delta matmul runs per
    forward -- O(T*(rH)) instead of O(V*H) per step.
    """

    def __init__(self, table: torch.Tensor, rank: int) -> None:
        super().__init__()
        self.lora = TableLoRA(table, rank)
        self.vocab_size, self.hidden_size = table.shape

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        base = self.lora.base[ids]                    # [T, H] gather
        a_sel = self.lora.A[ids]                      # [T, r]
        return base + a_sel @ self.lora.B              # [T, r] @ [r, H]


class _LoRAProjection(torch.nn.Module):
    """Head projection over the frozen base + low-rank delta.

    Exposes ``.weight`` = effective table (the LMHead contract), so
    ``F.linear`` and the fused-CE paths work unchanged; the cost is
    one O(V*r + r*H) materialization per forward on top of the same
    head GEMM.
    """

    def __init__(self, table: torch.Tensor, rank: int) -> None:
        super().__init__()
        self.lora = TableLoRA(table, rank)

    @property
    def weight(self) -> torch.Tensor:
        return self.lora.effective_weight()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf-config", required=True)
    ap.add_argument("--prior", default="checkpoints/leafdat_2xcorp/step-0001600.pt")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--train", action="store_true", help="train a LoRA leaf")
    ap.add_argument("--out-dir", default="checkpoints/lora_leaf_r16")
    ap.add_argument("--steps", type=int, default=1600)
    args = ap.parse_args()
    if args.train:
        res = train_lora_leaf(args.leaf_config, args.prior, args.rank,
                              args.out_dir, args.steps)
        print("done:", res)
        return 0
    model = build_lora_leaf(args.leaf_config, args.prior, args.rank)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    lora_params = [n for n, p in model.named_parameters()
                   if p.requires_grad and ".lora." in n]
    print("lora (adaptive) params:", len(lora_params),
          "| total trainable:", len(trainable))
    # The TableLoRA bases are BUFFERS, so the body stays trainable
    # (arm B design: frozen tables + trainable body + rank-r deltas).
    live = [n for n, p in model.named_parameters() if p.requires_grad]
    n_body = len([n for n in live if ".lora." not in n])
    n_lora = len([n for n in live if ".lora." in n])
    print(f"trainable: {n_body} body params + {n_lora} table-LoRA Bs")
    for n in live[:6]:
        print("  live:", n)
    return 0




def train_lora_leaf(leaf_cfg_path: str, prior_ckpt: str, rank: int,
                    out_dir: str, steps: int = 1600, device: str = "cuda") -> dict:
    """Train one TableLoRA leaf from the shared prior."""
    import numpy as np

    from hagi.train.loop import train, configure_runtime as _cr

    _cr()
    model = build_lora_leaf(leaf_cfg_path, prior_ckpt, rank, device)
    # cast to bf16 like the canonical recipe (the LoRA bases are already
    # the prior's bf16-able weights; the body follows the config)
    cfg = load_config(leaf_cfg_path)
    cfg.train.checkpoint_dir = out_dir
    cfg.train.max_steps = steps
    # dataloader on the canonical mix
    from hagi.data.dataset import build_dataloader  # noqa: E402

    dataloader = build_dataloader(cfg, cfg.train.data.data_dir, start_offset=0)
    import torch as _t
    model = model.to(device).to(_t.bfloat16)
    # keep the LoRA B in fp32 (a small tensor steering a frozen table
    # must not quantize to bf16 steps)
    model.encoder.embedding.lora.B.data = model.encoder.embedding.lora.B.data.float()
    model.head.projection.lora.B.data = model.head.projection.lora.B.data.float()
    last = None
    for metrics in train(model, dataloader, cfg, start_step=0):
        last = metrics
    return {"last_metrics": last}


if __name__ == "__main__":
    raise SystemExit(main())
