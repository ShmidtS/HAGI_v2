"""Merge N expert checkpoints into one block-diagonal MergedHAGI.

Usage:
    python scripts/merge_experts.py --config configs/v47_merged.yaml \
        --experts ckptA.pt ckptB.pt ckptC.pt ckptD.pt \
        --out checkpoints_v47_merged/step-0000000.pt

The merged model is written as a standard HAGI checkpoint (format 12) so it can
be resumed with ``scripts/train.py --config configs/v47_merged.yaml --resume``.

When ``--experts`` is omitted, the current model's weights are replicated N
times (machinery smoke test for the merge path).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.config import CHECKPOINT_FORMAT_VERSION, load_config  # noqa: E402
from hagi.model.merge import MergedHAGI, merge_experts  # noqa: E402
from hagi.train.checkpoint import config_to_dict, load_payload  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge N expert checkpoints block-diagonally")
    parser.add_argument("--config", default="configs/v47_merged.yaml")
    parser.add_argument("--experts", nargs="+", default=None, help="N expert step-*.pt paths")
    parser.add_argument("--out", default=None, help="output checkpoint path")
    parser.add_argument("--n-mixers", type=int, default=1)
    parser.add_argument("--mixer-init-scale", type=float, default=0.0)
    parser.add_argument("--drop-expert-mixers", action="store_true",
                        help="ignore mixers.* in expert states (hierarchical merge)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n = cfg.merge.n_experts
    if args.experts is not None and len(args.experts) != n:
        raise SystemExit(f"--experts must have exactly {n} paths, got {len(args.experts)}")

    if args.experts is not None:
        states = []
        for path in args.experts:
            payload = load_payload(path, args.device)
            states.append(payload["model"])
            print(f"expert {path}: completed_steps={payload['completed_steps']}")
    else:
        # Replicate a *narrow* expert's weights N times (machinery smoke test).
        # The merge expects expert states of shape [V, expert_hidden]; build a
        # narrow HAGI with the expert geometry and replicate it.
        from hagi.config import Config
        from hagi.model.model import HAGI

        eh = cfg.merge.expert_hidden
        n = cfg.merge.n_experts
        expert_cfg = Config()
        expert_cfg.model.vocab_size = cfg.model.vocab_size
        expert_cfg.model.hidden_size = eh
        expert_cfg.model.num_layers = cfg.model.num_layers
        expert_cfg.model.attention.num_query_heads = cfg.model.attention.num_query_heads // n
        expert_cfg.model.attention.num_kv_heads = cfg.model.attention.num_kv_heads // n
        expert_cfg.model.attention.head_dim = cfg.model.attention.head_dim
        expert_cfg.model.attention.rope_theta = cfg.model.attention.rope_theta
        expert_cfg.model.attention.max_seq_len = cfg.model.attention.max_seq_len
        expert_cfg.model.attention.qk_norm = cfg.model.attention.qk_norm
        expert_cfg.model.embedding.tie_lm_head = cfg.model.embedding.tie_lm_head
        expert_cfg.model.embedding.conv_kernel = cfg.model.embedding.conv_kernel
        expert_cfg.model.embedding.init_std = cfg.model.embedding.init_std
        expert_cfg.model.ffn.expansion = cfg.model.ffn.expansion
        expert_cfg.model.ffn.multiple_of = cfg.model.ffn.multiple_of
        expert_cfg.model.ternary.enabled = cfg.model.ternary.enabled
        expert_cfg.model.head = cfg.model.head
        expert_cfg.model.init_orthogonal = cfg.model.init_orthogonal
        base = HAGI(expert_cfg).to(args.device)
        base_sd = base.state_dict()
        states = [dict(base_sd) for _ in range(n)]
        print(f"no --experts: replicating narrow expert (H={eh}) weights {n} times (smoke test)")

    model = merge_experts(
        cfg,
        states,
        n_mixers=args.n_mixers,
        mixer_init_scale=args.mixer_init_scale,
        drop_expert_mixers=args.drop_expert_mixers,
    )
    model = model.to(args.device)
    counts = model.param_summary()
    print(
        f"merged: total {counts['total']/1e6:.1f}M | body {counts['body']/1e6:.1f}M | "
        f"embed {counts['embedding']/1e6:.1f}M | mixers {sum(p.numel() for p in model.mixers.parameters())/1e6:.2f}M"
    )

    out = args.out or f"{cfg.train.checkpoint_dir}/step-0000000.pt"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "config": config_to_dict(cfg),
        "completed_steps": 0,
    }
    torch.save(payload, out)
    print(f"wrote merged checkpoint: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
