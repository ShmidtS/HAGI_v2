"""Byte-GDN line training run (target architecture, autonomous).

Raw bytes (vocab 256) -> cosine-router H-Net chunker -> GDN delta-rule
backbone -> EMA dechunker -> per-byte LM head. Certified training loop
(SafeQP gates, Muon-capable) from hagi.train.loop.

    python scripts/train_byte_gdn.py --steps 2000 --hidden 256 --layers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-chunk", type=int, default=4)
    ap.add_argument("--max-chunk", type=int, default=12)
    ap.add_argument("--data", type=str, default="data_bytes")
    ap.add_argument("--start-frac", type=float, default=0.0, help="fresh-data rotation offset [0,1)")
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints/byte_gdn_v1")
    ap.add_argument("--init", type=str, default="", help="checkpoint to warm-start from")
    args = ap.parse_args()

    from hagi.config import Config, validate_config
    from hagi.data.dataset import build_dataloader
    from hagi.model.model import HAGI
    from hagi.train.checkpoint import save_checkpoint
    from hagi.train.loop import configure_runtime, train

    configure_runtime()
    cfg = Config()
    cfg.model.vocab_size = 256
    cfg.model.hidden_size = args.hidden
    cfg.model.num_layers = args.layers
    cfg.model.attention.num_query_heads = args.heads
    cfg.model.attention.num_kv_heads = max(1, args.heads // 2)
    cfg.model.attention.head_dim = args.hidden // args.heads
    cfg.model.input_mode = "bytes_hnet"
    cfg.model.backbone = "gdn"
    cfg.model.gdn.n_heads = 1
    cfg.model.byte_hnet.min_chunk = args.min_chunk
    cfg.model.byte_hnet.max_chunk = args.max_chunk
    cfg.train.data.data_dir = args.data
    cfg.train.data.seq_len = args.seq_len
    cfg.train.batch_size = args.batch_size
    cfg.train.max_steps = args.steps
    cfg.train.learning_rate = args.lr
    cfg.train.precision = "bf16"
    cfg.train.use_muon = False  # AdamW first: the line is brand new
    cfg.train.grad_checkpointing = True
    cfg.train.checkpoint_dir = args.ckpt_dir
    cfg.train.checkpoint_interval = max(100, args.steps // 10)
    cfg.train.checkpoint_keep_last = 3
    validate_config(cfg)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HAGI(cfg).to(device)
    if args.init:
        from hagi.train.checkpoint import load_model

        load_model(args.init, model, device, lenient_config=True)
        print(f"[byte-gdn] warm start from {args.init}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[byte-gdn] params {n_params/1e6:.2f}M, device {device}", flush=True)

    dl = build_dataloader(cfg, start_frac=args.start_frac)
    from hagi.train.checkpoint import save_checkpoint

    best_ce = float("inf")
    t0 = time.time()
    for metrics in train(model, dl, cfg):
        step = metrics.get("step", -1)
        ce = metrics.get("ce")
        if ce is not None and ce < best_ce:
            best_ce = ce
            save_checkpoint(
                model, cfg, max(0, int(step)), args.ckpt_dir, keep_last=1, name="best"
            )
        if step % 50 == 0:
            print(
                f"[byte-gdn] step {step} ce {ce} elapsed {time.time()-t0:.0f}s",
                flush=True,
            )
    with open(Path(args.ckpt_dir) / "final.json", "w", encoding="utf-8") as fh:
        json.dump({"best_ce": best_ce, "steps": args.steps, "params_M": n_params / 1e6}, fh)
    print(f"[byte-gdn] done, best_ce {best_ce}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
