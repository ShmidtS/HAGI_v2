"""Full-depth TTT: features -> delta LoRA on the real 27B stack.

Why not scripts/e4_ttt_real_dims.py: that harness drives both contours through
``self_improve``, which refuses ``model.loop_depth == 1`` ("repeated adapter
input"). The real Ternary-Bonsai-2-27B ships loop_depth=1 (64 distinct layers,
no weight-tied repeats), and loop_depth=2 at 64 layers is ~136 GiB against a
107.87 GiB card. So the contour comparison is a harness property, not a model
limit -- and the only way to measure the shipped depth is to drive the
primitive directly.

What this measures is the thing the user asked about: features -> adapter delta
without a second full generation. Each iteration is one teacher-forced pass over
already-known tokens, per-block ridge-solve, and a bounded apply. Reported per
iteration: wall time, blocks updated, achieved step size, and peak memory.

Memory is built directly on the gpu in bf16 (see adapters._qr_orthonormal for
why that is a precondition on a 31.6 GiB-RAM host), and the run refuses to start
past 75% of free gpu.
"""
import argparse
import sys
import time

sys.path.insert(0, "C:/HAGI_v2/src")
sys.path.insert(0, "C:/HAGI_v2")

import torch

from hagi.config import Config, validate_config
from hagi.model.model import HAGI
from hagi.train.ttt import TttRls

# Ternary-Bonsai-2-27B, from models/ternary-bonsai-2-27b-mtp/donor/config.json
# (text_config) and mtp_config.json. q_heads is 20, not the real 24: the donor
# q_proj is [12288, 5120] = 2 x (24 x 256), i.e. gated attention with a 6144-wide
# attention output, which src/hagi/model/attention.py:175 cannot express. 20x256
# = 5120 keeps hidden/FFN/head exact; attention cost is ~17% under the real layer.
REAL = dict(hidden=5120, vocab=248320, inter=17408, q_heads=20, kv_heads=4,
            head_dim=256, rank=8)
GIB = 2**30


def cfg_for(layers: int, loop: int) -> Config:
    c = Config()
    m = c.model
    m.hidden_size = REAL["hidden"]
    m.num_layers = layers
    m.vocab_size = REAL["vocab"]
    m.attention.num_query_heads = REAL["q_heads"]
    m.attention.num_kv_heads = REAL["kv_heads"]
    m.attention.head_dim = REAL["head_dim"]
    m.ffn.intermediate_size = REAL["inter"]
    m.init_orthogonal = False
    m.adapters.enabled = True
    m.adapters.pyramid.enabled = False
    m.adapters.ttt_lora.enabled = True
    m.adapters.ttt_lora.rank = REAL["rank"]
    m.loop_depth = loop
    c.train.adapt.freeze_base = True
    c.train.precision = "bf16"
    c.train.grad_accum_steps = 1
    c.train.batch_size = 1
    c.train.logging.exact_ce_interval = 0
    c.train.weight_decay = 0.0
    c.train.schedule.warmup_steps = 0
    c.train.max_steps = 4
    validate_config(c)
    return c


def build(cfg: Config) -> HAGI:
    """Construct directly on the gpu in bf16, or refuse to try."""
    free, _ = torch.cuda.mem_get_info()
    h = cfg.model.hidden_size
    per_layer = 4 * h * h + 3 * cfg.model.ffn.intermediate_size * h
    proj = (per_layer * cfg.model.num_layers + 2 * cfg.model.vocab_size * h) * 2 / GIB
    if proj > free * 0.75:
        raise SystemExit(
            f"refusing to build: projected bf16 footprint {proj:.1f} GiB exceeds "
            f"75% of {free / GIB:.1f} GiB free gpu"
        )
    prev = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            return HAGI(cfg)
    finally:
        torch.set_default_dtype(prev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[64])
    ap.add_argument("--tokens", type=int, default=512,
                    help="teacher-forced window per iteration. refit_rows=64 "
                         "means a short window solves on only some steps, so a "
                         "production-like length is what makes updates visible.")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--loop-depth", type=int, default=1)
    ap.add_argument("--refit-rows", type=int, default=64)
    args = ap.parse_args()

    print(f"gpu free: {torch.cuda.mem_get_info()[0] / GIB:.1f} GiB")
    print(f"real dims: {REAL}\n")
    for layers in args.layers:
        for seed in range(args.seeds):
            cfg = cfg_for(layers, args.loop_depth)
            torch.manual_seed(seed)
            t0 = time.perf_counter()
            model = build(cfg)
            build_s = time.perf_counter() - t0
            params = sum(p.numel() for p in model.parameters())
            ttt = TttRls(model, stream_frac=0.02, refit_rows=args.refit_rows)
            ids = torch.randint(1, cfg.model.vocab_size, (1, args.tokens),
                                device="cuda")
            tgt = torch.randint(1, cfg.model.vocab_size, (1, args.tokens),
                                device="cuda")
            torch.cuda.reset_peak_memory_stats()
            rows = []
            for k in range(args.iters):
                t1 = time.perf_counter()
                s = ttt.step(ids, tgt)
                rows.append((time.perf_counter() - t1, s.blocks_updated,
                             s.delta_rms_frac, s.ce, s.refits))
            peak = torch.cuda.max_memory_allocated() / GIB
            hdr = (f"{'layers':>7} {'params':>8} {'loop':>5} {'tok':>5} "
                   f"{'ms/iter':>9} {'upd/iter':>9} {'frac':>9} {'peak':>7}")
            print(hdr)
            print("-" * len(hdr))
            ms = sum(r[0] for r in rows) / len(rows) * 1000
            ups = sum(r[1] for r in rows) / len(rows)
            fr = max(r[2] for r in rows)
            print(f"{layers:>7} {params / 1e9:>7.1f}B {args.loop_depth:>5} "
                  f"{args.tokens:>5} {ms:>9.0f} {ups:>9.2f} {fr:>9.2e} "
                  f"{peak:>6.1f}G")
            ces = [r[3] for r in rows]
            print(f"        ce: {ces[0]:.4f} -> {ces[-1]:.4f} "
                  f"({ces[-1] - ces[0]:+.4f} over {len(rows)} iters)  "
                  f"refits={sum(r[4] for r in rows)}  "
                  f"build={build_s:.0f}s")
            del model, ttt
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
