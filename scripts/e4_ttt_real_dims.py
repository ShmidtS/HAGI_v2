"""Transfer check: does the features->delta speedup survive at real 27B shapes?

The tiny-model measurement counted FORWARD PASSES, which are shape-independent
(19/3 = 6.33x asymptote). Wall-clock is not: at hidden=5120 / vocab=248320 the
head materialises [1,T,248320] logits and every decode step is bound by weight
reads. So the ratio has to be re-measured at the real shapes before anyone
quotes 6.33x for the 27B.

Same harness as the tiny measurement (forward hook on model.encoder, both
contours driven through si.self_improve), only the dims change. Layers is
swept because the head cost is fixed while the stack cost scales, so a
small-layer count would over-credit the head and distort the ratio.
"""
import argparse
import sys
import time

sys.path.insert(0, "C:/HAGI_v2/src")
sys.path.insert(0, "C:/HAGI_v2")

import torch

from hagi.config import Config, validate_config
from hagi.model.model import HAGI
from hagi.train import self_improve as si
from hagi.train.ttt import TttRls

# models/ternary-bonsai-2-27b-mtp/{mtp_config,donor/config.json}.text_config
# Attention note: the real model is gated -- q_proj is [12288, 5120] = 2 x
# (24 x 256), so its attention output dim (6144) does not equal hidden_size
# (5120). HAGI forbids that in attention.py:175, not only in the validator, so
# the exact head split is inexpressible. The dominant cost drivers stay exact:
# hidden 5120, FFN 3 x 17408 x 5120, head 248320 x 5120, and kv 4 x 256 = 1024
# matches the real k_proj [1024, 5120]. Query heads drop to 20 x 256 = 5120 to
# satisfy the invariant, understating attention by ~17% of its own cost.
REAL = dict(hidden=5120, vocab=248320, inter=17408, q_heads=20,
            kv_heads=4, head_dim=256, rank=8)
N_NEW = 16
LOOP_DEPTH = 2   # overridden by --loop-depth; the real 27B ships 1
ITERS = 6


def cfg_for(layers: int) -> Config:
    c = Config()
    m = c.model
    m.hidden_size = REAL["hidden"]
    m.num_layers = layers
    m.vocab_size = REAL["vocab"]
    m.attention.num_query_heads = REAL["q_heads"]
    m.attention.num_kv_heads = REAL["kv_heads"]
    m.attention.head_dim = REAL["head_dim"]
    m.ffn.intermediate_size = REAL["inter"]
    m.init_orthogonal = False          # one-time QR over 27B params is not free
    m.adapters.enabled = True
    m.adapters.pyramid.enabled = False
    m.adapters.ttt_lora.enabled = True
    m.adapters.ttt_lora.rank = REAL["rank"]
    c.model.loop_depth = LOOP_DEPTH  # matches the tiny-model baseline harness;
                                   # self_improve's gradient contour needs >1
                                   # (repeated adapter input); params stay at
                                   # num_layers while effective depth doubles
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


def _build_on_gpu(cfg: Config) -> HAGI:
    """Construct the model directly on the gpu in bf16, and refuse to oversubscribe.

    Not an optimization -- on this host it is the only order that works. RAM is
    31.6 GiB, and the previous ``HAGI(cfg).to("cuda", bf16)`` held the whole
    fp32 build on the host first: 44.1 GiB at 32 layers, 83.5 GiB at 64, either
    of which pages into a crash before the gpu sees a weight. The 64-layer run
    that killed a machine did exactly this.

    Building under a ``cuda`` + bf16 ambient needs ``_qr_orthonormal`` to be
    device-aware (it pins its draw to cpu fp32 and moves the basis after),
    which is why ``adapters.py`` reads the ambient device before pinning.
    """
    free, _ = torch.cuda.mem_get_info()
    gib = 2**30
    # Params dominate; estimate from the config instead of building twice.
    h = cfg.model.hidden_size
    per_layer = (
        4 * h * h                                  # q/k/v/o at these dims
        + 3 * cfg.model.ffn.intermediate_size * h  # gate/up/down
    )
    proj = (per_layer * cfg.model.num_layers + 2 * cfg.model.vocab_size * h) * 2 / gib
    if proj > free * 0.75:
        raise SystemExit(
            f"refusing to build: projected bf16 footprint {proj:.1f} GiB exceeds "
            f"75% of {free / gib:.1f} GiB free gpu memory"
        )
    prev = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            return HAGI(cfg)
    finally:
        torch.set_default_dtype(prev)


def run(mode: str, layers: int, seed: int) -> dict:
    cfg = cfg_for(layers)
    torch.manual_seed(seed)
    t_build = time.perf_counter()
    model = _build_on_gpu(cfg)
    build = time.perf_counter() - t_build
    calls = {"n": 0}

    def _hit(module, args):
        calls["n"] += 1

    handle = model.encoder.register_forward_pre_hook(_hit)
    ttt = TttRls(model, stream_frac=0.5, refit_rows=8, max_delta_rms_frac=1.0)
    prompt = [1, 2, 3, 4]
    kw = dict(n_new_tokens=N_NEW, max_iterations=ITERS, patience=ITERS + 1)
    t0 = time.perf_counter()
    if mode == "rls":
        stats = si.self_improve(model, cfg, prompt, mode="rls", ttt=ttt, **kw)
    else:
        stats = si.self_improve(model, cfg, prompt, **kw)
    dt = time.perf_counter() - t0
    handle.remove()
    n = max(len(stats.iterations), 1)
    params = sum(p.numel() for p in model.parameters())
    del model, ttt
    torch.cuda.empty_cache()
    return {"mode": mode, "iters": n, "fw": calls["n"] / n,
            "ms": dt / n * 1e3, "params": params, "build": build,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[8])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--modes", nargs="+", default=["gradient", "rls"],
                    help="subset of contours; rls-only isolates the layer-count "
                         "limit from the contour comparison")
    ap.add_argument("--loop-depth", type=int, default=2,
                    help="model.loop_depth. The harness baseline is 2; the real "
                         "27B ships 1 (64 distinct layers, no weight-tied "
                         "repeats). Full depth only fits at 1: at 2 the harvest "
                         "graph covers 128 block passes and OOMs at ~136 GiB "
                         "against the 86.2 GiB measured at 1.")
    args = ap.parse_args()
    global LOOP_DEPTH
    LOOP_DEPTH = args.loop_depth
    print(f"device free: {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB")
    print(f"real dims: {REAL}\n")
    for layers in args.layers:
        agg = {}
        for seed in range(args.seeds):
            for mode in args.modes:
                r = run(mode, layers, seed)
                a = agg.setdefault(mode, {"iters": 0, "fw": 0.0, "ms": 0.0})
                a["iters"] += r["iters"]
                a["fw"] += r["fw"] * r["iters"]
                a["ms"] += r["ms"] * r["iters"]
                peak, params, build = r["peak_gib"], r["params"], r["build"]
        print(f"--- layers={layers}  params={params / 1e9:.1f}B  "
              f"build={build:.0f}s  peak={peak:.1f} GiB ---")
        hdr = f"{'mode':>9} {'iters':>6} {'fwd/iter':>9} {'ms/iter':>10}"
        print(hdr)
        print("-" * len(hdr))
        for mode in args.modes:
            a = agg[mode]
            print(f"{mode:>9} {a['iters']:>6} {a['fw'] / a['iters']:>9.2f} "
                  f"{a['ms'] / a['iters']:>10.0f}")
        if len(args.modes) != 2:
            print()
            continue
        g, r = agg[args.modes[0]], agg[args.modes[1]]
        fw_ratio = (g["fw"] / g["iters"]) / (r["fw"] / r["iters"])
        ms_ratio = (g["ms"] / g["iters"]) / (r["ms"] / r["iters"])
        print(f"\n  forwards ratio: {fw_ratio:.2f}x   wallclock ratio: "
              f"{ms_ratio:.2f}x   (tiny-model forwards asymptote: 6.33x)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
