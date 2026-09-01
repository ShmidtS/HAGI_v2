"""Sequential e2e collector: activations for ONE target layer, collected
through the ALREADY-COMPRESSED prefix (I4X hooks active for layers < L).

For the target layer L we capture:
  x_k = drifted input rows (through compressed 0..L-1)   -> fit target input
  y_k = ORIGINAL expert output (teacher, FP4-dequant)     -> fit target output
Layers > L run original compute (no capture). The result feeds the refit
for layer L, which then absorbs the accumulated prefix drift (telescoping).

Usage:
  SEQ_LAYER=1 I4X_LAYERS=0 python scripts/dsv4_collect_seq.py \
      [--max-tokens 258560] [--cap 2048] [--out-dir checkpoints_dsv4/seq]

Output: {out-dir}/acts_layer{L}.pt  {k: (x_k, y_k)} bf16.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import stub_import_tf  # noqa: F401
import dsv4_experts as de
from dsv4_experts import (
    HASH_LAYERS,
    LOSSLESS,
    N_LAYERS,
    ROUTED_SCALE,
    ROUTER_BIAS,
    ROUTER_TID,
    ROUTER_W,
    SWIGLU_LIMIT,
    TOP_K,
    ffn,
    load_router,
)
import dsv4_generate_ttt as gen  # compressed-expert machinery
from dsv4_generate_ttt import get_dequant, get_int4x, int4x_forward, int4x_forward_z

SEQ_LAYERS = set(int(x) for x in os.environ.get("SEQ_LAYERS", "").split(",") if x != "")
if not SEQ_LAYERS:
    SEQ_LAYERS = {int(os.environ["SEQ_LAYER"])}
COMP = set(int(x) for x in os.environ.get("I4X_LAYERS", "").split(",") if x != "")
if os.environ.get("BACKFILL_ANY_PREFIX") == "1":
    # backfill mode: capture ALL layers in one pass through the fully
    # compressed prefix. Layer L's input only depends on layers < L, so a
    # compressed layer >= L does not corrupt its capture (the y-teacher is
    # always computed from the original weights).
    pass
else:
    assert all(c < min(SEQ_LAYERS) for c in COMP), "compressed prefix must be BELOW the target layers"

MODEL_DIR = "C:/HAGI_v2/dsv4_shared_only"
TOKENIZER = r"C:/Users/shmid/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json"
OUT_DEFAULT = "checkpoints_dsv4/seq"

ACC: dict[int, dict[str, list]] = {}
COUNTS: dict[int, dict[str, int]] = {}
CUR_IDS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=258560)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    ap.add_argument("--no-vocab", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    snap = de.default_snapshot()
    wm = de.load_index(snap)["weight_map"]
    load_router(snap, wm)
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    torch.set_default_device("cuda")
    model = DeepseekV4ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    torch.set_default_device("cpu")
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = "eager"
    model.config.gradient_checkpointing = False
    global CUR_IDS

    def make_hook(li):
        def hook(module, args_, kwargs, output):
            x = args_[0]
            B, S, D = x.shape
            flat = x.reshape(-1, D).float()

            logits = flat @ ROUTER_W[li].T
            scores = F.softplus(logits).sqrt()
            if li in HASH_LAYERS:
                indices = ROUTER_TID[li][CUR_IDS.reshape(-1)]
            else:
                indices = torch.topk(scores + ROUTER_BIAS[li], TOP_K, dim=-1).indices
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * ROUTED_SCALE

            flatb = flat.to(torch.bfloat16)
            sw = gen.get_shared_dequant(li)
            out = ffn(flatb, sw["w1"], sw["w2"], sw["w3"]).float()

            collect = li in SEQ_LAYERS
            if li in COMP:
                # compressed expert path (z hoisted per layer; rows indexed by m_any!)
                _z = None
                for k in indices.unique().tolist():
                    d = get_int4x(li, k)
                    m_any = (indices == k).any(dim=1)
                    if _z is None:
                        d0 = d
                        _z = (flatb - d0["mu"]) @ d0["P"]
                    h = int4x_forward_z(d, _z[m_any])
                    out_k = (h.to(torch.bfloat16) @ d["w2b"].T).float()
                    pos = torch.cumsum(m_any.long(), 0) - 1
                    for kk in range(TOP_K):
                        m = indices[:, kk] == k
                        if m.any():
                            out[m] += weights[m, kk, None] * out_k[pos[m]]
            else:
                for k in indices.unique().tolist():
                    w1, w2, w3 = get_dequant(li, k)
                    m_any = (indices == k).any(dim=1)
                    x_k = flat[m_any]
                    y_k = ffn(x_k.to(torch.bfloat16), w1, w2, w3).float()
                    pos = torch.cumsum(m_any.long(), 0) - 1
                    for kk in range(TOP_K):
                        m = indices[:, kk] == k
                        if m.any():
                            out[m] += weights[m, kk, None] * y_k[pos[m]]
                    if collect:
                        key = str(k)
                        if COUNTS.setdefault(li, {}).get(key, 0) < args.cap:
                            take = min(args.cap - COUNTS.get(key, 0), x_k.shape[0])
                            ent = ACC.setdefault(li, {}).setdefault(key, ([], []))
                            ent[0].append(x_k[:take].detach().cpu().to(torch.bfloat16))
                            ent[1].append(y_k[:take].detach().cpu().to(torch.bfloat16))
                            COUNTS[li][key] = COUNTS[li].get(key, 0) + take
            return out.to(x.dtype).reshape(B, S, D)

        return hook

    for li in range(N_LAYERS):
        model.model.layers[li].mlp.register_forward_hook(make_hook(li), with_kwargs=True)

    vocab = 129280
    streams = []
    if not args.no_vocab:
        streams.append(("vocab", torch.arange(vocab).unsqueeze(0)))
    n_rand = max(0, args.max_tokens - (0 if args.no_vocab else vocab))
    if n_rand > 0:
        torch.manual_seed(1234)
        streams.append(("rand", torch.randint(0, vocab, (1, n_rand))))
    t0 = time.time()
    with torch.no_grad():
        for nm, ids in streams:
            CH = int(os.environ.get("SEQ_CH", "8192"))
            for c0 in range(0, ids.shape[1], CH):
                chunk = ids[:, c0 : c0 + CH]
                CUR_IDS = chunk.reshape(-1)
                model(chunk.cuda())
    for SL in sorted(SEQ_LAYERS):
        merged = {}
        for k, (xs, ys) in ACC.get(SL, {}).items():
            merged[k] = (torch.cat(xs, 0), torch.cat(ys, 0))
        outp = os.path.join(args.out_dir, f"acts_layer{SL}.pt")
        torch.save(merged, outp)
        tot = sum(v[0].shape[0] for v in merged.values())
        print(f"collected L{SL}: {len(merged)} experts, {tot} rows -> {outp}", flush=True)
    print(f"all layers done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
