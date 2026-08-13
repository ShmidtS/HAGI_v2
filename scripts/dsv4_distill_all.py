"""Distill ALL DeepSeek MoE layers into compact students, efficiently.

Two phases:
  1. COLLECT — one forward over text with the real 256-expert MoE (reusing the
     packed-expert LRU cache from dsv4_generate_fast). Every hooked layer
     captures its (x, y) pairs in the same pass: x = FFN input, y = MoE output.
  2. DISTILL — each layer, model-free: train a compact student (M glued SwiGLU
     blocks) to reproduce x -> y. No model skeleton, no expert re-loading — the
     teacher output was already computed and captured in phase 1.

The model is loaded ONCE (for activations). The distillation itself is the
board-probe method scaled to the whole model. Only the target layers are
hooked; unhooked layers in dsv4_shared_only already have zeroed experts, so
their FFN contributes nothing and the collected activations stay correct.

Usage:
    python scripts/dsv4_distill_all.py [layers] [m_blocks] [steps] [max_tokens]
    # layers: "all" or a comma list, e.g. "0,1,2,3"
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stub_import_tf  # noqa: F401
import dsv4_generate_fast as gf
from dsv4_distill_hagi import GluedStudent, distill

XS: dict[int, list] = {}
YS: dict[int, list] = {}
OUT_DIR = "C:/HAGI_v2/dsv4_distilled"

TEXT = (
    "The transformer architecture processes a sequence of tokens through stacked "
    "layers. Each layer applies multi-head attention followed by a feed-forward "
    "network. Mixture-of-experts models replace the dense feed-forward network "
    "with a set of expert networks and a router that selects a small subset for "
    "each token. Training such models requires balancing the load across experts. "
    "The router scores each token against every expert and selects the top few, "
    "normalizing their scores into routing weights. Each selected expert "
    "contributes its output scaled by the routing weight, and a shared expert "
    "adds a constant baseline. During inference only the selected experts are "
    "loaded and executed, which saves memory and compute. Distillation transfers "
    "the behavior of a large teacher into a smaller student by matching outputs "
    "rather than copying weights. This matters when the teacher's experts are "
    "mutually orthogonal, because orthogonal components cannot be linearly merged "
    "without losing the diversity that gives the mixture its power. A growing "
    "model can glue small experts together to increase its hidden dimension step "
    "by step, and use distillation to teach the glue how to combine the "
    "orthogonal blocks. "
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2) "
    "import numpy as np; x = np.linspace(0, 1, 100) "
    "The theory of relativity transformed our understanding of space and time. "
    "Quantum mechanics describes the behavior of matter at the smallest scales. "
    "Rust is a systems programming language with fearless concurrency. "
    "Привет, как дела? Машинное обучение учится на данных через оптимизацию."
)


def make_collect_hook(li):
    def hook(module, args, kwargs, output):
        x = args[0]
        B, S, D = x.shape
        flat = x.reshape(-1, D).float()

        logits = flat @ gf.ROUTER_W[li].T
        scores = F.softplus(logits).sqrt()
        if li in gf.HASH_LAYERS:
            indices = gf.ROUTER_TID[li][gf.CURRENT_IDS.reshape(-1)]
        else:
            indices = torch.topk(scores + gf.ROUTER_BIAS[li], gf.TOP_K, dim=-1).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * gf.ROUTED_SCALE

        sw = gf.get_shared_dequant(li)
        out = gf.ffn(flat, sw["w1"], sw["w2"], sw["w3"])
        for k in indices.unique().tolist():
            p = gf.get_expert_packed(li, k)
            w1 = gf.de.dequant_fp4(p["w1.weight"], p["w1.scale"])
            w2 = gf.de.dequant_fp4(p["w2.weight"], p["w2.scale"])
            w3 = gf.de.dequant_fp4(p["w3.weight"], p["w3.scale"])
            ek = gf.ffn(flat, w1, w2, w3)
            del w1, w2, w3
            for kk in range(gf.TOP_K):
                m = indices[:, kk] == k
                if m.any():
                    out[m] += weights[m, kk, None] * ek[m]
            del ek

        XS[li].append(flat.detach().cpu())
        YS[li].append(out.detach().cpu())
        return out.to(x.dtype).reshape(B, S, D)

    return hook


def main() -> int:
    layers_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    m_blocks = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    if layers_arg == "all":
        layers = list(range(gf.N_LAYERS))
    else:
        layers = [int(s) for s in layers_arg.split(",") if s.strip()]

    print(f"layers={layers}, m={m_blocks}, steps={steps}, max_tokens={max_tokens or 'all'}",
          flush=True)

    tok = gf.gigatoken.Tokenizer.from_json(open(gf.TOKENIZER, "rb").read())
    ids = [gf.BOS_ID] + list(tok.encode(TEXT))
    if max_tokens:
        ids = ids[:max_tokens]
    print(f"collect text: {len(ids)} tokens", flush=True)

    snap = gf.de.default_snapshot()
    wm = gf.de.load_index(snap)["weight_map"]
    print("loading router...", flush=True)
    gf.load_router(snap, wm)

    print("loading model skeleton (once)...", flush=True)
    t0 = time.time()
    torch.set_default_device("cuda")
    model = gf.DeepseekV4ForCausalLM.from_pretrained(gf.MODEL_DIR, torch_dtype=torch.float32)
    torch.set_default_device("cpu")
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = "eager"
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

    for li in layers:
        XS[li] = []
        YS[li] = []
    handles = [model.model.layers[li].mlp.register_forward_hook(make_collect_hook(li),
                                                                 with_kwargs=True)
               for li in layers]

    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    gf.CURRENT_IDS = input_ids
    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)
    print(f"forward (collect) in {time.time() - t0:.1f}s "
          f"(cache {gf.HIT}H/{gf.MISS}M, {len(gf.PACKED_CACHE)} experts)",
          flush=True)

    for h in handles:
        h.remove()
    del model
    gf.PACKED_CACHE.clear()
    gf.SHARED_DEQUANT.clear()
    torch.cuda.empty_cache()

    # Phase 2: distill each layer model-free.
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'layer':>5} | {'blocks':>6} | {'params':>9} | {'samples':>7} | "
          f"{'teacher E':>9} | {'MSE':>10} | {'residual %':>10} | {'saved':>8}")
    print("-" * 80, flush=True)

    stats = {}
    for li in layers:
        x = torch.cat(XS[li], dim=0)
        y = torch.cat(YS[li], dim=0)
        n = x.shape[0]
        teacher_energy = float((y.float() ** 2).mean())
        xg, yg = x.cuda(), y.cuda()
        student = GluedStudent(gf.de.DIM, gf.de.INTER, m_blocks).cuda()
        n_params = sum(p.numel() for p in student.parameters())
        t0 = time.time()
        distill(student, xg, yg, steps)
        with torch.no_grad():
            mse = float(F.mse_loss(student(xg), yg))
        residual_pct = mse / max(teacher_energy, 1e-9) * 100

        # Save the distilled layer (the new small model's FFN block).
        sd = {k: v.detach().cpu().to(torch.bfloat16)
              for k, v in student.state_dict().items()}
        out_path = os.path.join(OUT_DIR, f"layer_{li}.pt")
        torch.save(sd, out_path)

        stats[li] = {"blocks": m_blocks, "params": n_params, "samples": n,
                     "teacher_energy": teacher_energy, "mse": mse,
                     "residual_pct": residual_pct}
        print(f"{li:>5} | {m_blocks:>6} | {n_params / 1e6:>8.1f}M | {n:>7} | "
              f"{teacher_energy:>9.4f} | {mse:>10.6f} | {residual_pct:>9.1f}% | "
              f"{os.path.basename(out_path):>8}",
              flush=True)
        del student, xg, yg
        torch.cuda.empty_cache()

    import json
    with open(os.path.join(OUT_DIR, "report.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\nDone. Distilled weights saved to {OUT_DIR}/ (report.json).", flush=True)
    print(f"'residual %' = fraction of the teacher's own energy the compact", flush=True)
    print(f"student ({m_blocks} blocks) failed to absorb, per layer.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
