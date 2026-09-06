"""Latent-feedback gate training (full-bandwidth idea, arXiv:2608.08888).

Frozen compressed DSV4 + tiny trained gate that feeds the previous top-layer
hidden state back into the input at decode time:

    fused_t = e_t + RMSNorm(W_U @ h_{t-1}) * sigmoid(W_G @ e_t + b)

Adaptation vs the paper: our stack is a FROZEN pretrained model (never saw
fused inputs), so the fusion is an additive zero-init residual — at init the
model behaves exactly as standard. Token gates the state (paper's asymmetry);
RMSNorm on the value path (paper's stability recipe); jitter on the carried
state (paper's noise regularization).

Training = paper's multi-pass scheme, 2 passes, prefix mixin:
  pass 1: plain embeddings, no grad -> top states h (detached)
  pass 2: fused inputs (plain prefix, fused suffix), grad flows through the
          frozen stack into the gate only; loss = next-token CE on suffix.

Data: NO real corpora — synthetic streams only (vocab sweep + random tokens),
self-consistent with the compression calibration distribution.

Usage:
  GATE_TOKENS=400000 .venv/Scripts/python.exe scripts/dsv4_train_gate.py
Env: I4X_LAYERS (compressed set), GATE_OUT, GATE_TOKENS, GATE_CHUNK, GATE_LR
"""
import os, sys, time, math, random
import torch
import torch.nn.functional as F

os.chdir("C:/HAGI_v2")
sys.path.insert(0, "scripts")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import dsv4_generate_ttt as gen
import dsv4_refit_experts  # noqa: F401  (env plumbing for gen loading)
from gate_moe_fast import install_hooks
from transformers import AutoTokenizer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

GATE_TOKENS = int(os.environ.get("GATE_TOKENS", "400000"))
CHUNK = int(os.environ.get("GATE_CHUNK", "512"))
LR = float(os.environ.get("GATE_LR", "3e-3"))
JIT_SIGMA = float(os.environ.get("GATE_JITTER", "0.02"))
GATE_OUT = os.environ.get("GATE_OUT", "dsv4_latent_gate.pt")
VOCAB = 129280
N_LAYERS = 43
D = 4096


def rmsnorm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def build_stream(total: int, seed: int = 1234) -> torch.Tensor:
    """Structured SYNTHETIC stream (no real data). Random tokens carry no
    NTP signal (loss ~ ln V, damage invisible); these structures make the
    compression damage measurable/learnable:
      - cycles: period-P token loops (repetition tracking — the pathology
        the gate must repair)
      - copy blocks: random 64-token block repeated (pattern completion)
      - zipf-markov: order-1 chain over zipf-distributed tokens (local
        structure, router-realistic distribution)
      - random: coverage floor
    """
    g = torch.Generator().manual_seed(seed)
    segs = []
    n = 0
    V = VOCAB
    while n < total:
        kind = ["cycle", "copy", "markov", "rand"][int(torch.randint(0, 4, (1,), generator=g))]
        L = int(torch.randint(256, 1024, (1,), generator=g))
        L = min(L, total - n)
        if kind == "cycle":
            P = int(torch.randint(4, 64, (1,), generator=g))
            head = torch.randint(0, V, (P,), generator=g)
            segs.append(head.repeat(L // P + 1)[:L])
        elif kind == "copy":
            head = torch.randint(0, V, (64,), generator=g)
            segs.append(head.repeat(L // 64 + 1)[:L])
        elif kind == "markov":
            # zipf-ish stationary chain: heavy head + geometric jumps
            cur = int(torch.randint(0, 4096, (1,), generator=g))
            ids = []
            for _ in range(L):
                ids.append(cur)
                if torch.rand(1, generator=g) < 0.7:
                    cur = int(torch.randint(0, 4096, (1,), generator=g))
                else:
                    cur = (cur + int(torch.randint(1, 32, (1,), generator=g))) % 4096
            segs.append(torch.tensor(ids, dtype=torch.long))
        else:
            segs.append(torch.randint(0, V, (L,), generator=g))
        n += L
    return torch.cat(segs)[:total]


def main():
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M

    torch.set_default_device("cuda")
    model = DeepseekV4ForCausalLM.from_pretrained(gen.MODEL_DIR, torch_dtype=torch.bfloat16)
    torch.set_default_device("cpu")
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = "eager"
    for p in model.parameters():
        p.requires_grad_(False)
    # router tables for the MoE hooks (hash layers index by CURRENT_IDS)
    de = gen.de
    snap = de.default_snapshot()
    gen.load_router(snap, de.load_index(snap)["weight_map"])
    # inputs_embeds passes: hash routers get input_ids=None -> route by the
    # global CURRENT_IDS (set per chunk; no grad through routing tables)
    _HR = M.DeepseekV4HashRouter
    _orig_hr_fwd = _HR.forward

    def _hr_fwd(self, hidden_states, input_ids=None, *a, **kw):
        if input_ids is None:
            input_ids = gen.CURRENT_IDS.reshape(-1)[: hidden_states.reshape(-1, self.hidden_dim).shape[0]]
        return _orig_hr_fwd(self, hidden_states, input_ids, *a, **kw)

    _HR.forward = _hr_fwd
    # FAST MoE PATH (2026-09-05): resident packed terni4 banks (compressed
    # layers) + FP4 mmap streaming (original layers) + hoisted z + detach.
    # Replaces gen.make_hook: 143s/chunk -> target <30s (no torch.load/H2D
    # per expert). Gradients: residual+attention only (MoE detached).
    compressed = set(int(x) for x in
                     (os.environ.get("I4X_LAYERS") or "").split(",") if x != "")
    if not compressed:
        compressed = None  # all layers that have banks (load lazily)
    DETACH_MLP = os.environ.get("GATE_DETACH_MLP", "1") == "1"
    handles = install_hooks(model, compressed or set(range(N_LAYERS)), detach=DETACH_MLP)
    print(f"model loaded, i4x layers: {str(gen._i4x_layers())[:60]}", flush=True)

    # gate params (fp32 on GPU)
    W_U = torch.zeros(D, D, device="cuda", dtype=torch.float32)
    W_G = torch.zeros(D, D, device="cuda", dtype=torch.float32)
    b_G = torch.zeros(D, device="cuda", dtype=torch.float32)
    W_U.requires_grad_(True)
    W_G.requires_grad_(True)
    b_G.requires_grad_(True)
    opt = torch.optim.AdamW([W_U, W_G, b_G], lr=LR, weight_decay=0.0)

    embed = model.model.embed_tokens
    final_norm = model.model.norm

    stream = build_stream(GATE_TOKENS, seed=1234).long()
    n_chunks = stream.shape[0] // CHUNK
    print(f"stream {stream.shape[0]} tokens -> {n_chunks} chunks of {CHUNK}", flush=True)

    t0 = time.time()
    losses = []
    import gate_moe_fast as fast
    for ci in range(n_chunks):
        ids = stream[ci * CHUNK:(ci + 1) * CHUNK + 1].unsqueeze(0).cuda()
        inp = ids[:, :-1]
        tgt = ids[0, 1:]
        gen.CURRENT_IDS = inp  # hash-layer routing needs the token ids
        fast.CHUNK_ID = ci
        fast.PHASE = 1
        fast.REUSE.clear()
        tp = {"p1": 0.0, "gate": 0.0, "p2": 0.0, "bwd": 0.0}
        _t = time.time()
        with torch.no_grad():
            out1 = model(input_ids=inp, use_cache=False, output_hidden_states=True)
            h = out1.hidden_states[-1][0]           # [S, D] pre-final-norm
        torch.cuda.synchronize(); tp["p1"] = time.time() - _t; _t = time.time()
        h_norm = final_norm(h).float()              # head-space state
        if JIT_SIGMA > 0:
            h_norm = h_norm + JIT_SIGMA * torch.randn_like(h_norm)
        e = embed(inp)[0].float()                   # [S, D]
        # prefix mixin: random plain prefix
        p = random.randint(0, CHUNK - 2) if ci else CHUNK - 1
        fused = e.clone()
        if p >= 1:
            h_prev = h_norm[p - 1:-1]               # state from position t-1
            gate = torch.sigmoid(h_prev @ W_G if False else e[p:] @ W_G + b_G)
            val = rmsnorm(h_prev @ W_U)
            fused[p:] = e[p:] + val * gate
        fused_b = fused.to(torch.bfloat16)
        torch.cuda.synchronize(); tp["gate"] = time.time() - _t; _t = time.time()
        fast.PHASE = 2
        out2 = model(inputs_embeds=fused_b.unsqueeze(0), use_cache=False)
        logits = out2.logits[0, max(p - 1, 0):].float()
        tgts = tgt[max(p - 1, 0):]
        loss = F.cross_entropy(logits, tgts)
        torch.cuda.synchronize(); tp["p2"] = time.time() - _t; _t = time.time()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize(); tp["bwd"] = time.time() - _t; _t = time.time()
        torch.nn.utils.clip_grad_norm_([W_U, W_G, b_G], 1.0)
        opt.step()
        losses.append(float(loss))
        if os.environ.get("GATE_BENCH") or ci % 20 == 0:
            prof = "  ".join(f"{k}={gen.PROF_T[k]:.1f}s" for k in gen.PROF_T)
            st = fast.STATS
            tps = "  ".join(f"{k}={v:.1f}s" for k, v in tp.items())
            print(f"chunk {ci}/{n_chunks} loss {sum(losses[-20:]) / len(losses[-20:]):.4f} "
                  f"({time.time() - t0:.0f}s) [{tps}] "
                  f"reuse={st['reuse']} compute={st['compute']} hot={st['hot_hit']}/{st['hot_hit']+st['hot_miss']}", flush=True)
            for k_ in gen.PROF_T:
                gen.PROF_T[k_] = 0.0
        if ci % 400 == 399:
            torch.save({"W_U": W_U.detach().cpu(), "W_G": W_G.detach().cpu(),
                        "b_G": b_G.detach().cpu(), "chunk": ci}, GATE_OUT + ".tmp")
    for h_ in handles:
        h_.remove()
    torch.save({"W_U": W_U.detach().cpu(), "W_G": W_G.detach().cpu(),
                "b_G": b_G.detach().cpu(), "chunks": n_chunks,
                "i4x_layers": list(gen._i4x_layers() or [])}, GATE_OUT)
    print(f"DONE {n_chunks} chunks in {time.time() - t0:.0f}s -> {GATE_OUT}", flush=True)


if __name__ == "__main__":
    main()
