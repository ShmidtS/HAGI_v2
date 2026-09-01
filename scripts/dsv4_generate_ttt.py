"""Generation with int4x-compressed experts + TTT (test-time training) + self-evolution.

Builds on dsv4_generate_fast.py (custom MoE routing hook). Mechanisms:
  - experts with an int4x checkpoint (dsv4_reduced/layer_L/expert_k.pt) run COMPRESSED:
    z = (x - mu) @ P, h = silu(soft_lim(z@W1q^T+b1)) * soft_lim(z@W3q^T+b3),
    y = h @ (int4W2*s2)^T  (W2 is the ADAPTIVE readout, refit online by RLS)
  - FFN TTT: anchored RLS against the ORIGINAL FP4 expert applied to the same
    routed activations (local proximal teacher; measured 2.51% -> 1.66% resid).
  - Attention TTT (o_b_proj readout, [.,8192] -> 4096): SHADOW TEACHER pass -
    before each student prefill the same tokens run through the ORIGINAL model
    (all-FP4 experts, original o_b); the teacher's grouped context is regressed
    against by the student's features: min ||g_student @ w^T - y_orig||^2.
    This compensates upstream compression error (a self-referential teacher
    would converge to a no-op: the target must exist independently of the
    student's own weights).
  - honest validation: every 5th row NEVER enters G/C (Hva/Yva buffers);
    save decisions are computed ONLY on those rows.
  - save-on-improvement: snap adapted readout (int4 for experts, int8 for o_b -
    measured: the int4 grid collapses the attention adaptation gain) and write
    back ATOMICALLY (tmp + os.replace) when the honest validation improves by
    >= TTT_MIN_GAIN.
  - self-evolution (--evolve): endless self-talk sessions; the model continues
    ITS OWN stream of thought (context tail persisted in evolve_state.json -
    no fixed prompts); curiosity turns at temperature 1.25, repetition penalty;
    attention adapts in a rotating window (all 43 layers over sessions); FFN
    adapts on every checkpointed expert that fires; live adaptations are
    consolidated before any LRU eviction and on Ctrl+C.

Usage:
    python scripts/dsv4_generate_ttt.py "<prompt>" [max_new] [--no-ttt] [--no-save] [--no-attn]
    python scripts/dsv4_generate_ttt.py "" 0 --evolve [--sessions 0] [--turns 6]
        [--gen-tokens 200] [--attn-window 8] [--no-save]
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stub_import_tf  # noqa: F401  (must precede transformers: stubs torch.distributed)
import dsv4_experts as de
import gigatoken
from dsv4_experts import unpack_binary, unpack_int4
from dsv4_refit_experts import soft_lim
from safetensors import safe_open
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

import os as _os
if _os.name != "nt" and _os.path.exists("/mnt/c/HAGI_v2"):
    # WSL: transparently remap the hardcoded Windows paths
    def _w2l(p):
        return p.replace("C:/", "/mnt/c/") if p.startswith("C:/") else p
else:
    def _w2l(p):
        return p

MODEL_DIR = _w2l("C:/HAGI_v2/dsv4_shared_only")
TOKENIZER = _w2l(r"C:/Users/shmid/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json")
LOSSLESS = _w2l("C:/HAGI_v2/lossless_layers")
REDUCED = "dsv4_reduced"
# WSL ext4 copy (drvfs torch.load is ~5x slower): set HAGI_DATA=/root/hagi
# to read the big dirs from the native FS instead of /mnt/c.
_data = os.environ.get("HAGI_DATA")
if _data:
    MODEL_DIR = os.path.join(_data, "dsv4_shared_only")
    LOSSLESS = os.path.join(_data, "lossless_layers")
    REDUCED = os.path.join(_data, "dsv4_reduced")

N_LAYERS = 43
HASH_LAYERS = {0, 1, 2}
TOP_K = 6
ROUTED_SCALE = 1.5
SWIGLU_LIMIT = 10.0
BOS_ID = 0
EOS_ID = 1
GPU_HEADROOM = 12 * 1024**3

# --- TTT knobs ---
TTT_PER_LAYER = int(os.environ.get("TTT_PER_LAYER", "2"))  # experts TTT-updated per layer per prefill
TTT_REFIT = 64  # exact solve after this many accumulated train rows
TTT_PRECOND = os.environ.get("TTT_PRECOND", "1") == "1"  # delta-mem style: per-feature
# scale normalization when accumulating G/C (anti-drift preconditioning)
TTT_ALPHA = 1000.0  # anchor weight (in "token" units) toward the current readout
TTT_LAM = 0.9995  # forgetting factor
TTT_MIN_GAIN = 0.02  # save only if honest-validation resid improves by >= 2% relative
TTT_DELTA_RANK = int(os.environ.get("TTT_DELTA_RANK", "128"))  # low-rank persist of the W2 adaptation
TTT_MAX_EXPERTS = int(os.environ.get("TTT_MAX_EXPERTS", "64"))  # live FFN RLS states (G/C ~48MB each) - LRU
I4X_HIT = 0  # instrumentation
PROF_DECODE = bool(os.environ.get("PROF_DECODE"))
PROF_T = {"shared": 0.0, "route": 0.0, "get": 0.0, "fwd": 0.0, "outk": 0.0, "scatter": 0.0}
I4X_MISS = 0
TTT_ROWS_MAX = 2048  # max train rows kept per expert
# --- attention TTT: o_b_proj readout (grouped ctx [.,8192] -> hidden 4096) ---
TTT_ATTN_LAYERS = [int(v) for v in os.environ.get("TTT_ATTN_LAYERS", "0,1,2,3").split(",") if v != ""]
TTT_ATTN_MAX = 8  # live attention RLS states (G 8192^2 + C 8192x4096 fp32 ~ 384MB each)
TTT_ATTN_ALPHA = 1000.0
# --- self-evolution session loop (--evolve) ---
EVOL_STATE = os.path.join(REDUCED, "evolve_state.json")
CURIOSITY_TEMP = 1.25  # odd turns: high-temp exploration
NORMAL_TEMP = 0.9
EVOL_TOP_P = 0.98
EVOL_CTX = 1024
EVOL_CTX_KEEP = 256  # thought-tail persisted between sessions (the model thinks itself)
REP_PEN = 1.5  # repetition penalty divisor for tokens seen in the last REP_WIN
REP_WIN = 128
MIN_NEW_TOKENS = 8  # do not allow EOS before this many tokens in a turn

N_STREAMS = int(os.environ.get("EVOL_STREAMS", "4"))  # parallel thought streams (batched decode)
# routing-aware expert pinning: hash layers route by token id (EXACT sets from
# the stream), topk layers have stable usage -> pin the hot set in VRAM
PIN_BUDGET = int(os.environ.get("PIN_BUDGET", "30")) * 2**30
PINNED: set = set()
PINNED_BYTES = 0
COLLECT_USAGE = False
USAGE_T: dict[int, torch.Tensor] = {}  # li -> [256] pick counts
FLUSH_EVERY = 2  # teacher/RLS flush every N turns (amortizes expert materialization)
NGRAM = 3  # no-repeat n-gram blocking size (kills degenerate loops)
SAVE_ENABLED = True  # set from --no-save; gates persistence on eviction too
INT4X_MAX = int(os.environ.get("INT4X_MAX", "256"))  # hot tier + pins; cold = ~1.8ms bf16 unpack  # live int4x experts (~128MB each); LRU
DEQUANT_MAX = int(os.environ.get("DEQUANT_MAX", "128"))  # cached original experts (bf16 ~50MB each)
EXPERT_NOISE = float(os.environ.get("EXPERT_NOISE", "0"))  # inject eps relative noise into ORIGINAL expert outputs (quality-tolerance probe)
ATTN_ROWS_CAP = 256  # max student rows per attention RLS update (strided subsample)

MODE = "student"  # "teacher" during the shadow pass (hooks branch on this)
SHADOW: dict[int, list] = {}  # li -> list of teacher grouped-ctx chunks (aligned by feed order)

CURRENT_IDS: torch.Tensor | None = None
ROUTER_W: dict[int, torch.Tensor] = {}
ROUTER_BIAS: dict[int, torch.Tensor] = {}
ROUTER_TID: dict[int, torch.Tensor] = {}

PACKED_CACHE: collections.OrderedDict[tuple, dict] = collections.OrderedDict()
PACKED_MAX = int(os.environ.get("PACKED_MAX", "256"))  # backs dequant misses (no disk thrash)
CACHE_BYTES = 0
SHARED_DEQUANT: dict[int, dict] = {}
HIT = 0
MISS = 0

INT4X: dict[tuple, dict] = {}  # (L,k) -> dequant-int4x expert (frozen parts + adaptive w2)
I4X_PACKED: collections.OrderedDict[tuple, dict] = collections.OrderedDict()  # (L,k) -> packed ckpt ON GPU (~6MB)
I4X_PACKED_MAX = int(os.environ.get("I4X_PACKED_MAX", "1600"))
# decode-time TTT collections (per turn): attention ctx rows + FFN (x,h) per expert
COLLECT_ATTN: dict[int, list] = {}
COLLECT_FFN: dict[tuple, dict] = {}
COLLECT_CAP = 192  # max rows per expert per turn
TTT_STATE: collections.OrderedDict[tuple, dict] = collections.OrderedDict()
ATTN_ADAPT: dict[int, dict] = {}  # L -> {"w": [4096,8192] fp32 adapted, "w0": bf16 ref to module weight}
ATTN_STATE: collections.OrderedDict[int, dict] = collections.OrderedDict()
ATTN_SAVE_LOG: list[str] = []
SAVE_LOG: list[str] = []


def ffn(x, w1, w2, w3):
    gate = (x @ w1.T).clamp(max=SWIGLU_LIMIT)
    up = (x @ w3.T).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return (torch.nn.functional.silu(gate) * up) @ w2.T


def load_router(snap, wm):
    for li in range(N_LAYERS):
        p = f"layers.{li}.ffn.gate"
        ROUTER_W[li] = de.read_tensor(snap, wm, f"{p}.weight", device="cuda").to(torch.float32)
        if li in HASH_LAYERS:
            ROUTER_TID[li] = de.read_tensor(snap, wm, f"{p}.tid2eid", device="cuda").to(torch.long)
        else:
            ROUTER_BIAS[li] = de.read_tensor(snap, wm, f"{p}.bias", device="cuda").to(torch.float32)


def get_expert_packed(li, k):
    global CACHE_BYTES, HIT, MISS
    key = (li, k)
    if key in PACKED_CACHE:
        PACKED_CACHE.move_to_end(key)
        HIT += 1
        return PACKED_CACHE[key]
    MISS += 1
    fp = os.path.join(LOSSLESS, f"layers_{li}_ffn.safetensors")
    base = f"layers.{li}.ffn.experts.{k}"
    with safe_open(fp, framework="pt", device="cuda") as f:
        d = {}
        for proj in ("w1", "w2", "w3"):
            d[f"{proj}.weight"] = f.get_tensor(f"{base}.{proj}.weight")
            d[f"{proj}.scale"] = f.get_tensor(f"{base}.{proj}.scale")
    PACKED_CACHE[key] = d
    PACKED_CACHE.move_to_end(key)
    CACHE_BYTES += sum(t.numel() * t.element_size() for t in d.values())
    free, _ = torch.cuda.mem_get_info()
    lim = max(0, free - GPU_HEADROOM)
    while (CACHE_BYTES > lim or len(PACKED_CACHE) > PACKED_MAX) and len(PACKED_CACHE) > 1:
        _, ev = PACKED_CACHE.popitem(last=False)
        CACHE_BYTES -= sum(t.numel() * t.element_size() for t in ev.values())
        del ev
    return d


def get_shared_dequant(li):
    """bf16 shared experts (halves per-token bandwidth vs fp32 decode)."""
    if li in SHARED_DEQUANT:
        return SHARED_DEQUANT[li]
    fp = os.path.join(LOSSLESS, f"layers_{li}_ffn.safetensors")
    base = f"layers.{li}.ffn.shared_experts"
    d = {}
    with safe_open(fp, framework="pt", device="cuda") as f:
        for proj in ("w1", "w2", "w3"):
            w = f.get_tensor(f"{base}.{proj}.weight")
            s = f.get_tensor(f"{base}.{proj}.scale")
            d[proj] = de._decode(base, w, s).to(torch.bfloat16)
    SHARED_DEQUANT[li] = d
    return d


DEQUANT_CACHE: collections.OrderedDict[tuple, tuple] = collections.OrderedDict()
DEQUANT_BYTES = 0


def get_dequant(li, k):
    """Cached bf16 dequantized ORIGINAL expert (decode speed: per-token dequant
    of FP4 experts was the bottleneck; bf16 halves bandwidth and doubles GEMM
    throughput - teacher targets get ~0.4% bf16 rounding, acceptable)."""
    global DEQUANT_BYTES
    key = (li, k)
    if key in DEQUANT_CACHE:
        DEQUANT_CACHE.move_to_end(key)
        return DEQUANT_CACHE[key]
    p = get_expert_packed(li, k)
    w1 = de.dequant_fp4(p["w1.weight"], p["w1.scale"]).to(torch.bfloat16)
    w2 = de.dequant_fp4(p["w2.weight"], p["w2.scale"]).to(torch.bfloat16)
    w3 = de.dequant_fp4(p["w3.weight"], p["w3.scale"]).to(torch.bfloat16)
    DEQUANT_CACHE[key] = (w1, w2, w3)
    DEQUANT_CACHE.move_to_end(key)
    DEQUANT_BYTES += w1.numel() * 2 + w2.numel() * 2 + w3.numel() * 2
    free, _ = torch.cuda.mem_get_info()
    while (len(DEQUANT_CACHE) > DEQUANT_MAX or DEQUANT_BYTES > free - GPU_HEADROOM) and len(DEQUANT_CACHE) > 1:
        evk = next((kk for kk in DEQUANT_CACHE if kk not in PINNED), None)
        if evk is None:
            break
        ev = DEQUANT_CACHE.pop(evk)
        DEQUANT_BYTES -= ev[0].numel() * 2 + ev[1].numel() * 2 + ev[2].numel() * 2
        del ev
    return DEQUANT_CACHE[key]


POD_CACHE: dict[int, tuple] = {}
POD_BF16: dict[int, torch.Tensor] = {}


def _pod_bf16(li):
    if li not in POD_BF16:
        POD_BF16[li] = load_pod(li)[0].to(torch.bfloat16)
    return POD_BF16[li]


def get_w2_fp32(li, k):
    """Materialize the fp32 adaptive readout (TTT-only path; decode never calls)."""
    d = get_int4x(li, k)
    if d["w2"] is None:
        d["w2"] = d["w2b"].float()
    return d["w2"]


def unpack_binary_bf16(p: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """uint8 [out, in/8] (cuda) -> bf16 [out, in] {-1,+1} x per-row scale. ~1ms."""
    O, N = p.shape
    t = p.to(torch.uint8)
    bits = (t.unsqueeze(-1) >> torch.arange(8, device=p.device, dtype=torch.uint8)) & 1
    w = bits.reshape(O, N * 8).to(torch.bfloat16)
    w = w * 2 - 1
    return w * scale.to(torch.bfloat16)[:, None]


def unpack_nbit_bf16(p: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """uint8 bitstream [out, bytes] (cuda) -> bf16 [out, in] LEVELS[bits] x per-row scale."""
    import dsv4_experts as _de

    lv = torch.tensor(_de.LEVELS[bits], device=p.device)
    t = p.to(torch.int64)
    out_, nb = t.shape
    if bits == 2:
        res = torch.empty(out_, nb * 4, dtype=torch.float32, device=p.device)
        for i in range(4):
            res[:, i::4] = lv[(t >> (2 * i)) & 3]
    else:  # bits == 3
        n = nb // 3
        v = t[:, 0::3] | (t[:, 1::3] << 8) | (t[:, 2::3] << 16)
        res = torch.empty(out_, n * 8, dtype=torch.float32, device=p.device)
        for k in range(8):
            res[:, k::8] = lv[(v >> (3 * k)) & 7]
    sc = scale
    if sc.dim() == 2:  # group scales [out, ng] -> [out, in], elementwise
        gs = res.shape[1] // sc.shape[1]
        sc = sc.repeat_interleave(gs, dim=1)
        return (res * sc.to(torch.float32)).to(torch.bfloat16)
    return (res * sc.to(torch.float32)[:, None]).to(torch.bfloat16)


def unpack_ternary_bf16(p: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """uint8 [out, ceil(in/5)] (cuda) -> bf16 [out, in] {-1,0,1} x scale.
    scale: [out] per-row OR [out, ng] group scales."""
    from dsv4_experts import unpack_ternary

    t = unpack_ternary(p).to(torch.float32)
    if scale.dim() == 2:  # group scales [out, ng], gs=128 fixed by refit
        in_ = scale.shape[1] * 128
        t = t[:, :in_]
        sc = scale.repeat_interleave(128, dim=1)
        return (t * sc.to(torch.float32)).to(torch.bfloat16)
    sc = scale
    return (t * sc.to(torch.float32)[:, None]).to(torch.bfloat16)


def unpack_int4_bf16(p: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """uint8 [out, in/2] (cuda) -> bf16 [out, in] {-7..7} x scale.
    scale: [out] per-row OR [out, ng] group scales (expanded). ~0.5ms."""
    t = p.to(torch.int16)
    lo = ((t & 15) - 8).to(torch.bfloat16)
    hi = ((t >> 4) - 8).to(torch.bfloat16)
    out = torch.empty(t.shape[0], t.shape[1] * 2, dtype=torch.bfloat16, device=p.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    sc = scale
    if sc.dim() == 2:  # group scales [out, ng] -> [out, in], elementwise
        gs = out.shape[1] // sc.shape[1]
        sc = sc.repeat_interleave(gs, dim=1)
        return out * sc.to(torch.bfloat16)
    return out * sc.to(torch.bfloat16)[:, None]


def load_pod(li):
    if li in POD_CACHE:
        return POD_CACHE[li]
    red = os.path.join(REDUCED, f"layer_{li}")
    P = torch.load(os.path.join(red, "P.pt"), map_location="cuda").float()
    mu = torch.load(os.path.join(red, "mu.pt"), map_location="cuda").float()
    POD_CACHE[li] = (P, mu)
    return P, mu


def prewarm_packed(n_threads=8, verbose=True):
    """One sequential sweep over all expert files into the GPU packed cache.
    Turns per-token random small-file IO (drvfs/9P per-file overhead is
    ~40ms on WSL) into a one-time parallel prefetch. With I4X_PACKED_MAX
    >= 11008 the whole routed mass (~72GB) stays resident (unified memory)
    and generation then does ZERO file IO."""
    import concurrent.futures as cf
    t0 = time.time()
    paths = []
    for li in range(N_LAYERS):
        for k in range(256):
            fp = os.path.join(REDUCED, f"layer_{li}", f"expert_{k}.pt")
            if os.path.exists(fp):
                paths.append((li, k, fp))
    if verbose:
        print(f"prewarm: {len(paths)} expert files, {n_threads} threads", flush=True)

    def _load(item):
        li, k, fp = item
        return li, k, torch.load(fp, map_location="cpu", weights_only=False)

    n = 0
    with cf.ThreadPoolExecutor(n_threads) as ex:
        for li, k, e in ex.map(_load, paths):
            key_p = (li, k)
            if key_p in I4X_PACKED:
                continue
            e = {kk: (v.cuda() if torch.is_tensor(v) else v) for kk, v in e.items()}
            I4X_PACKED[key_p] = e
            I4X_PACKED.move_to_end(key_p)
            n += 1
            if verbose and n % 1024 == 0:
                print(f"  prewarm {n}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
    while len(I4X_PACKED) > I4X_PACKED_MAX:
        I4X_PACKED.popitem(last=False)
    if verbose:
        print(f"prewarm done: {n} experts resident in {time.time() - t0:.0f}s", flush=True)


_I4X_LAYERS = None  # bisect: None = all compressed; else a set of layers


def _i4x_layers():
    global _I4X_LAYERS
    if _I4X_LAYERS is None:
        v = os.environ.get("I4X_LAYERS")
        _I4X_LAYERS = set(int(x) for x in v.split(",")) if v else None
    return _I4X_LAYERS


def get_int4x(li, k):
    """Dequantized int4x expert: frozen parts + current (possibly adapted) W2."""
    global I4X_HIT, I4X_MISS
    if os.environ.get("INT4X_OFF"):
        return None  # baseline: original FP4 experts everywhere
    _ll = _i4x_layers()
    if _ll is not None and li not in _ll:
        return None  # bisect: this layer runs the original FP4 expert
    key = (li, k)
    if key in INT4X:
        I4X_HIT += 1
        return INT4X[key]
    I4X_MISS += 1
    key_p = (li, k)
    if key_p in I4X_PACKED:
        e = I4X_PACKED[key_p]
        I4X_PACKED.move_to_end(key_p)
    else:
        fp = os.path.join(REDUCED, f"layer_{li}", f"expert_{k}.pt")
        if not os.path.exists(fp):
            return None
        e = torch.load(fp, map_location="cpu", weights_only=False)
        e = {kk: (v.cuda() if torch.is_tensor(v) else v) for kk, v in e.items()}
        I4X_PACKED[key_p] = e
        I4X_PACKED.move_to_end(key_p)
        while len(I4X_PACKED) > I4X_PACKED_MAX:
            I4X_PACKED.popitem(last=False)
    mode = e.get("mode", "")
    _b13 = _b2 = None
    if mode == "int4x":
        _b13, _b2 = 1, 4
    elif mode == "terni4":
        _b13, _b2 = "tern", 4
    elif len(mode) == 4 and mode[0] == "i" and mode[2] == "i" and mode[1].isdigit() and mode[3].isdigit():
        _b13, _b2 = int(mode[1]), int(mode[3])
    if _b13 is None:
        return None
    P, mu = load_pod(li)
    dev = "cuda"
    e_gpu = I4X_PACKED[key_p]  # already on cuda
    if _b13 == 1:
        _u13 = unpack_binary_bf16
    elif _b13 == "tern":
        _u13 = unpack_ternary_bf16
    elif _b13 == 4:
        _u13 = unpack_int4_bf16
    else:
        _u13 = lambda p_, s_: unpack_nbit_bf16(p_, s_, _b13)
    if _b2 == 4:
        _u2 = unpack_int4_bf16
    elif _b2 == 6:
        def _u2(p_, s_):
            t = p_.to(torch.int64)
            n4 = t.shape[1] // 3
            w24 = t[:, 0::3] | (t[:, 1::3] << 8) | (t[:, 2::3] << 16)
            res = torch.empty(t.shape[0], n4 * 4, dtype=torch.float32, device=p_.device)
            for i in range(4):
                res[:, i::4] = ((w24 >> (6 * i)) & 63).float() - 31.0
            return (res * s_.to(torch.float32)[:, None]).to(torch.bfloat16)
    else:
        _u2 = lambda p_, s_: unpack_nbit_bf16(p_, s_, _b2)
    d = {
        "P": _pod_bf16(li),  # shared per layer
        "mu": mu.to(torch.bfloat16),
        "w1": _u13(e_gpu["w1a"], e_gpu["w1a_scale"]),
        "w3": _u13(e_gpu["w3a"], e_gpu["w3a_scale"]),
        "b1": e_gpu["bias1a"].to(torch.bfloat16),
        "b3": e_gpu["bias3a"].to(torch.bfloat16),
        "w2": None,  # fp32 materialized lazily for TTT only
        "w2b": _u2(e_gpu["w2a"], e_gpu["w2a_scale"]),
        "residual": e.get("residual", float("inf")),
        "adapted": False,
    }
    d["adapted"] = False
    if e.get("ttt_A") is not None:  # persisted low-rank TTT delta (previous sessions)
        dA = e["ttt_A"].float()
        dB = e["ttt_B"].float()
        d["w2b"] = (d["w2b"].float() + dA @ dB.T).to(torch.bfloat16)
    INT4X[key] = d
    while len(INT4X) > INT4X_MAX:
        ev = next((kk for kk in INT4X if kk not in TTT_STATE and kk not in PINNED), None)
        if ev is None:
            break  # pinned/states protect the rest
        INT4X.pop(ev)  # adapted entries are consolidated on STATE eviction, never lost
    return d


def int4x_forward(d, x_rows):
    """x_rows [n, 4096] raw fp32 -> h [n, 2048] fp32 (bf16 GEMMs inside)."""
    xb = x_rows.to(torch.bfloat16)
    z = (xb - d["mu"]) @ d["P"]
    g = soft_lim(z @ d["w1"].T + d["b1"])
    u = soft_lim(z @ d["w3"].T + d["b3"])
    return (torch.nn.functional.silu(g) * u).float()


def int4x_forward_z(d, z):
    """Pre-rotated variant: z [n, 4096] (already (x-mu)@P, bf16-friendly fp32)
    -> h [n, 2048]. The (x-mu)@P GEMM reads the 32MB P once per LAYER, not
    once per expert (8x less traffic; FreeToken-style hoisting)."""
    g = soft_lim(z @ d["w1"].T + d["b1"])
    u = soft_lim(z @ d["w3"].T + d["b3"])
    return (torch.nn.functional.silu(g) * u).float()


def _teacher_y(li, k, x_rows):
    w1, w2, w3 = get_dequant(li, k)
    return ffn(x_rows.to(torch.bfloat16), w1, w2, w3).float()


def _new_state(w_anchor_T, Fin, alpha):
    return {
        "G": torch.eye(Fin, device="cuda") * alpha,
        "C": alpha * w_anchor_T.clone(),
        "Fin": Fin,
        "H": [],  # train rows (fed into G/C; refit guard)
        "Y": [],
        "Hva": [],  # validation rows (NEVER fed into G/C; save decision)
        "Yva": [],
        "rows": 0,
        "since": 0,
        "refits": 0,
    }


def _rls_step(st, h_rows, y_rows, anchor_w, apply_to):
    """Shared anchored-RLS step: 4/5 rows -> G/C, 1/5 -> honest validation;
    guarded exact refit every TTT_REFIT train rows; apply_to updated in place."""
    G, C = st["G"], st["C"]
    n = h_rows.shape[0]
    n_va = n // 5 if n >= 5 else (1 if n > 1 else 0)
    tr_h, tr_y = h_rows[: n - n_va], y_rows[: n - n_va]
    va_h, va_y = h_rows[n - n_va :], y_rows[n - n_va :]
    nt = tr_h.shape[0]
    if TTT_PRECOND:
        # delta-mem stabilization: normalize feature columns (RMS) so the
        # Gram stays O(1) despite the 1e12 eigenvalue spread of h; the solve
        # is a mere reparameterization, W2 is recovered by un-scaling.
        sc = st.setdefault("fsc", None)
        if sc is None:
            sc = tr_h.pow(2).mean(0).sqrt().clamp_min(1e-6)
            st["fsc"] = sc
        tr_h = tr_h / sc
        tr_y = tr_y  # targets unchanged; the solved readout acts on scaled h
    G.mul_(TTT_LAM**nt).add_(tr_h.T @ tr_h)
    C.mul_(TTT_LAM**nt).add_(tr_h.T @ tr_y)
    st["rows"] += n
    st["since"] += nt
    st["H"].append(tr_h.detach())
    st["Y"].append(tr_y.detach())
    if va_h.shape[0]:
        st["Hva"].append(va_h.detach())
        st["Yva"].append(va_y.detach())
        if sum(t.shape[0] for t in st["Hva"]) > TTT_ROWS_MAX // 2:
            st["Hva"].pop(0)
            st["Yva"].pop(0)
    if sum(t.shape[0] for t in st["H"]) > TTT_ROWS_MAX:
        st["H"].pop(0)
        st["Y"].pop(0)
    if st["since"] >= TTT_REFIT:
        st["since"] = 0
        st["refits"] += 1
        Hb, Yb = torch.cat(st["H"]), torch.cat(st["Y"])
        r_before = _resid(anchor_w, Hb, Yb)
        reg = G.diagonal().mean() * 1e-3
        w_new = torch.linalg.solve(G + reg * torch.eye(st["Fin"], device="cuda"), C).T.contiguous()
        if TTT_PRECOND and st.get("fsc") is not None:
            w_new = w_new / st["fsc"]  # un-scale: readout acts on raw h again
        r_after = _resid(w_new, Hb, Yb)
        if r_after < r_before:
            apply_to.copy_(w_new)
            ret = (r_before, r_after)
        else:
            ret = None
        del w_new, Hb, Yb
        return ret
    return None


def ttt_touch(li, k):
    key = (li, k)
    if key not in TTT_STATE:
        while len(TTT_STATE) >= TTT_MAX_EXPERTS:
            ev_key = next(iter(TTT_STATE))
            d_ev = INT4X.get(ev_key)
            if d_ev is not None and d_ev.get("adapted"):
                consolidate_and_save(ev_key[0], ev_key[1], enabled=SAVE_ENABLED)  # pops state
            else:
                TTT_STATE.pop(ev_key, None)
        st = {
            "G": None,  # lazy: needs h energy scale from the first rows
            "C": None,
            "Fin": INT4X[key]["w2b"].shape[1],
            "H": [], "Y": [], "Hva": [], "Yva": [],
            "rows": 0, "since": 0, "refits": 0,
        }
        TTT_STATE[key] = st
    TTT_STATE.move_to_end(key)
    return TTT_STATE[key]


def ttt_update(li, k, h_rows, y_rows):
    """FFN: anchored RLS against the original FP4 expert outputs."""
    st = ttt_touch(li, k)
    w2 = get_w2_fp32(li, k)
    if st["G"] is None:
        alpha = (h_rows.T @ h_rows).diagonal().mean().item() / max(1, h_rows.shape[0]) * TTT_ALPHA
        fresh = _new_state(w2.T, st["Fin"], alpha)
        st["G"], st["C"] = fresh["G"], fresh["C"]
    ret = _rls_step(st, h_rows, y_rows, w2, w2)
    if ret is not None:
        INT4X[(li, k)]["adapted"] = True
        INT4X[(li, k)]["w2b"].copy_(w2.to(torch.bfloat16))
    return ret


def _resid(w2, h, y):
    yh = h @ w2.T
    return (((yh - y) ** 2).sum() / (y**2).sum().clamp_min(1e-12)).item()


def snap_int8(w):
    """int8 {-127..127} + per-channel LS scale; measured: holds the adaptation
    gain where int4 collapses (attn o_b: 0.50% vs 3.10%)."""
    sg = w.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 127.0
    q = (w.float() / sg).round().clamp(-127, 127)
    num = (q * w.float()).sum(dim=1)
    den = (q * q).sum(dim=1).clamp_min(1e-9)
    a = num / den
    a = a.sign() * a.abs().clamp_min(1e-9)
    return q, a


def snap_int4(w2):
    sg = w2.abs().amax(dim=1, keepdim=True).clamp_min(1e-9) / 7.0
    q = (w2 / sg).round().clamp(-7, 7)
    num = (q * w2).sum(dim=1)
    den = (q * q).sum(dim=1).clamp_min(1e-9)
    a = num / den
    a = a.sign() * a.abs().clamp_min(1e-6)
    return q, a


def consolidate_and_save(li, k, enabled=True):
    """Snap adapted W2 to int4; save ONLY if the honest validation (rows never
    fed into G/C) improves over the current checkpoint readout."""
    key = (li, k)
    if key not in TTT_STATE:
        return
    st = TTT_STATE.pop(key)
    if key in INT4X:
        INT4X[key]["adapted"] = False
    if not st["Hva"]:  # no honest validation rows -> refuse to decide
        return
    Hh, Yh = torch.cat(st["Hva"]), torch.cat(st["Yva"])
    w2 = get_w2_fp32(li, k)  # adapted continuous readout
    e = torch.load(os.path.join(REDUCED, f"layer_{li}", f"expert_{k}.pt"), map_location="cpu", weights_only=False)
    q2_0 = unpack_int4(e["w2a"]).float().cuda()
    w2_base = q2_0 * e["w2a_scale"].float().cuda()[:, None]
    del q2_0
    # baseline = current file state (int4 + any previously persisted delta)
    dA0, dB0 = e.get("ttt_A"), e.get("ttt_B")
    old_delta = dA0.float().cuda() @ dB0.float().cuda().T if dA0 is not None else 0.0
    w2_cur = w2_base + old_delta
    r_cur = _resid(w2_cur, Hh, Yh)
    r_cont = _resid(w2, Hh, Yh)
    # full delta vs the int4 base (fold the previous delta in: cumulative lineage)
    total = (w2 - w2_cur) + old_delta
    r = min(TTT_DELTA_RANK, total.shape[1] - 1)
    U, S, V = torch.svd_lowrank(total, q=min(r + 8, total.shape[1] - 1), niter=4)
    A = (U[:, :r] * S[:r])   # [4096, r]
    B = V[:, :r]             # [2048, r]
    w2_new = w2_base + A @ B.T
    r_new = _resid(w2_new, Hh, Yh)
    improved = (r_cur - r_new) / max(r_cur, 1e-12)
    print(
        f"  [ttt] L{li} k{k}: rows={st['rows']} refits={st['refits']} val "
        f"cur={r_cur*100:.3f}% cont={r_cont*100:.3f}% lowrank(r={r})={r_new*100:.3f}% "
        f"(gain {improved*100:+.1f}%)",
        flush=True,
    )
    if not enabled or improved < TTT_MIN_GAIN:
        return
    fp = os.path.join(REDUCED, f"layer_{li}", f"expert_{k}.pt")
    e["ttt_A"] = A.half().cpu()
    e["ttt_B"] = B.half().cpu()
    e["residual"] = r_new
    e["ttt"] = {"rows": st["rows"], "refits": st["refits"], "cur": r_cur, "new": r_new, "rank": r}
    tmp = fp + ".tmp"
    torch.save(e, tmp)
    os.replace(tmp, fp)
    I4X_PACKED.pop((li, k), None)  # drop stale packed entry (it has no delta)
    SAVE_LOG.append(f"L{li}_{k}: {r_cur*100:.3f}% -> {r_new*100:.3f}%")
    print(f"  [ttt] SAVED {fp}", flush=True)


# ---------------------------------------------------------------- attention ---

def attn_ob_path(li):
    return os.path.join(REDUCED, f"layer_{li}", "attn_ob.pt")


def attn_init(li, module):
    """w0 = reference to the module's original bf16 weight (no fp32 copy);
    w starts from the saved int8 adaptation if present, else from w0."""
    w0 = module.weight.detach()  # bf16, on cuda (teacher reference)
    w = w0.float().clone()
    fp = attn_ob_path(li)
    if os.path.exists(fp):
        e = torch.load(fp, map_location="cpu", weights_only=False)
        q = e["ob"].float().cuda()
        w = q * e["ob_scale"].float().cuda()[:, None]
        print(f"  [attn-ttt] L{li}: loaded adapted o_b (resid {e.get('residual', float('nan'))*100:.3f}%)", flush=True)
    ATTN_ADAPT[li] = {"w": w, "w0": w0}


def attn_touch(li, h_rows):
    if li not in ATTN_STATE:
        if len(ATTN_STATE) >= TTT_ATTN_MAX:
            ATTN_STATE.popitem(last=False)  # coldest state dropped; adapted w persists
        alpha = (h_rows.T @ h_rows).diagonal().mean().item() / max(1, h_rows.shape[0]) * TTT_ATTN_ALPHA
        ATTN_STATE[li] = _new_state(ATTN_ADAPT[li]["w"].T, h_rows.shape[1], alpha)
    ATTN_STATE.move_to_end(li)
    return ATTN_STATE[li]


def attn_ttt_update(li, h_rows, y_rows):
    st = attn_touch(li, h_rows)
    w = ATTN_ADAPT[li]["w"]
    _rls_step(st, h_rows, y_rows, w, w)


def attn_consolidate_and_save(li, enabled=True):
    """Snap adapted o_b to int8; save when honest validation improves vs original."""
    if li not in ATTN_STATE:
        return
    st = ATTN_STATE.pop(li)
    if not st["Hva"]:
        return
    Hh, Yh = torch.cat(st["Hva"]), torch.cat(st["Yva"])
    w = ATTN_ADAPT[li]["w"]
    w0 = ATTN_ADAPT[li]["w0"].float()
    r_static = _resid(w0, Hh, Yh)
    r_cont = _resid(w, Hh, Yh)
    q, a = snap_int8(w)
    r_cons = _resid(q * a[:, None], Hh, Yh)
    if r_cons > r_static:
        r_cons = r_static
        q, a = snap_int8(w0)
    improved = (r_static - r_cons) / max(r_static, 1e-12)
    print(
        f"  [attn-ttt] L{li}: rows={st['rows']} refits={st['refits']} val "
        f"orig={r_static*100:.4f}% cont={r_cont*100:.4f}% cons={r_cons*100:.4f}% "
        f"(gain {improved*100:+.1f}%)",
        flush=True,
    )
    if not enabled or improved < TTT_MIN_GAIN:
        return
    fp = attn_ob_path(li)
    e = {"ob": q.to(torch.int8).cpu(), "ob_scale": a.cpu(), "residual": r_cons,
         "ttt": {"rows": st["rows"], "refits": st["refits"], "static": r_static, "cons": r_cons}}
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    tmp = fp + ".tmp"
    torch.save(e, tmp)
    os.replace(tmp, fp)
    ATTN_SAVE_LOG.append(f"L{li}: {r_static*100:.4f}% -> {r_cons*100:.4f}%")
    print(f"  [attn-ttt] SAVED {fp}", flush=True)


def make_attn_hook(li, ttt_on):
    """Teacher mode: record the original model's grouped context (shadow).
    Student mode: output = grouped @ w_adapted.T; on prefill, regress against
    the shadow teacher outputs (y_orig = g_teacher @ w0.T)."""
    def hook(module, args, kwargs, output):
        if MODE == "teacher":
            g = args[0]
            SHADOW.setdefault(li, []).append(g.reshape(-1, g.shape[-1]).detach())
            return None  # original module forward (original o_b weights)
        grouped = args[0]
        ad = ATTN_ADAPT[li]
        g = grouped.reshape(-1, grouped.shape[-1]).float()
        out = (g @ ad["w"].T).to(output.dtype).reshape(output.shape)
        if grouped.shape[1] > 1:  # prefill: consume the teacher chunks directly
            chunks = SHADOW.pop(li, None)
            if ttt_on and chunks:
                g_t = torch.cat(chunks)
                n = min(g.shape[0], g_t.shape[0])
                if n > 1:
                    if n > ATTN_ROWS_CAP:
                        sel = torch.arange(0, n, n // ATTN_ROWS_CAP, device=g.device)
                    else:
                        sel = slice(None)
                    y_t = (g_t[sel].to(ad["w0"].dtype) @ ad["w0"].T).float()
                    attn_ttt_update(li, g[sel], y_t)
                    del y_t
        elif ttt_on:  # decode step: collect student rows for the turn-end flush
            COLLECT_ATTN.setdefault(li, []).append(g.detach())
        return out

    return hook


def make_hook(li, ttt_on):
    """MoE routing hook. Teacher mode (shadow pass): original experts only, no
    TTT - this IS the original model. Student mode: int4x experts where
    checkpoints exist, original FP4 (cached bf16 dequant) elsewhere.
    TTT runs ONLY at prefill (S>1), on the top TTT_PER_LAYER experts by routed
    row count - decode steps carry zero TTT overhead (rows are recovered by the
    next turn's prefill anyway)."""
    def hook(module, args, kwargs, output):
        x = args[0]
        B, S, D = x.shape
        flat = x.reshape(-1, D).float()

        logits = flat @ ROUTER_W[li].T
        scores = torch.nn.functional.softplus(logits).sqrt()
        if li in HASH_LAYERS:
            indices = ROUTER_TID[li][CURRENT_IDS.reshape(-1)]
        else:
            indices = torch.topk(scores + ROUTER_BIAS[li], TOP_K, dim=-1).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * ROUTED_SCALE

        _pr = PROF_DECODE
        _t0 = _pr and time.perf_counter()
        sw = get_shared_dequant(li)
        flatb = flat.to(torch.bfloat16)
        out = ffn(flatb, sw["w1"], sw["w2"], sw["w3"]).float()
        if _pr:
            PROF_T["shared"] += time.perf_counter() - _t0
            _t0 = time.perf_counter()

        teacher = MODE == "teacher"
        if COLLECT_USAGE and not teacher:
            USAGE_T[li] += torch.bincount(indices.reshape(-1), minlength=256)
        prefill = (not teacher) and S > 1 and ttt_on
        cands = []  # (rows, k) TTT candidates at prefill
        if _pr:
            PROF_T["route"] += time.perf_counter() - _t0  # includes unique().tolist() sync
        if flat.shape[0] == 1:
            # decode fast path: single row, no scatter machinery, two syncs total;
            # z = (x-mu)@P hoisted per layer (shared by all routed experts)
            _w_list = weights[0].tolist()
            _d0 = get_int4x(li, indices[0, 0].item()) if not teacher else None
            _z_layer = None
            for j, k in enumerate(indices[0].tolist()):
                _tk = _pr and time.perf_counter()
                d = None if teacher else get_int4x(li, k)
                if _pr:
                    PROF_T["get"] += time.perf_counter() - _tk
                if d is None:
                    # baseline/bisect: this expert runs the ORIGINAL FP4 weights
                    w1, w2, w3 = get_dequant(li, k)
                    ek = ffn(flatb, w1, w2, w3).float()
                    out[0] += _w_list[j] * ek[0]
                    continue
                if _z_layer is None:
                    _z_layer = (flat.to(torch.bfloat16) - d["mu"]) @ d["P"]
                if _pr:
                    _tk = time.perf_counter()
                h = int4x_forward_z(d, _z_layer)
                out[0] += _w_list[j] * (h.to(torch.bfloat16) @ d["w2b"].T).float()[0]
                if _pr:
                    PROF_T["fwd"] += time.perf_counter() - _tk
            del _d0, _z_layer
            if _pr:
                _t0 = time.perf_counter()
            for kk in range(0):  # keep later accumulators coherent
                pass
        else:
            for k in indices.unique().tolist():
                _tk = _pr and time.perf_counter()
                d = None if teacher else get_int4x(li, k)
                if _pr:
                    PROF_T["get"] += time.perf_counter() - _tk
                if d is not None:
                    if _pr:
                        _tk = time.perf_counter()
                    m_any = (indices == k).any(dim=1)
                    h = int4x_forward(d, flat[m_any])
                    if _pr:
                        PROF_T["fwd"] += time.perf_counter() - _tk
                        _tk = time.perf_counter()
                    out_k = (h.to(torch.bfloat16) @ d["w2b"].T).float()  # [rows_any, D]
                    if _pr:
                        PROF_T["outk"] += time.perf_counter() - _tk
                        _tk = time.perf_counter()
                    pos = torch.cumsum(m_any.long(), 0) - 1  # row -> position in xm
                    for kk in range(TOP_K):
                        m = indices[:, kk] == k
                        if m.any():
                            out[m] += weights[m, kk, None] * out_k[pos[m]]
                    if _pr:
                        PROF_T["scatter"] += time.perf_counter() - _tk
                    if prefill:
                        cands.append((int(m_any.sum()), k))
                    elif ttt_on:  # decode step: collect rows for the turn-end flush
                        ekey = (li, k)
                        ent = COLLECT_FFN.get(ekey)
                        if ent is None:
                            ent = COLLECT_FFN[ekey] = {"x": [], "h": [], "n": 0}
                        if ent["n"] < COLLECT_CAP:
                            ent["x"].append(flat[m_any].detach())
                            ent["h"].append(h.detach())
                            ent["n"] += int(m_any.sum())
                    del h, out_k, m_any, pos
                else:
                    w1, w2, w3 = get_dequant(li, k)
                    ek = ffn(flat.to(torch.bfloat16), w1, w2, w3).float()
                    if EXPERT_NOISE > 0.0:
                        rms = ek.pow(2).mean(dim=1, keepdim=True).sqrt()
                        ek = ek + rms * EXPERT_NOISE * torch.randn_like(ek)
                    for kk in range(TOP_K):
                        m = indices[:, kk] == k
                        if m.any():
                            out[m] += weights[m, kk, None] * ek[m]
                    del ek

        if cands:  # prefill TTT: hottest experts only, teacher = original FP4
            cands.sort(reverse=True)
            for rows, k in cands[:TTT_PER_LAYER]:
                d = get_int4x(li, k)
                m_any = (indices == k).any(dim=1)
                h = int4x_forward(d, flat[m_any])
                y_t = _teacher_y(li, k, flat[m_any])
                ttt_update(li, k, h, y_t)
                del h, y_t, m_any

        correct = out.to(x.dtype).reshape(B, S, D)
        del out, flat, scores, logits, indices, weights
        return correct

    return hook


def setup_model():
    """Load router + bf16 skeleton; shared by the one-shot generator and evolve loop."""
    print("loading router...", flush=True)
    snap = de.default_snapshot()
    wm = de.load_index(snap)["weight_map"]
    load_router(snap, wm)

    print("loading model skeleton...", flush=True)
    t0 = time.time()
    torch.set_default_device("cuda")
    model: DeepseekV4ForCausalLM = DeepseekV4ForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    torch.set_default_device("cpu")
    model.eval()
    model = model.to(torch.bfloat16)
    model.config._experts_implementation = "eager"
    model.config.gradient_checkpointing = False
    free, total = torch.cuda.mem_get_info()
    print(f"model loaded in {time.time() - t0:.1f}s, GPU free={free / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)
    return model


def shadow_prefill(model, input_ids, past=None, keep_past=False):
    """Forward through the ORIGINAL model (teacher); returns its past if kept."""
    global MODE, CURRENT_IDS
    MODE = "teacher"
    CURRENT_IDS = input_ids
    try:
        out = model(input_ids=input_ids, use_cache=True, past_key_values=past)
    finally:
        MODE = "student"
    return out.past_key_values if keep_past else None


# ------------------------------------------------------------------- evolve ---

def load_lineage():
    if os.path.exists(EVOL_STATE):
        try:
            with open(EVOL_STATE, encoding="utf-8") as f:
                st = json.load(f)
            return (int(st.get("cycle", 0)), int(st.get("saves", 0)),
                    int(st.get("sessions", 0)), list(st.get("ctx", [])))
        except (json.JSONDecodeError, OSError):
            pass
    return 0, 0, 0, []


def save_lineage(cycle, saves, sessions, ctx):
    tmp = EVOL_STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"cycle": cycle, "saves": saves, "sessions": sessions, "ctx": ctx}, f)
    os.replace(tmp, EVOL_STATE)


def _top_p_filter(logits, top_p=EVOL_TOP_P):
    probs = logits.softmax(-1)
    sp, si = torch.sort(probs, descending=True)
    cum = sp.cumsum(0)
    sp[cum - sp > top_p] = 0.0
    out = torch.zeros_like(probs).scatter(0, si, sp)
    return out / out.sum().clamp_min(1e-12)


@torch.no_grad()
def gen_sample(model, ids, max_new, temp, shadow=False):
    """Autoregressive top-p sampling with repetition penalty; optional shadow
    teacher prefill (honest attention targets) before the student pass."""
    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    if shadow and len(ids) > 1:
        shadow_prefill(model, input_ids)
    global CURRENT_IDS
    CURRENT_IDS = input_ids
    out = model(input_ids=input_ids, use_cache=True, past_key_values=None)
    past = out.past_key_values
    logits = out.logits[0, -1].float()
    generated: list[int] = []
    _t0 = time.time()
    for _ in range(max_new):
        probs = _top_p_filter(logits / temp)
        recent = generated[-REP_WIN:]
        if recent:
            idx = torch.tensor(list(set(recent)), device="cuda", dtype=torch.long)
            probs[idx] /= REP_PEN
            probs = probs / probs.sum().clamp_min(1e-12)
        if len(generated) < MIN_NEW_TOKENS:
            probs[EOS_ID] = 0.0
            probs = probs / probs.sum().clamp_min(1e-12)
        nxt = int(torch.multinomial(probs, 1).item())
        if nxt == EOS_ID:
            break
        generated.append(nxt)
        if len(generated) % 10 == 0:
            global I4X_HIT, I4X_MISS
            dt = time.time() - _t0
            print(
                f"    ...{len(generated)} tok ({dt/len(generated):.2f} s/tok, "
                f"i4x h/m={I4X_HIT}/{I4X_MISS}, deq={len(DEQUANT_CACHE)}, packed={len(PACKED_CACHE)})",
                flush=True,
            )
            I4X_HIT = I4X_MISS = 0
        CURRENT_IDS = torch.tensor([[nxt]], device="cuda", dtype=torch.long)
        out = model(input_ids=CURRENT_IDS, use_cache=True, past_key_values=past)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
    del past, out
    return generated


def _block_ngram(probs, hist, n=NGRAM):
    """Zero out tokens that would complete an already-seen n-gram."""
    if len(hist) >= n - 1:
        tail = hist[-(n - 1):]
        banned = {hist[i + n - 1] for i in range(len(hist) - n + 1) if hist[i:i + n - 1] == tail}
        if banned:
            probs[torch.tensor(sorted(banned), device=probs.device, dtype=torch.long)] = 0.0
    return probs


def _attn_window(session_idx, width):
    """Rotating coverage: session s covers layers [s*width % 43, +width)."""
    start = (session_idx * width) % N_LAYERS
    return sorted((start + i) % N_LAYERS for i in range(width))


def build_pins(prefix_ids):
    """Routing-aware residency: hash layers pin EXACT expert sets for the known
    token stream; topk layers pin by measured prefill usage. Budget-capped."""
    global PINNED, PINNED_BYTES, COLLECT_USAGE
    COLLECT_USAGE = False
    PINNED = set()
    PINNED_BYTES = 0
    cands = []  # (priority, li, k)
    ids_t = torch.tensor(prefix_ids, device="cuda", dtype=torch.long)
    for li in HASH_LAYERS:  # deterministic token-id routing: exact needs
        flat_eids = ROUTER_TID[li][ids_t].reshape(-1).tolist()
        for k in set(flat_eids):
            cands.append((float("inf"), li, k))
    for li in range(N_LAYERS):  # stable topk routing: measured usage
        for k, c in enumerate(USAGE_T[li].tolist()):
            if c > 0:
                cands.append((c, li, k))
    cands.sort(key=lambda t: (0 if t[0] == float("inf") else 1, -t[0]))
    n_pin = 0
    for _, li, k in cands:
        if (li, k) in PINNED:
            continue
        if PINNED_BYTES + 50 * 2**20 > PIN_BUDGET:
            break
        # protect BEFORE materializing: LRU eviction skips pinned keys, but the
        # entry must survive the tight INT4X/DEQUANT caps during the pin sweep
        ckpt = os.path.exists(os.path.join(REDUCED, f"layer_{li}", f"expert_{k}.pt"))
        PINNED.add((li, k))
        if ckpt:
            if get_int4x(li, k) is None:
                PINNED.discard((li, k))
                continue
            if (li, k) not in INT4X:  # checkpoint exists but not int4x mode
                PINNED.discard((li, k))
                continue
        else:
            get_dequant(li, k)
            if (li, k) not in DEQUANT_CACHE:
                PINNED.discard((li, k))
                continue
        PINNED_BYTES += 50 * 2**20
        n_pin += 1
    print(f"pinned {n_pin} experts ({PINNED_BYTES / 2**30:.1f} GB budget {PIN_BUDGET / 2**30:.0f})", flush=True)


@torch.no_grad()
def gen_turn_b(model, state, n_new, temp):
    """One decode turn for B parallel streams, single KV: feed [B,1] per step,
    per-stream sampling (top-p + rep penalty + n-gram block)."""
    B = state["B"]
    gen = [[] for _ in range(B)]
    fed = []
    t0 = time.time()
    global CURRENT_IDS, I4X_HIT, I4X_MISS
    for step in range(n_new):
        nxt_list = [EOS_ID] * B
        for b in range(B):
            if state["finished"][b]:
                continue
            probs = _top_p_filter(state["logits"][b] / temp)
            hist = state["all"][b]
            recent = hist[-REP_WIN:]
            if recent:
                idx = torch.tensor(sorted(set(recent)), device="cuda", dtype=torch.long)
                probs[idx] /= REP_PEN
            probs = _block_ngram(probs, hist)
            if len(gen[b]) < MIN_NEW_TOKENS:
                probs[EOS_ID] = 0.0
            probs = probs / probs.sum().clamp_min(1e-12)
            nxt = int(torch.multinomial(probs, 1).item())
            if nxt == EOS_ID:
                state["finished"][b] = True
            else:
                nxt_list[b] = nxt
                gen[b].append(nxt)
                hist.append(nxt)
        ids = torch.tensor(nxt_list, device="cuda", dtype=torch.long).unsqueeze(1)  # [B, 1]
        CURRENT_IDS = ids
        out = model(input_ids=ids, use_cache=True, past_key_values=state["past"])
        state["past"] = out.past_key_values
        state["logits"] = out.logits[:, -1].float()
        fed.append(nxt_list)
        if (step + 1) % 10 == 0:
            dt = time.time() - t0
            print(
                f"    ...{(step + 1) * B} tok ({dt / ((step + 1) * B):.2f} s/tok, "
                f"i4x h/m={I4X_HIT}/{I4X_MISS})",
                flush=True,
            )
            I4X_HIT = I4X_MISS = 0
    return gen, fed


@torch.no_grad()
def ttt_flush(model, tstate, fed_steps):
    """Teacher prefill (stepwise, feed-aligned) over the turn tokens, then
    fire the RLS updates from the collected student rows."""
    if not fed_steps:
        return
    global CURRENT_IDS, MODE
    ids = torch.tensor(fed_steps, device="cuda", dtype=torch.long).t()  # [B, n] ONE batched prefill
    CURRENT_IDS = ids
    MODE = "teacher"
    try:
        out = model(input_ids=ids, use_cache=True, past_key_values=tstate.get("past"))
        tstate["past"] = out.past_key_values
    finally:
        MODE = "student"
    # attention: collected student rows vs teacher chunks (both step-major)
    n_steps = len(fed_steps)
    for li in list(ATTN_ADAPT.keys()):
        chunks = SHADOW.pop(li, None)
        rows = COLLECT_ATTN.pop(li, None)
        if chunks and rows:
            g_t = torch.cat(chunks)  # [B*n, Fin] stream-major (batched prefill flatten)
            g_s = torch.cat(rows)    # [n*B, Fin] step-major -> reorder to stream-major
            g_s = g_s.view(n_steps, -1, g_s.shape[-1]).transpose(0, 1).reshape(-1, g_s.shape[-1])
            n = min(g_t.shape[0], g_s.shape[0])
            if n > 1:
                if n > ATTN_ROWS_CAP:
                    sel = torch.arange(0, n, n // ATTN_ROWS_CAP, device=g_s.device)
                else:
                    sel = slice(None)
                y_t = (g_t[sel].to(ATTN_ADAPT[li]["w0"].dtype) @ ATTN_ADAPT[li]["w0"].T).float()
                attn_ttt_update(li, g_s[sel], y_t)
                del y_t
    # FFN: top experts by collected rows -> teacher pass -> RLS
    per_layer: dict[int, list] = {}
    for (li, k), v in COLLECT_FFN.items():
        if v["x"]:
            per_layer.setdefault(li, []).append((v["n"], k))
    for li, lst in per_layer.items():
        lst.sort(reverse=True)
        for n, k in lst[:TTT_PER_LAYER]:
            v = COLLECT_FFN[(li, k)]
            x = torch.cat(v["x"])
            h = torch.cat(v["h"])
            y_t = _teacher_y(li, k, x)
            ttt_update(li, k, h, y_t)
            del y_t
    COLLECT_FFN.clear()


def run_evolve(model, tok, ttt_on, save_on, sessions_limit, session_sec, turns, gen_tokens, attn_window_w):
    """Background self-evolution: endless self-talk. The model continues ITS OWN
    stream of thought (context tail persisted in the lineage state - no fixed
    prompts); weights evolve live; lineage consolidates after every session."""
    cycle, saves, sessions, ctx = load_lineage()
    print(f"lineage: cycle={cycle} sessions={sessions} saves={saves} ctx={len(ctx)} tok", flush=True)

    handles = [model.model.layers[li].mlp.register_forward_hook(make_hook(li, ttt_on), with_kwargs=True) for li in range(N_LAYERS)]
    attn_handles: list = []
    print(f"FFN TTT hooks on all {N_LAYERS} layers (experts adapt where int4x checkpoints exist)", flush=True)

    session = 0
    prev_layers: set[int] = set()
    try:
        while sessions_limit == 0 or session < sessions_limit:
            session += 1
            t0 = time.time()
            layers = _attn_window(session, attn_window_w)
            # rotate the attention window: drop adapters of layers that left it
            for h in attn_handles:
                h.remove()
            attn_handles = []
            for li in prev_layers - set(layers):
                ATTN_ADAPT.pop(li, None)  # ~128MB fp32 + bf16 ref per layer
                SHADOW.pop(li, None)
                ATTN_STATE.pop(li, None)  # consolidated at the previous session end
            prev_layers = set(layers)
            for li in layers:
                attn_init(li, model.model.layers[li].self_attn.o_b_proj)
                attn_handles.append(
                    model.model.layers[li].self_attn.o_b_proj.register_forward_hook(
                        make_attn_hook(li, ttt_on), with_kwargs=True
                    )
                )

            # session start: B streams continue the lineage thought (same prefix,
            # diverging via sampling); teacher mirrors the prefix in its own KV
            prefix = [BOS_ID] + ctx[-(EVOL_CTX - 1):]
            ids0 = torch.tensor([prefix] * N_STREAMS, device="cuda", dtype=torch.long)
            tstate: dict = {"past": None}
            if ttt_on and len(prefix) > 1:
                tstate["past"] = shadow_prefill(model, ids0, keep_past=True)
            state = {"B": N_STREAMS, "past": None, "logits": None, "finished": [False] * N_STREAMS,
                     "all": [list(prefix) for _ in range(N_STREAMS)]}
            global CURRENT_IDS, COLLECT_USAGE, USAGE_T
            for li in range(N_LAYERS):
                USAGE_T[li] = torch.zeros(256, dtype=torch.long, device="cuda")
            COLLECT_USAGE = True
            CURRENT_IDS = ids0
            out = model(input_ids=ids0, use_cache=True, past_key_values=None)
            COLLECT_USAGE = False
            state["past"] = out.past_key_values
            state["logits"] = out.logits[:, -1].float()
            del out
            build_pins(prefix)  # routing-aware residency for this session's stream

            fed_all = []
            for turn in range(turns):
                temp = CURIOSITY_TEMP if turn % 2 == 1 else NORMAL_TEMP
                gen, fed = gen_turn_b(model, state, gen_tokens, temp)
                fed_all.extend(fed)
                if ttt_on and ((turn + 1) % FLUSH_EVERY == 0 or turn == turns - 1):
                    ttt_flush(model, tstate, fed_all)
                    fed_all = []
                text = tok.decode(gen[0][-60:]) if gen[0] else ""
                if isinstance(text, bytes):
                    text = text.decode("utf-8", errors="replace")
                snippet = " ".join(text.split())
                alive = sum(1 for f in state["finished"] if not f)
                print(f"  [turn {turn} T={temp} streams={alive}] ...{snippet}", flush=True)
                if alive == 0:
                    break

            ctx = state["all"][0][-EVOL_CTX_KEEP:]  # stream 0 carries the lineage
            global PINNED_BYTES
            PINNED.clear()
            PINNED_BYTES = 0
            n_exp = len(TTT_STATE)
            for (li, k) in list(TTT_STATE.keys()):
                consolidate_and_save(li, k, enabled=save_on)
            for li in list(ATTN_STATE.keys()):
                attn_consolidate_and_save(li, enabled=save_on)
            nsaved = len(SAVE_LOG) + len(ATTN_SAVE_LOG)

            cycle += 1
            sessions += 1
            saves += 1 if nsaved else 0
            save_lineage(cycle, saves, sessions, ctx[-EVOL_CTX_KEEP:])
            SAVE_LOG.clear()
            ATTN_SAVE_LOG.clear()
            dt = time.time() - t0
            print(
                f"[session {session}] {turns} turns, experts-touched={n_exp} "
                f"attn-layers={layers} -> saved {nsaved} | lineage cycle {cycle}, saves={saves} ({dt:.0f}s)",
                flush=True,
            )
            time.sleep(max(1, session_sec - int(dt)))
    finally:
        for h in handles:
            h.remove()
        for h in attn_handles:
            h.remove()
        # do not lose live adaptations on Ctrl+C
        for (li, k) in list(TTT_STATE.keys()):
            consolidate_and_save(li, k, enabled=save_on)
        for li in list(ATTN_STATE.keys()):
            attn_consolidate_and_save(li, enabled=save_on)
        print("evolve loop stopped; live states consolidated", flush=True)


def main():
    try:  # generated text can contain any unicode; console/log pipe is cp1251
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    ttt_on = "--no-ttt" not in sys.argv
    save_on = "--no-save" not in sys.argv
    evolve_on = "--evolve" in sys.argv
    global SAVE_ENABLED
    SAVE_ENABLED = save_on

    print("loading tokenizer...", flush=True)
    tok = gigatoken.Tokenizer.from_json(open(TOKENIZER, "rb").read())
    ids = [BOS_ID] + list(tok.encode(prompt))

    model = setup_model()

    if evolve_on:
        def _arg(name, default, cast=int):
            if name in sys.argv:
                i = sys.argv.index(name)
                if i + 1 < len(sys.argv):
                    return cast(sys.argv[i + 1])
            return default

        run_evolve(
            model, tok,
            ttt_on=ttt_on, save_on=save_on,
            sessions_limit=_arg("--sessions", 0),
            session_sec=_arg("--session-sec", 60),
            turns=_arg("--turns", 6),
            gen_tokens=_arg("--gen-tokens", 16),
            attn_window_w=_arg("--attn-window", 8),
        )
        return
    n_int4x = sum(1 for li in range(N_LAYERS) for k in range(256) if os.path.exists(os.path.join(REDUCED, f"layer_{li}", f"expert_{k}.pt")))
    print(f"int4x checkpoints available: {n_int4x} experts; ttt={'on' if ttt_on else 'off'} save={'on' if save_on else 'off'}", flush=True)
    if os.environ.get("I4X_PREWARM") and not evolve_on:
        prewarm_packed(n_threads=int(os.environ.get("I4X_PREWARM_THREADS", "8")))

    handles = [model.model.layers[li].mlp.register_forward_hook(make_hook(li, ttt_on), with_kwargs=True) for li in range(N_LAYERS)]
    attn_on = "--no-attn" not in sys.argv
    attn_handles = []
    if attn_on:
        for li in TTT_ATTN_LAYERS:
            if li >= N_LAYERS:
                continue
            attn_init(li, model.model.layers[li].self_attn.o_b_proj)
            attn_handles.append(
                model.model.layers[li].self_attn.o_b_proj.register_forward_hook(make_attn_hook(li, ttt_on), with_kwargs=True)
            )
        print(f"attention TTT on layers {TTT_ATTN_LAYERS} (o_b_proj readout)", flush=True)

    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    global CURRENT_IDS
    generated = list(ids)
    t0 = time.time()
    with torch.no_grad():
        if attn_handles and ttt_on:
            shadow_prefill(model, input_ids)  # honest attention targets for the prompt
        CURRENT_IDS = input_ids
        out = model(input_ids=input_ids, use_cache=True, past_key_values=None)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        generated.append(nxt)
        for _ in range(max_new - 1):
            if nxt == EOS_ID:
                break
            CURRENT_IDS = torch.tensor([[nxt]], device="cuda", dtype=torch.long)
            out = model(input_ids=CURRENT_IDS, use_cache=True, past_key_values=past)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax().item())
            generated.append(nxt)

    for h in handles:
        h.remove()
    for h in attn_handles:
        h.remove()

    dt = time.time() - t0
    n_tok = len(generated) - len(ids)
    print(f"generated {n_tok} tokens in {dt:.1f}s ({n_tok / max(dt, 1e-9):.2f} tok/s)", flush=True)
    print(f"i4x hit/miss: {I4X_HIT}/{I4X_MISS} (miss-rate {I4X_MISS / max(I4X_HIT + I4X_MISS, 1) * 100:.0f}%)", flush=True)
    if PROF_DECODE:
        tot = sum(PROF_T.values())
        print("hook profile: " + "  ".join(f"{k}={v:.2f}s" for k, v in PROF_T.items()) + f"  total={tot:.2f}s", flush=True)

    print("consolidating TTT experts...", flush=True)
    for (li, k) in list(TTT_STATE.keys()):
        consolidate_and_save(li, k, enabled=save_on)
    for li in list(ATTN_STATE.keys()):
        attn_consolidate_and_save(li, enabled=save_on)
    if ATTN_SAVE_LOG:
        print(f"attention saved {len(ATTN_SAVE_LOG)} layer(s): " + "; ".join(ATTN_SAVE_LOG), flush=True)
    if SAVE_LOG:
        print(f"saved {len(SAVE_LOG)} expert(s): " + "; ".join(SAVE_LOG), flush=True)
    else:
        print("no expert met the save threshold", flush=True)

    text = tok.decode(generated)
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    print("=== OUTPUT ===", flush=True)
    print(text, flush=True)


if __name__ == "__main__":
    main()
