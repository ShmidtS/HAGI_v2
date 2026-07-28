"""One-off SDPA backend probe + isolated forward/backward benchmark.

Compares repeat_kv vs enable_gqa=True at the small-config scale AND the
google.yaml scale, and reports which SDPA backend fires for each path.
Run: .venv/Scripts/python.exe scripts/_probe_sdpa.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
sys.path.insert(0, r"C:\HAGI_v2\src")

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
dt = torch.bfloat16


def backends():
    return {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_eff": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
    }


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, n_kv, t, hdd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, hdd).reshape(b, n_kv * n_rep, t, hdd)


def bench(B, T, nq, nkv, hd, n_iter=30, label=""):
    q = torch.randn(B, nq, T, hd, device=dev, dtype=dt, requires_grad=True)
    k = torch.randn(B, nkv, T, hd, device=dev, dtype=dt, requires_grad=True)
    v = torch.randn(B, nkv, T, hd, device=dev, dtype=dt, requires_grad=True)
    go = torch.randn(B, nq, T, hd, device=dev, dtype=dt)

    # warmup + which-backend detection via context manager
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def which(callable_fn):
            for bk in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION):
                try:
                    with sdpa_kernel([bk]):
                        callable_fn()
                    return bk.name
                except Exception:
                    continue
            return "none"

        def gqa_call():
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            out.backward(go)

        def rep_call():
            kr = repeat_kv(k, nq // nkv)
            vr = repeat_kv(v, nq // nkv)
            out = F.scaled_dot_product_attention(q, kr, vr, is_causal=True)
            out.backward(go)

        bk_gqa = which(gqa_call)
        bk_rep = which(rep_call)
    except Exception as e:
        bk_gqa = f"err:{type(e).__name__}"
        bk_rep = f"err:{type(e).__name__}"

    def time_call(setup, call, n):
        ts = []
        for _ in range(n):
            q.grad = k.grad = v.grad = None
            setup()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            call()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        # drop slowest 3 (autotune tails) -> median of the rest
        return ts[3:n - 1] if n > 6 else ts

    def gqa_setup():
        pass

    def gqa_fwd_bwd():
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        out.backward(go)

    def rep_setup():
        pass

    def rep_fwd_bwd():
        kr = repeat_kv(k, nq // nkv)
        vr = repeat_kv(v, nq // nkv)
        out = F.scaled_dot_product_attention(q, kr, vr, is_causal=True)
        out.backward(go)

    t_gqa = time_call(gqa_setup, gqa_fwd_bwd, n_iter)
    t_rep = time_call(rep_setup, rep_fwd_bwd, n_iter)
    med = lambda xs: sum(xs) / len(xs)
    g = med(t_gqa)
    r = med(t_rep)
    print(f"[{label}] B={B} T={T} nq={nq} nkv={nkv} hd={hd} n_rep={nq//nkv}")
    print(f"  backend_gqa={bk_gqa}  backend_rep={bk_rep}")
    print(f"  gqa  med={g:.3f}ms  rep med={r:.3f}ms  speedup_rep_over_gqa={g/r:.3f}x")
    print(f"  gqa_samples={[round(x,2) for x in t_gqa[:5]]}... rep_samples={[round(x,2) for x in t_rep[:5]]}...")


def main():
    print("=== SDPA backends enabled ===")
    print(backends())
    print()
    # small config (google_small.yaml): H=512, T=2048, 8q/2kv, hd=64
    bench(1, 2048, 8, 2, 64, n_iter=40, label="SMALL (google_small)")
    torch.cuda.empty_cache()
    print()
    # google.yaml: H=2048, T=8192, 32q/8kv, hd=64
    bench(1, 8192, 32, 8, 64, n_iter=30, label="LARGE (google.yaml)")


if __name__ == "__main__":
    main()