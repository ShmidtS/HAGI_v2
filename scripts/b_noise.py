"""B_noise / B* measurement (reviewer round-4 §30; McCandlish et al. 2018).

E||g_B||^2 = ||G||^2 + tr(Sigma)/B  ->  from two batch sizes B_s < B_b:
  ||G||^2   = (B_b*||g_b||^2 - B_s*||g_s||^2) / (B_b - B_s)
  tr(Sigma) = (||g_s||^2 - ||g_b||^2) / (1/B_s - 1/B_b)
  B_noise   = tr(Sigma) / ||G||^2
Then B* = sqrt(B_noise * t0 / c) from step timings: T(B) ~ (1 + B_n/B)(t0 + cB)
=> t0/c estimated from two timing points.

Averages over `steps` steps; gradient norms in fp32 over ALL trainable
params (tables dominate at leaf scale -- that is the honest global g).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hagi.config import load_config  # noqa: E402
from hagi.model.factory import build_model_for_config  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402


def grad_sq_norm(model: torch.nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += float(p.grad.double().pow(2).sum())
    return s


def run(path: str, b_small: int = 8, b_big: int = 32, steps: int = 6) -> int:
    configure_runtime()
    device = "cuda"
    cfg = load_config(path)
    model = build_model_for_config(cfg).to(device)
    # fresh leaf init; no need for trained weights -- B_noise is measured
    # on the loss landscape the model actually sits on early in training.
    model = model.to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    seq = cfg.model.attention.max_seq_len
    V = cfg.model.vocab_size
    torch.manual_seed(0)

    def batch(n: int):
        return torch.randint(0, V, (n, seq), device=device)

    norms, times = {}, {}
    for tag, n in (("s", b_small), ("b", b_big)):
        sq, tt = [], []
        for k in range(steps + 2):
            ids = batch(n)
            t0 = time.perf_counter()
            out = model(ids[:, :-1].contiguous(), ids[:, 1:].contiguous())
            loss = out.loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if k >= 2:  # skip warmup
                sq.append(grad_sq_norm(model))
                tt.append(dt)
        norms[tag] = sum(sq) / len(sq)
        times[tag] = sum(tt) / len(tt)
        print(f"B={n:3d}: mean||g||^2={norms[tag]:.4e}  step={times[tag]*1000:7.1f} ms")

    G2 = (b_big * norms["b"] - b_small * norms["s"]) / (b_big - b_small)
    trS = (norms["s"] - norms["b"]) / (1.0 / b_small - 1.0 / b_big)
    G2, trS = max(G2, 1e-30), max(trS, 0.0)
    b_noise = trS / G2
    print(f"\n||G||^2 = {G2:.3e} | tr Sigma = {trS:.3e} | B_noise = {b_noise:.1f} tokens-ish rows")
    # timings: T(B) = t0' + c' * B (linear in B for the fwd+bwd piece).
    c = (times["b"] - times["s"]) / (b_big - b_small)
    t0 = times["s"] - c * b_small
    # B* in units of batch rows per optimizer step (B in the formula is
    # rows per step; B_noise above is in the same row units).
    b_star = (max(b_noise, 1e-9) * t0 / max(c, 1e-9)) ** 0.5
    print(f"t0 = {t0*1000:.1f} ms | c = {c*1000:.2f} ms/row | B* = sqrt(B_n*t0/c) = {b_star:.1f} rows")
    print(f"(current config batch_size = {cfg.train.batch_size})")
    from hagi.model.formal import optimal_batch
    print(f"formal.optimal_batch check: {optimal_batch(b_noise, t0, c):.1f}")
    return 0


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dbridge_leaf_s1.yaml"
    raise SystemExit(run(cfg_path))
