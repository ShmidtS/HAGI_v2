"""Per-expert distillation: each HAGI expert learns one DeepSeek expert.

DeepSeek's routed experts are output-orthogonal (shared-Q residual 45-57%),
so each expert covers a non-overlapping direction in the output space. The
router has therefore already *sorted* the token stream by specialization:
``(x_k, y_k)`` from one expert is a clean, pre-sorted dataset, and the HAGI
expert that reproduces it can be trained independently of every other expert
(no cross-expert coordination, embarrassingly parallel).

Data (already collected, AR pass of the full MoE):
    ``checkpoints_dsv4/pod_accurate/acts_layer{L}.pt`` = ``{k: (x_k, y_k)}``
    43 layers, ~256 experts/layer, 9580 experts, 774K samples total.
    ``x_k, y_k`` are ``[n, 4096]`` float32.

Student per expert (one HAGI block, the growth principle):
    ``in_proj [4096 -> n_blocks*H]``
    ``n_blocks`` ternary SwiGLU ``FeedForward`` (H=128, inter=192, BitNet
    b1.58) glued block-diagonally,
    ``out_proj [n_blocks*H -> 4096]``.
With ``n_blocks=1`` it is exactly one HAGI expert per DeepSeek expert; a
growth sweep (``--blocks 1,2,4,...``) measures the residual-vs-size curve.

Two optimizers supported:
    ``--optim adam``  AdamW on everything (matches the existing probe).
    ``--optim muon``  Muon on 2D channel weights (``is_channel_weight``) +
                      AdamW on 1D gains/scales — the HAGI training split.
The ternary master is trained in fp32; BitLinear ternarizes in the forward.

Resume: an expert whose checkpoint exists is skipped, so a power-loss restart
continues rather than redoing finished experts. ``--only-layer`` / ``--only``
run a single (layer, expert) for smoke tests.

Usage:
    python scripts/dsv4_distill_experts.py [layers] [blocks] [steps] [batch]
    # layers: "all" | "0,1,2" | "0-5"
    python scripts/dsv4_distill_experts.py all 1 300 1024 --optim muon
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from hagi.model.ffn import FeedForward

ACTS_DIR = "C:/HAGI_v2/checkpoints_dsv4/pod_accurate"
OUT_DIR = "C:/HAGI_v2/dsv4_distilled_experts"

DIM = 4096
H = 128
INTER = 192
N_LAYERS = 43


def parse_layers(spec: str) -> list[int]:
    """Parse ``all`` | ``0,1,2`` | ``0-5`` | ``0-2,5``."""
    if spec == "all":
        return list(range(N_LAYERS))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def load_layer_acts(layer: int) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Load the whole ``{k: (x_k, y_k)}`` dict for one layer (loaded once).

    File is ~0.5-0.7 GB fp32; load it once per layer and iterate experts in
    memory instead of re-reading the file per expert.
    """
    fp = os.path.join(ACTS_DIR, f"acts_layer{layer}.pt")
    d = torch.load(fp, map_location="cpu", weights_only=False)
    return {int(k): (x.float(), y.float()) for k, (x, y) in d.items()}


class ExpertStudent(nn.Module):
    """One HAGI expert: in_proj -> block-diagonal ternary SwiGLU -> out_proj.

    Mirrors ``dsv4_distill_hagi.HAGIGlue`` but fixed ``teacher_dim=4096`` and a
    clearer name: ``n_blocks`` HAGI experts glued block-diagonally (the growth
    axis). Each block sees only its own H-slice of the hidden stream, which is
    exactly HAGI's merge discipline — no block reads another's slice.
    """

    def __init__(self, n_blocks: int = 1, hidden: int = H, inter: int = INTER) -> None:
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.hidden = hidden
        self.in_proj = nn.Linear(DIM, self.n_blocks * hidden, bias=False)
        self.in_proj.is_channel_weight = True  # 2D hidden-mixing -> Muon group
        self.blocks = nn.ModuleList(
            [FeedForward(hidden, inter, use_ternary=True, init_orthogonal=True)
             for _ in range(self.n_blocks)]
        )
        self.out_proj = nn.Linear(self.n_blocks * hidden, DIM, bias=False)
        self.out_proj.is_channel_weight = True  # 2D hidden-mixing -> Muon group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x).view(-1, self.n_blocks, self.hidden)
        outs = [self.blocks[i](h[:, i]) for i in range(self.n_blocks)]
        h = torch.stack(outs, dim=1).reshape(-1, self.n_blocks * self.hidden)
        return self.out_proj(h)


def build_optimizer(student: ExpertStudent, optim: str, lr: float):
    """AdamW everywhere, or Muon on channel weights + AdamW on 1D gains."""
    if optim == "adam":
        return torch.optim.AdamW(student.parameters(), lr=lr)

    from hagi.train.optim import Muon, _muon_parameters

    muon_params = _muon_parameters(student)
    muon_ids = {id(p) for p in muon_params}
    rest = [p for p in student.parameters() if id(p) not in muon_ids]
    muon = Muon(muon_params, lr=lr, momentum=0.95, nesterov=True, ns_steps=5)
    adamw = torch.optim.AdamW(rest, lr=lr)
    for group in muon.param_groups:
        group["_muon"] = True

    class _Hybrid:
        def zero_grad(self, set_to_none: bool = True) -> None:
            muon.zero_grad(set_to_none=set_to_none)
            adamw.zero_grad(set_to_none=set_to_none)

        def step(self) -> None:
            muon.step()
            adamw.step()

    return _Hybrid()


def distill_expert(
    student: ExpertStudent,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
    bs: int,
    optim: str,
    log: int,
) -> float:
    """Train one student on one expert's (x_k, y_k); return final residual %."""
    opt = build_optimizer(student, optim, lr)
    n = x.shape[0]
    if n == 0:
        raise ValueError("empty expert activations")
    teacher_energy = float((y.float() ** 2).mean())
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), device=x.device)
        loss = F.mse_loss(student(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log and (step + 1) % log == 0:
            print(f"      step {step + 1}/{steps}: loss={float(loss.detach()):.6f}",
                  flush=True)
    with torch.no_grad():
        mse = float(F.mse_loss(student(x), y))
    return mse / max(teacher_energy, 1e-9) * 100.0


def save_expert(student: ExpertStudent, layer: int, k: int, residual_pct: float) -> str:
    out_dir = os.path.join(OUT_DIR, f"layer_{layer}")
    os.makedirs(out_dir, exist_ok=True)
    sd = {kk: v.detach().cpu().to(torch.bfloat16) for kk, v in student.state_dict().items()}
    path = os.path.join(out_dir, f"expert_{k}.pt")
    torch.save({"state_dict": sd, "residual_pct": residual_pct, "layer": layer,
                "expert": k, "n_blocks": student.n_blocks}, path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-expert HAGI distillation from DeepSeek experts")
    ap.add_argument("layers", nargs="?", default="all")
    ap.add_argument("blocks", nargs="?", default="1")
    ap.add_argument("steps", nargs="?", default="300", type=int)
    ap.add_argument("batch", nargs="?", default="1024", type=int)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optim", choices=("adam", "muon"), default="adam")
    ap.add_argument("--log", type=int, default=100)
    ap.add_argument("--only-layer", type=int, default=None)
    ap.add_argument("--only", type=int, default=None, help="expert id for --only-layer")
    ap.add_argument("--force", action="store_true", help="re-distill even if saved")
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    if args.only_layer is not None:
        layers = [args.only_layer]

    # blocks: "1" or "1,2,4" growth sweep
    blocks = [int(b) for b in str(args.blocks).split(",") if b.strip()]

    print(f"layers={layers}, blocks={blocks}, steps={args.steps}, batch={args.batch}, "
          f"optim={args.optim}", flush=True)
    t_start = time.time()

    for li in layers:
        acts = load_layer_acts(li)
        experts = sorted(acts.keys())
        if args.only is not None:
            experts = [k for k in experts if k == args.only]
        print(f"\nlayer {li}: {len(experts)} experts", flush=True)
        for k in experts:
            out_path = os.path.join(OUT_DIR, f"layer_{li}", f"expert_{k}.pt")
            # Growth sweep writes one file per (blocks[-1]) config; a simpler
            # resume check: re-distill only when --force or no file exists.
            if os.path.exists(out_path) and not args.force:
                continue
            x, y = acts[k]
            x = x.cuda()
            y = y.cuda()
            n = x.shape[0]
            for nb in blocks:
                student = ExpertStudent(n_blocks=nb).cuda()
                n_params = sum(p.numel() for p in student.parameters())
                t0 = time.time()
                residual = distill_expert(student, x, y, args.steps, args.lr,
                                          args.batch, args.optim, args.log)
                path = save_expert(student, li, k, residual)
                print(f"  expert {k:>3} | blocks={nb} | params={n_params/1e6:.2f}M | "
                      f"n={n} | residual={residual:.2f}% | {time.time()-t0:.1f}s | "
                      f"{os.path.basename(path)}", flush=True)
                del student
                torch.cuda.empty_cache()
            del x, y
            torch.cuda.empty_cache()
        del acts
        torch.cuda.empty_cache()

    print(f"\nDone in {time.time()-t_start:.1f}s. Checkpoints -> {OUT_DIR}/",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
