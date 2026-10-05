"""Merge admission gate (MacroCycleV2.lean R175 kernel 1, runtime wire).

Before a merge, each sibling's logits are measured on the gate windows
(the SAME fixed 2048-token tail windows ``gate_score.py`` uses) and
admitted by the variance gate: an expert whose CENTERED spread ``R_i``
(deviation from the weighted pool mean -- the measurable R106 states the
PoE bound in) overshoots the certified admission bar ``R_bar`` is
repaired by ``tau_i = R_i/R_bar``: its post-gate spread is exactly
``R_bar``, its PoE slack exactly ``R_bar^2/8``
(``variance_gate_normalize``).

In WEIGHT space the repair is exact and one line: logits are linear in
the head projection, so scaling the head's ``logit_scale`` by
``1/tau_i`` reproduces the temperature division on every token.
``--write-dir`` emits scaled COPIES of the expert checkpoints; the
inputs on disk are never touched.

Usage:
    python scripts/merge_admission.py --experts a.pt b.pt c.pt \
        --r-bar 4.0 [--write-dir checkpoints/gen7_admitted]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hagi.config import load_config  # noqa: E402
from hagi.model.merge import build_model_from_payload  # noqa: E402
from hagi.train.checkpoint import load_payload, save_checkpoint  # noqa: E402
from hagi.train.loop import configure_runtime  # noqa: E402


def gate_batches(cfg, n_tokens: int = 2048, seq: int = 1024):
    """The canonical windows: last-2M tail of each weighted corpus."""
    data_dir = Path(cfg.train.data.data_dir)
    for name, w in sorted((cfg.train.data.weights or {}).items()):
        p = data_dir / f"{name}.compact.bin"
        if not p.exists():
            continue
        total = p.stat().st_size // 4
        start = max(0, total - 2_000_000)
        with p.open("rb") as fh:
            fh.seek(start * 4)
            t = np.frombuffer(fh.read(n_tokens * 4), dtype=np.uint32).astype(np.int64)
        ids = torch.from_numpy(t[:n_tokens]).reshape(2, seq)
        yield name, w, ids[:, :-1], ids[:, 1:]


@torch.no_grad()
def _probe_positions(model, x: torch.Tensor, stride: int = 64) -> torch.Tensor:
    """Logits at strided positions over the window, fp32."""
    o = model(x, None)
    pos = list(range(0, x.shape[1], stride))
    h = o.hidden[:, pos, :]
    return model.head.logits(h.reshape(-1, h.shape[-1])).float()


def centered_spreads(
    expert_logit_rows: list[torch.Tensor],
) -> torch.Tensor:
    """R106 centered spread per expert: deviation from the pooled mean.

    Args:
        expert_logit_rows: one ``[N, V]`` logits tensor per expert,
            row-aligned across experts (same tokens, same positions).

    Returns:
        ``[E]`` tensor: each expert's mean (over rows) centered range.
    """
    stacked = torch.stack(expert_logit_rows)  # [E, N, V]
    mean = stacked.mean(0, keepdim=True)  # equal weights pool
    devs = stacked - mean
    per_row = devs.max(dim=-1).values - devs.min(dim=-1).values  # [E, N]
    return per_row.mean(-1)  # [E]


def repair_tau(r: float, r_bar: float) -> float:
    """``variance_gate_normalize``: exact repair temperature, 1.0 in range."""
    return 1.0 if r <= r_bar else r / r_bar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experts", nargs="+", required=True)
    ap.add_argument("--r-bar", type=float, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--write-dir", default=None,
                    help="write tau-scaled expert checkpoint copies here "
                         "(merge_init step-0 format); inputs are not touched")
    ap.add_argument("--config", default=None,
                    help="merged-joint config (needed only with --write-dir)")
    args = ap.parse_args()

    configure_runtime()

    # 1. Measure every expert on the shared canonical windows.
    shared = None  # list of (name, weight, x, y)
    per_expert: list[dict[str, torch.Tensor]] = []
    payloads = []
    for path in args.experts:
        payload = load_payload(Path(path), args.device)
        payloads.append(payload)
        from hagi.config import config_from_dict

        tcfg = config_from_dict(payload["config"])
        n_mixers = int(getattr(tcfg.merge, "n_mixers", 1))
        model = build_model_from_payload(
            tcfg, payload["model"], n_mixers=n_mixers,
            mixer_init_scale=tcfg.merge.mixer_init_scale,
            device=args.device,
        )
        model.eval()
        if shared is None:
            shared = list(gate_batches(tcfg))
        per_expert.append(
            {name: _probe_positions(model, x.to(args.device))
             for name, _, x, _ in shared}
        )
        del model

    names = [n for n, *_ in shared]
    n_e = len(args.experts)

    # 2. Per-expert centered spread, averaged over corpora (equal corpus
    #    weight: the admission question is about the expert, not the mix).
    tau_sum = [0.0] * n_e
    for name in names:
        rows = [pe[name] for pe in per_expert]
        r = centered_spreads(rows)
        for i, ri in enumerate(r.tolist()):
            tau_sum[i] += repair_tau(ri, args.r_bar)
    taus = [t / len(names) for t in tau_sum]

    print("merge admission (R175 variance gate):")
    for path, tau in zip(args.experts, taus):
        tag = "ok" if tau <= 1.0 + 1e-9 else "SCALED"
        print(f"  {path}  tau={tau:.4f}  [{tag}]")

    # 3. Optional: write scaled copies. The repair acts on the head's
    #    logit_scale (logits are linear in the projection; scaling the
    #    temperature knob reproduces the division on every token).
    if args.write_dir is not None:
        if not args.config:
            raise SystemExit("--write-dir needs --config (the joint config)")
        joint_cfg = load_config(args.config)
        outdir = Path(args.write_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        from hagi.config import config_from_dict
        from hagi.model.factory import build_model_for_config
        from hagi.train.checkpoint import load_model

        for path, tau, payload in zip(args.experts, taus, payloads):
            tcfg = config_from_dict(payload["config"])
            model = build_model_for_config(tcfg).to(args.device)
            load_model(path, model, args.device)
            with torch.no_grad():
                if hasattr(model.head, "logit_scale"):
                    model.head.logit_scale.div_(tau)
                else:
                    raise SystemExit(
                        f"{path}: head has no logit_scale; the weight-space "
                        "repair needs the scalar temperature knob"
                    )
            sub = outdir / Path(path).stem
            save_checkpoint(model, joint_cfg, 0, sub, keep_last=1,
                            name="step-0000000.pt")
            print(f"  wrote {sub / 'step-0000000.pt'} (tau={tau:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
