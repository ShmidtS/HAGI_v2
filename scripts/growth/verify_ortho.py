"""verify_ortho — сертификат rank_orthogonal_growth (RankGrowth.lean:56).

Теорема: rank(stack M R) = rank M + k, если (а) k новых строк R
линейно независимы, (б) каждая строка R ортогональна каждой строке
M (OrthRows = все dot-произведения 0). Тогда обновление в span R
имеет нулевую интерференцию со старыми направлениями при ε=0
(span_growth_zero_interference, :141) — новые домены не трогают
старые.

Практика (измерение, не доказательство): для каждого factorized
тензора считаем max |cos| между строками ROOT-факторов нового
поколения и волокнами-дельтами (row-space листьев). Малый max|cos|
≈ условия теоремы ≈ слияние безопасно. Гейт: max|cos| < ORTH_MAX.

Usage:
  python scripts/growth/verify_ortho.py --merged checkpoints/rnn_gen2_merged
"""
from __future__ import annotations

import argparse
import torch

# Порог гейта: при |cos| > ORTH_MAX направление не ортогонально
# старому row-space — интерференция не сертифицирована как нулевая.
ORTH_MAX = 0.30


def _rows(t: torch.Tensor) -> torch.Tensor:
    """[out, in] → плоские строки-векторы; 1D → один вектор."""
    return t.reshape(t.shape[0], -1) if t.ndim >= 2 else t.reshape(1, -1)


def max_abs_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |cos| между любой строкой a и любой строкой b (нормировано)."""
    a = torch.nn.functional.normalize(_rows(a).float(), dim=-1)
    b = torch.nn.functional.normalize(_rows(b).float(), dim=-1)
    cos = a @ b.T
    return cos.abs().max().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="каталог с root.pt + fibers.pt")
    ap.add_argument("--orth-max", type=float, default=ORTH_MAX)
    args = ap.parse_args()
    from pathlib import Path

    d = Path(args.merged)
    root = torch.load(d / "root.pt", map_location="cpu", weights_only=False)
    fibers = torch.load(d / "fibers.pt", map_location="cpu", weights_only=False)["fibers"]
    root_sd = root["model"] if isinstance(root, dict) and "model" in root else root

    worst = 0.0
    worst_name = ""
    n_checked = 0
    for name, per_expert in fibers.items():
        base = root_sd.get(name)
        if base is None or not isinstance(per_expert, list):
            continue
        for exp, t in enumerate(per_expert):
            if not isinstance(t, torch.Tensor) or t.numel() < 2:
                continue
            c = max_abs_cos(t, base)
            n_checked += 1
            if c > worst:
                worst, worst_name = c, f"{name}[{exp}]"
    # Интерпретация (измерено на rnn_gen1/2: |cos| 0.42–0.89):
    # волокна НЕ ортогональны row-space корня — условия
    # rank_orthogonal_growth (нулевая интерференция ε=0) НЕ
    # выполняются для latent merge. Слиянием управляет более
    # слабый per-tensor гейт «twoGap против цены» (merge_price),
    # что соответствует проектной линии (Oracle 2026-10-12:
    # эквивалентность живёт в волокнах, экономика — в гейте).
    verdict = "zero-interference CERTIFIED" if worst < args.orth_max else \
        "zero-interference NOT certified — governed by gap-price gate"
    print(f"tensors checked: {n_checked}")
    print(f"max |cos| fiber->root-row-space: {worst:.4f} at {worst_name}")
    print(f"rank_orthogonal_growth (<{args.orth_max}): {verdict}")


if __name__ == "__main__":
    main()
