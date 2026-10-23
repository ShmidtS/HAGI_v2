"""End merge: EMA-банк + покомпонентный sweep с held-out приёмкой.

Порт end-merge из qlabs-eng/noise train/val/run.py (record #11):
вдоль траектории ft держатся экспоненциальные средние весов; после
последнего апдейта каждая часть (тензор) двигается к одному из средних
на долю f при масштабе s, и ход принимается ТОЛЬКО если held-out лосс
на фиксированном окне падает. Порядок: лучший одиночный кандидат ->
полный координатный sweep -> повторный sweep по лучшим 8 частям.

Отличия от оригинала (наши артефакты вместо их живых EMA):
- банк = экспоненциально взвешенное среднее доступных step-*.pt
  (новейший — вес 0.5, далее /2; нормализация), best.pt = финал;
- без разложения на top-input-covariance направления (наши части —
  целые тензоры; 9-EMA банк даёт то же сглаживание траектории);
- приёмка: точный gate-CE на фиксированных tail-окнах (тот же
  протокол, что saturation/divergence гварды), принимать строго
  при снижении; финальное решение всё равно за certified A/B.

Сертификационная основа: ход меняет веса только при измеренном
снижении лосса на held-out окне (consolidation_gate / certified
accept); отказ = нулевой ход (function-preserving по построению).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hagi.config import load_config  # noqa: E402
from hagi.model.model import HAGI  # noqa: E402
from hagi.train.checkpoint import load_model  # noqa: E402
from hagi.train.loop import cast_model, configure_runtime  # noqa: E402

FRACTIONS = (0.5, 1.0)
SCALES = (0.75, 1.0, 1.25)


def load_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def ema_bank(paths: list[Path]) -> dict[str, torch.Tensor]:
    """Экспоненциально взвешенное среднее state_dict'ов (новейший вес 1/2)."""
    paths = sorted(paths, key=lambda p: p.stem)  # step-0000100 < step-0000600
    weights = [0.5 ** (len(paths) - 1 - i) for i in range(len(paths))]
    total = sum(weights)
    bank: dict[str, torch.Tensor] = {}
    for key in load_state(paths[0]):
        acc = None
        for p, w in zip(paths, weights):
            t = load_state(p)[key].float() * (w / total)
            acc = t if acc is None else acc + t
        bank[key] = acc
    return bank


def gate_batches(cfg, device) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Фиксированные tail-окна по источникам mix (тот же протокол, что _gate_ce)."""
    batches = []
    data_dir = Path(cfg.train.data.data_dir)
    weights = []
    for name, w in sorted((cfg.train.data.weights or {}).items()):
        p = data_dir / f"{name}.compact.bin"
        if not p.exists():
            continue
        total = p.stat().st_size // 4
        start = max(0, total - 2_000_000)
        with p.open("rb") as fh:
            fh.seek(start * 4)
            arr = np.frombuffer(fh.read(2048 * 4), dtype=np.uint32).astype(np.int64)
        ids = torch.from_numpy(arr[:2048]).reshape(2, 1024)
        batches.append((ids[:, :-1].to(device), ids[:, 1:].to(device)))
        weights.append(float(w))
    wsum = sum(weights) or 1.0
    return [(x, y, w / wsum) for (x, y), w in zip(batches, weights)]


@torch.no_grad()
def gate_ce(model, windows) -> float:
    total = 0.0
    for x, y, w in windows:
        out = model(x, y)
        f = out.hidden.reshape(-1, out.hidden.shape[-1])
        total += w * float(model.head.exact_loss(f, y.reshape(-1)))
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--out", default=None, help="итоговый state_dict (default: <dir>/end_merged.pt)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    configure_runtime()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    final_path = ckpt_dir / "best.pt"
    step_paths = sorted(ckpt_dir.glob("step-*.pt"))
    if len(step_paths) < 2:
        print(f"end-merge: need >=2 step checkpoints, found {len(step_paths)}; skip")
        return 1

    model = HAGI(cfg).to(device)
    start_step, _ = load_model(str(final_path), model, str(device))
    cast_model(model, cfg.train.precision)
    windows = gate_batches(cfg, device)

    final_sd = {k: v.float() for k, v in load_state(final_path).items()}
    bank = ema_bank(step_paths)
    base_ce = gate_ce(model, windows)
    print(f"base gate_ce={base_ce:.4f} (best.pt step {start_step}, bank={len(step_paths)} ckpts)")

    param_names = [n for n, _ in model.named_parameters()]
    moves: dict[str, tuple[float, float]] = {}

    def try_move(name: str, frac: float, scale: float) -> float:
        """Применить ход к тензору, вернуть gate_ce (откат при отсутствии выигрыша)."""
        module_key = name
        final_t = final_sd[module_key]
        bank_t = bank[module_key].to(final_t.device)
        restored = final_t.detach().clone()
        with torch.no_grad():
            param = dict(model.named_parameters())[name]
            param.copy_(restored)
            target = final_t + (bank_t - final_t) * (frac * scale)
            param.copy_(target.to(param.dtype))
        ce = gate_ce(model, windows)
        if ce < current["ce"] - 1e-4:
            current["ce"] = ce
            return ce
        with torch.no_grad():
            dict(model.named_parameters())[name].copy_(restored.to(param.dtype))
        return current["ce"]

    current = {"ce": base_ce}

    # (1) лучший одиночный кандидат по всем частям
    best_single = None
    for name in param_names:
        if name not in bank or final_sd[name].shape != bank[name].shape:
            continue
        for frac in FRACTIONS:
            for scale in SCALES:
                ce = try_move(name, frac, scale)
                if best_single is None or ce < best_single[0]:
                    best_single = (ce, name, frac, scale)
    if best_single is None:
        print(json.dumps({"base": base_ce, "final": base_ce, "moves": 0, "verdict": "no-improvement"}))
        return 0
    print(f"single best: {best_single[1]} f={best_single[2]} s={best_single[3]} -> {best_single[0]:.4f}")

    # (2) полный координатный sweep (порядок — от худшего вклада к лучшему)
    improved: list[tuple[float, str, float, float]] = []
    for name in param_names:
        if name not in bank or final_sd[name].shape != bank[name].shape:
            continue
        before = current["ce"]
        best_here = (before, 0.0, 0.0)
        for frac in FRACTIONS:
            for scale in SCALES:
                ce = try_move(name, frac, scale)
                if ce < best_here[0]:
                    best_here = (ce, frac, scale)
        if best_here[1] > 0:
            moves[name] = (best_here[1], best_here[2])
            improved.append((before - best_here[0], name, best_here[1], best_here[2]))

    # (3) повторный sweep по лучшим 8 частям
    improved.sort(reverse=True)
    for _, name, _, _ in improved[:8]:
        best_here = (current["ce"], 0.0, 0.0)
        for frac in FRACTIONS:
            for scale in SCALES:
                ce = try_move(name, frac, scale)
                if ce < best_here[0]:
                    best_here = (ce, frac, scale)
        moves[name] = (best_here[1], best_here[2]) if best_here[1] > 0 else moves.get(name, (0.0, 0.0))

    final_ce = gate_ce(model, windows)
    out_path = Path(args.out) if args.out else ckpt_dir / "end_merged.pt"
    torch.save(
        {"model": {k: v.cpu() for k, v in model.state_dict().items()}, "step": start_step,
         "meta": {"end_merge": True, "base_gate_ce": base_ce, "final_gate_ce": final_ce}},
        out_path,
    )
    report = {
        "base": base_ce, "final": final_ce, "delta": final_ce - base_ce,
        "moves": {k: {"frac": v[0], "scale": v[1]} for k, v in moves.items() if v[0] > 0},
        "verdict": "improved" if final_ce < base_ce - 1e-4 else "no-improvement",
        "out": str(out_path),
    }
    print(json.dumps(report, indent=2))
    with open(ckpt_dir / "end_merge_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
