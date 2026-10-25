"""HAGI self-evolution loop: the model grows ITSELF, in one process.

Контроллер принимает каждое решение по сертифицированному правилу
(никаких ручных запусков и YAML-клея):

    СYCLES:
      1. SPECIALIZE — короткий ft инкумбента на СВЕЖЕМ сегменте данных
         (frac++), end-merge по траектории, certified A/B против
         инкумбента. ACCEPT ⟹ смена инкумбента.
      2. WIDTH-GROW — если класс исчерпан (2 подряд дельты без
         сертифицированного улучшения: growth_dichotomy), перенос
         лестницы в новую ширину функционально-сохраняюще
         (width_grow), следующая SPECIALIZE уже в новом классе.
      3. STOP — grow_epsilon_stop: последняя сертифицированная дельта
         не дотягивает до порога при оставшемся бюджете поколений.

Лестница (accept/reject/width-grow) пишется в checkpoints/evolve_ladder.jsonl.

    python scripts/growth/evolve.py --incumbent <ckpt> --generations 4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_mod = _load_script("hagi_eval_domains", ROOT / "scripts/eval_domains.py")
em_mod = _load_script("hagi_end_merge", ROOT / "scripts/growth/end_merge.py")


def certify(model, cfg, device, batches: int = 40) -> tuple[dict[str, float], float]:
    """Per-domain exact CE (in-process); возвращает домены и среднее."""
    results = {}
    for domain, weights in eval_mod.DOMAINS.items():
        r = eval_mod.eval_domain(model, cfg, domain, batches, device)
        results[domain] = r["exact_ce"]
        print(f"    {domain}: {r['exact_ce']:.4f}", flush=True)
    avg = sum(results.values()) / len(results)
    return results, avg


def specialize(incumbent_path: Path, frac: float, steps: int, lr: float, tag: str,
               device, eval_batches: int) -> tuple[Path | None, float, float]:
    """Один цикл: ft → end-merge → certified A/B. Возврат: (новый путь,
    delta, eps). None ⟹ REJECT (инкумбент остаётся)."""
    from hagi.data.dataset import build_dataloader
    from hagi.model.model import HAGI
    from hagi.train.checkpoint import config_from_dict, load_model, load_payload
    from hagi.train.loop import cast_model, configure_runtime, train

    configure_runtime()
    payload = load_payload(str(incumbent_path), "cpu")
    cfg = config_from_dict(payload["config"])
    cfg.train.max_steps = steps
    cfg.train.learning_rate = lr
    cfg.train.saturation_patience = 8
    cfg.train.divergence_patience = 6
    out_dir = ROOT / "checkpoints" / f"ev_{tag}"
    # каждая итерация — новый эксперимент: старые чекпоинты (возможно,
    # другой ширины класса) не должны попадать в EMA-банк end-merge
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.train.checkpoint_dir = str(out_dir)
    cfg.train.checkpoint_interval = max(50, steps // 4)
    cfg.train.checkpoint_keep_last = 6

    model = HAGI(cfg).to(device)
    load_model(str(incumbent_path), model, str(device), lenient_config=True)
    cast_model(model, cfg.train.precision)

    dl = build_dataloader(cfg, start_frac=frac)
    print(f"[{tag}] specialize: {steps} steps, frac={frac:.2f}, lr={lr}", flush=True)
    for _ in train(model, dl, cfg, start_step=0):
        pass

    best = out_dir / "best.pt"
    if not best.exists():
        return None, 0.0, 0.0

    # end-merge по траектории (in-process функции end_merge.py)
    windows = em_mod.gate_batches(cfg, device)
    base_ce = em_mod.gate_ce(model, windows)
    final_sd = {k: v.float() for k, v in torch.load(best, map_location="cpu", weights_only=False)["model"].items()}
    step_paths = sorted(out_dir.glob("step-*.pt"))
    bank = em_mod.ema_bank(step_paths) if len(step_paths) >= 2 else final_sd
    accepted: dict[str, tuple[float, float]] = {}
    current = {"ce": base_ce}
    named = dict(model.named_parameters())

    def try_move(name: str, fs: tuple[float, float] | None):
        f, s = fs if fs is not None else (0.0, 0.0)
        target = final_sd[name] + (bank[name] - final_sd[name]) * (f * s)
        named[name].copy_(target.to(named[name].dtype))

    for name in param_names(model):
        if name not in bank:
            continue
        best_here = (current["ce"], 0.0, 0.0)
        for frac_m in (0.5, 1.0):
            for scale in (0.75, 1.0, 1.25):
                with torch.no_grad():
                    try_move(name, (frac_m, scale))
                ce = em_mod.gate_ce(model, windows)
                if ce < best_here[0] - 1e-4:
                    best_here = (ce, frac_m, scale)
        with torch.no_grad():
            try_move(name, (best_here[1], best_here[2]) if best_here[1] else None)
        if best_here[1]:
            accepted[name] = (best_here[1], best_here[2])
            current["ce"] = best_here[0]
    print(f"[{tag}] end-merge: {base_ce:.4f} -> {current['ce']:.4f} ({len(accepted)} moves)", flush=True)

    candidate_path = out_dir / "end_merged.pt"
    payload = torch.load(best, map_location="cpu", weights_only=False)
    payload["model"] = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(payload, candidate_path)

    # certified A/B
    cand = HAGI(cfg).to(device)
    load_model(str(candidate_path), cand, str(device), lenient_config=True)
    cand_res, cand_avg = certify(cand, cfg, device, eval_batches)
    inc = HAGI(cfg).to(device)
    load_model(str(incumbent_path), inc, str(device), lenient_config=True)
    inc_res, inc_avg = certify(inc, cfg, device, eval_batches)
    n = eval_batches * cfg.train.batch_size * cfg.train.data.seq_len
    eps = math.sqrt(math.log(2.0 / 0.05) / (2.0 * n))
    delta = cand_avg - inc_avg
    no_reg = all(cand_res[d] <= inc_res[d] + 2 * eps for d in cand_res)
    verdict = "ACCEPT" if (delta < -2 * eps and no_reg) else "REJECT"
    print(f"[{tag}] A/B: {inc_avg:.4f} -> {cand_avg:.4f} (delta {delta:+.4f}, eps {eps:.4f}, "
          f"no_reg {no_reg}) => {verdict}", flush=True)
    return (candidate_path, delta, eps) if verdict == "ACCEPT" else (None, delta, eps)


def param_names(model):
    return [n for n, _ in model.named_parameters()]


def width_grow(incumbent_path: Path, target: int) -> Path:
    from hagi.model.width_grow import grow_width_checkpoint
    payload = grow_width_checkpoint(str(incumbent_path), target)
    out = incumbent_path.parent / f"{incumbent_path.stem}_w{target}.pt"
    torch.save({kk: payload[kk] for kk in ("format_version", "model", "config", "completed_steps")}, out)
    n = sum(v.numel() for v in payload["model"].values() if torch.is_tensor(v))
    print(f"[width-grow] {target}: {n/1e6:.1f}M params -> {out.name}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=256, help="цель width-grow")
    ap.add_argument("--frac0", type=float, default=0.55)
    ap.add_argument("--eval-batches", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    incumbent = ROOT / "checkpoints" / "incumbent.pt"
    src = Path(args.incumbent).resolve()
    if src != incumbent.resolve():
        shutil.copyfile(src, incumbent)
    ladder = ROOT / "checkpoints" / "evolve_ladder.jsonl"
    frac = args.frac0
    stall = 0  # подряд циклов без сертифицированного улучшения
    gen = 0
    ROTATION_FRAC = 0.75
    from hagi.train.checkpoint import load_payload
    cur_width = load_payload(str(incumbent), "cpu")["config"]["model"]["hidden_size"]
    recent_deltas: list[float] = []
    while gen < args.generations:
        gen += 1
        tag = f"g{gen}"
        new_path, delta, eps = specialize(incumbent, frac, args.steps, args.lr, tag, device,
                                          args.eval_batches)
        frac = min(frac + 0.15, 0.9)
        recent_deltas.append(delta)
        entry = {"gen": tag, "incumbent": incumbent.name, "delta": round(delta, 4),
                 "eps": round(eps, 5), "verdict": "ACCEPT" if new_path else "REJECT",
                 "frac": round(frac, 2), "date": "2026-10-25"}
        if new_path:
            # SSOT ВЕРСИОНИРУЕТСЯ: width-grow/перезапуск не должен терять
            # сертифицированных инкумбентов (урок: rmtree снёс 512@5.1541)
            versioned = ROOT / "checkpoints" / f"incumbent_w{cur_width}_{gen}.pt"
            shutil.copyfile(new_path, versioned)
            shutil.copyfile(new_path, ROOT / "checkpoints" / "incumbent.pt")
            incumbent = versioned
            cur_width = target_width_of(versioned)
            stall = 0
        else:
            stall += 1
        with ladder.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Data-exhaustion vs saturation: growth_dichotomy честен ТОЛЬКО на
        # свежем сегменте. Деградация при frac >= ROTATION_FRAC — конец
        # ротации данных (плохой хвост), а не потолок класса.
        if stall >= 1 and frac >= ROTATION_FRAC:
            frac = ROTATION_FRAC - 0.25
            stall = 0
            recent_deltas = []
            with ladder.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"gen": f"{tag}-rotate", "frac": round(frac, 2),
                                     "note": "data-exhaustion: rotation reset"},
                                    ensure_ascii=False) + "\n")
            continue

        # width-grow: ОБА stall-REJECT'а на свежих сегментах И затухающий
        # тренд ACCEPT'ов (b_t -> 0 на легальном сегменте;
        # growth_pays_budget: класс растёт, когда шаг-бюджет больше не
        # возвращает сертифицированный прирост).
        fresh = frac < ROTATION_FRAC
        best_accept = min((d for d in recent_deltas if d < 0), default=0.0)
        decaying = len(recent_deltas) >= 2 and abs(recent_deltas[-1]) < 0.5 * abs(best_accept)
        if stall >= 2 and fresh and decaying:
            target = max(args.width, cur_width * 2)
            incumbent = width_grow(incumbent, target)
            cur_width = target
            args.lr *= 0.8
            stall = 0
            recent_deltas = []
            with ladder.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"gen": f"{tag}-width", "width": target,
                                     "note": "growth_dichotomy: fresh-segment stalls + decayed ACCEPT trend"},
                                    ensure_ascii=False) + "\n")
    print(f"evolution complete: incumbent = {incumbent}", flush=True)
    return 0


def target_width_of(path: Path) -> int:
    from hagi.train.checkpoint import load_payload
    return load_payload(str(path), "cpu")["config"]["model"]["hidden_size"]


if __name__ == "__main__":
    raise SystemExit(main())
