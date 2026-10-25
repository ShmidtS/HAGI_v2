"""HAGI v3 fast-cycle конвейер: максимальная скорость при том же качестве.

Редизайн по полной перечитке теории (2026-10-24) + профилирование:
  * ce_keep_rate 0.5 — пунктированный CE (несмещённый, Bernoulli-маска;
    head = 29% шага в bf16, измерено [32768,32768] 481ms -> 242ms при
    половине строк) -> ~-14% шага;
  * КОРОТКИЕ циклы (WallClockTakeoff: малые циклы доминируют, slack
    (k^2/2)·Sum tau^2) — best-значения листьев стабильно на шагах
    150-400 (лестница gen4/gen6), бюджет срезан: math 700, прочие 500,
    books 1200, ft 400;
  * ПАРАЛЛЕЛЬНЫЕ листья — 6 независимых процессов, 3 concurrently
    (GPU-память), лист-фаза wall-clock /2.5;
  * СВЕЖИЕ данные каждого поколения — start-frac = 0.3 + 0.15*(gen-6)
    (seed_only_grow_stop: без инъекции свежести дисперсия затухает);
  * ft с KD от инкумбента (insight_kl_descent: сертифицированный
    KL-спуск; teacher = предыдущий инкумбент, alpha 0.25, T=1) +
    end-merge (noise-порт, стабильно -0.02..-0.15);
  * noise_probe_interval 200 — телеметрия Y для вывода капа батча
    (b* = 9Y^2/(4(eps-Z)^2), grow_arithmetic).

Запуск: python scripts/growth/v3_cycle.py --gen 7 [--prior path]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
DOMS = ["math", "code", "ru", "enedu", "enwiki", "books"]
# Короткие бюджеты v3: best стабильно до этих шагов (измерено gen4/gen6)
STEPS = {"math": 700, "code": 500, "ru": 500, "enedu": 500, "enwiki": 500, "books": 1200}
FT_STEPS = 400


def gen_leaf_configs(gen: int, base_gen: int = 4) -> None:
    """v3-конфиги листьев: база rnng{base_gen}, короткий бюджет, keep 0.5."""
    for d in DOMS:
        src = ROOT / f"configs/rnng{base_gen}_leaf_{d}.yaml"
        s = src.read_text(encoding="utf-8")
        s = re.sub(r"init_seed: (\d+)", lambda m: f"init_seed: {int(m.group(1)) + 200 * (gen - base_gen)}", s)
        s = re.sub(r"max_steps: \d+", f"max_steps: {STEPS[d]}", s)
        s = s.replace("ce_keep_rate: 1.0", "ce_keep_rate: 0.5")
        if "ce_keep_rate" not in s:
            s = s.replace("  compile_model: false", "  compile_model: false\n  ce_keep_rate: 0.5\n  noise_probe_interval: 200")
        else:
            s = s.replace("ce_keep_rate: 0.5", "ce_keep_rate: 0.5\n  noise_probe_interval: 200")
        s = s.replace(f"checkpoints/rnn{base_gen}g_leaf_", f"checkpoints/g{gen}_leaf_")
        out = ROOT / f"configs/g{gen}_leaf_{d}.yaml"
        out.write_text(s, encoding="utf-8")
        print(f"wrote {out.name}: steps={STEPS[d]} keep=0.5")


def gen_ft_config(gen: int, prior_incumbent: str, base_gen: int = 4) -> None:
    """v3 ft: короткий, KD от инкумбента, банк для end-merge."""
    src = ROOT / f"configs/rnn_gen{base_gen}_root_ft.yaml"
    s = src.read_text(encoding="utf-8")
    s = re.sub(r"init_seed: (\d+)", lambda m: f"init_seed: {int(m.group(1)) + 200 * (gen - base_gen)}", s)
    s = re.sub(r"max_steps: \d+", f"max_steps: {FT_STEPS}", s)
    s = s.replace(f"checkpoints/rnn_gen{base_gen}_merged/root.pt", f"checkpoints/g{gen}_merged/root.pt")
    s = s.replace(f"checkpoint_dir: checkpoints/rnn_gen{base_gen}_root_ft", f"checkpoint_dir: checkpoints/g{gen}_root_ft")
    # KD от инкумбента (reverse-recursion канал, cancellation-fixed
    # soft_target_kl) — заменяем СУЩЕСТВУЮЩЕЕ distill: false в merge:
    # (дубликат ключа в YAML молча берёт последнее значение)
    s = s.replace(
        "distill: false",
        f"distill: true\n  distill_teacher: {prior_incumbent}\n"
        "  distill_alpha: 0.25\n  distill_temperature: 1.0",
    )
    out = ROOT / f"configs/g{gen}_root_ft.yaml"
    out.write_text(s, encoding="utf-8")
    print(f"wrote {out.name}: ft={FT_STEPS}, KD alpha=0.25")


def run_leaf(args: tuple[str, str, float, str]) -> tuple[str, int]:
    d, prior, frac, gen = args
    log = open(ROOT / f"logs/g{gen}_leaf_{d}.err", "a")
    r = subprocess.run(
        [PY, "scripts/train.py", "--config", f"configs/g{gen}_leaf_{d}.yaml",
         "--init-from", prior, "--start-frac", str(frac)],
        stdout=log, stderr=log, cwd=ROOT,
    )
    return d, r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--prior", required=True, help="инкумбент-приор для листьев")
    ap.add_argument("--incumbent", required=True, help="текущий инкумбент (KD-учитель ft)")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--base-frac", type=float, default=0.3)
    args = ap.parse_args()
    gen = args.gen
    frac = args.base_frac + 0.15 * max(0, gen - 7)

    gen_leaf_configs(gen)
    gen_ft_config(gen, args.incumbent)

    jobs = [(d, args.prior, frac, str(gen)) for d in DOMS]
    print(f"launching {len(jobs)} leaves, concurrency={args.concurrency}, frac={frac:.2f}", flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for d, rc in pool.map(run_leaf, jobs):
            print(f"=== leaf {d} exit={rc}", flush=True)

    leaves = [f"checkpoints/g{gen}_leaf_{d}/best.pt" for d in DOMS]
    with open(ROOT / f"logs/g{gen}_merge.log", "a") as fh:
        rc = subprocess.run([PY, "scripts/growth/pilot_latent_merge.py", "--leaves", *leaves,
                             "--shared-init", args.prior,
                             "--out", f"checkpoints/g{gen}_merged"], stdout=fh, stderr=fh, cwd=ROOT).returncode
    if rc != 0:
        print("MERGE FAILED", flush=True)
        return 1

    with open(ROOT / f"logs/g{gen}_ft.log", "a") as fh:
        rc = subprocess.run([PY, "scripts/train.py", "--config", f"configs/g{gen}_root_ft.yaml"],
                            stdout=fh, stderr=fh, cwd=ROOT).returncode
    if rc != 0:
        print("FT FAILED", flush=True)
        return 1

    with open(ROOT / f"logs/g{gen}_endmerge.log", "a") as fh:
        subprocess.run([PY, "scripts/growth/end_merge.py", "--config", f"configs/g{gen}_root_ft.yaml",
                        "--checkpoint-dir", f"checkpoints/g{gen}_root_ft"], stdout=fh, stderr=fh, cwd=ROOT)
    print("v3 cycle complete: merge+ft+end-merge done; certify next", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
