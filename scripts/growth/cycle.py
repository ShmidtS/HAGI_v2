"""Единый параметризованный цикл роста HAGI (замена всех одноразовых chain-скриптов).

Цикл поколения N:
  листья (параллельно, fresh-data frac) -> latent merge -> root-ft
  (KD от инкумбента) -> end-merge; certified A/B отдельным certify.py.

Рост класса (по теории):
  --domains      множество доменов поколения (M↑ = новые волокна, рост
                 параметрической массы; fiber_params_floor, growth_dichotomy)
  --leaf-width   ширина листа (новый класс: сброс V0, рост тела; при смене
                 ширины prior не переносится — листья свежего init, SeedOnly:
                 разнообразие дают домены)

    python scripts/growth/cycle.py --gen 8 --prior <ckpt> --incumbent <ckpt> \
        --domains math code ru enedu enwiki books smoltalk tinystories wikien \
        --new-domains smoltalk tinystories wikien --frac 0.5
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

# Домен -> (mix-веса, шаги). Шаги: prior-специализированные коротко (best
# на 150-400), СВЕЖИЕ домены дольше (специализация с нуля ~800).
DOMAIN_MIX: dict[str, dict[str, float]] = {
    "math": {"openwebmath": 0.6, "edu": 0.25, "slimpajama": 0.15},
    "code": {"python_instruct": 0.6, "edu": 0.25, "slimpajama": 0.15},
    "ru": {"oscar_ru": 0.5, "wikipedia_ru": 0.5},
    "enedu": {"edu": 0.6, "slimpajama": 0.4},
    "enwiki": {"wikipedia_en": 0.6, "edu": 0.4},
    "books": {"slimpajama": 0.7, "edu": 0.3},
    "smoltalk": {"smoltalk": 0.6, "edu": 0.25, "slimpajama": 0.15},
    "tinystories": {"tinystories": 0.6, "edu": 0.25, "slimpajama": 0.15},
    "wikien": {"wikipedia_en": 0.7, "edu": 0.3},
}
DEFAULT_STEPS = {"math": 700, "books": 1200, "smoltalk": 800, "tinystories": 800, "wikien": 800}
PRIOR_STEPS = 500
FT_STEPS = 400
BASE_CONFIG = ROOT / "configs/rnng4_leaf_math.yaml"  # шаблон геометрии/обучения


def leaf_config(gen: int, domain: str, width: int, steps: int, seed_offset: int) -> Path:
    s = BASE_CONFIG.read_text(encoding="utf-8")
    s = re.sub(r"init_seed: (\d+)", lambda m: f"init_seed: {int(m.group(1)) + seed_offset}", s)
    s = re.sub(r"max_steps: \d+", f"max_steps: {steps}", s)
    s = re.sub(r"hidden_size: \d+", f"hidden_size: {width}", s)
    # геометрия голов следует ширине (validate: q_heads*head_dim == width)
    q_heads = max(1, width // 64)
    s = re.sub(r"num_query_heads: \d+", f"num_query_heads: {q_heads}", s)
    s = re.sub(r"num_kv_heads: \d+", f"num_kv_heads: {max(1, q_heads // 2)}", s)
    s = re.sub(r"checkpoint_dir: .*", f"checkpoint_dir: checkpoints/c{gen}_{domain}", s)
    # Замена весов mix построчно (устойчиво к CRLF): блок после "weights:"
    # до первой строки того же уровня отступа без слова-источника.
    lines = s.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        out.append(lines[i])
        if lines[i].strip() == "weights:":
            block = "".join(f"      {k}: {v}\n" for k, v in DOMAIN_MIX[domain].items())
            out[-1] = lines[i].rstrip("\r")[: len(lines[i]) - len(lines[i].lstrip())] + "weights:"
            out.append(block.rstrip("\n"))
            i += 1
            while i < len(lines) and lines[i].startswith("      ") and lines[i].strip():
                i += 1
            continue
        i += 1
    s = "\n".join(out) + "\n"
    if "ce_keep_rate" not in s:
        s = s.replace("  compile_model: false", "  compile_model: false\n  ce_keep_rate: 0.5\n  noise_probe_interval: 200")
    out_path = ROOT / f"configs/c{gen}_{domain}.yaml"
    out_path.write_text(s, encoding="utf-8")
    return out_path


def ft_config(gen: int, width: int, n_domains: int, incumbent: str, kd: bool) -> Path:
    s = BASE_CONFIG.read_text(encoding="utf-8")
    s = re.sub(r"init_seed: (\d+)", lambda m: f"init_seed: {int(m.group(1)) + 200 * (gen - 4)}", s)
    s = re.sub(r"max_steps: \d+", f"max_steps: {FT_STEPS}", s)
    s = re.sub(r"hidden_size: \d+", f"hidden_size: {width}", s)
    s = re.sub(r"checkpoint_dir: .*", f"checkpoint_dir: checkpoints/c{gen}_root_ft", s)
    # банк для end-merge (EMA): частые чекпоинты, держим 8
    s = s.replace("checkpoint_interval: 400", "checkpoint_interval: 100")
    s = s.replace("checkpoint_keep_last: 3", "checkpoint_keep_last: 8")
    # init_from в шаблоне листа НЕТ — вставляем явно (no-op замена обучала
    # ft с нуля: диагностировано на gen8, gate 8.9 вместо ~6.2)
    s = s.replace(
        "train:\n  max_steps:",
        f"train:\n  init_from: checkpoints/c{gen}_merged/root.pt\n  max_steps:",
    )
    # геометрия merge-секции: norm_block_dim = ширина листа
    s = re.sub(r"norm_block_dim: \d+", f"norm_block_dim: {width}", s)
    if kd:
        s = s.replace(
            "distill: false",
            f"distill: true\n  distill_teacher: {incumbent}\n  distill_alpha: 0.25\n  distill_temperature: 1.0",
        )
    out = ROOT / f"configs/c{gen}_root_ft.yaml"
    out.write_text(s, encoding="utf-8")
    return out


def run_leaf(job: tuple[int, str, str, float, bool]) -> tuple[str, int]:
    gen, d, prior, frac, fresh = job
    log = open(ROOT / f"logs/c{gen}_leaf_{d}.err", "a")
    cmd = [PY, "scripts/train.py", "--config", f"configs/c{gen}_{d}.yaml",
           "--start-frac", str(frac)]
    if not fresh:
        cmd += ["--init-from", prior]
    r = subprocess.run(cmd, stdout=log, stderr=log, cwd=ROOT)
    return d, r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--new-domains", nargs="*", default=[], help="домены без prior-специализации (длинные шаги)")
    ap.add_argument("--frac", type=float, default=0.45)
    ap.add_argument("--leaf-width", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--no-kd", action="store_true")
    ap.add_argument("--skip-leaves", action="store_true", help="листья уже обучены: только merge+ft+em")
    ap.add_argument("--fresh-init", action="store_true",
                    help="сброс класса: без init_from у листьев (новая ширина; SeedOnly-разнообразие дают домены)")
    args = ap.parse_args()
    gen = args.gen
    kd = not args.no_kd and args.leaf_width == 128  # KD требует ту же геометрию

    for d in args.domains:
        steps = DEFAULT_STEPS.get(d, PRIOR_STEPS) if d in args.new_domains else PRIOR_STEPS
        leaf_config(gen, d, args.leaf_width, steps, seed_offset=200 * (gen - 4))
        print(f"config c{gen}_{d}: steps={steps} width={args.leaf_width}")
    ft_config(gen, args.leaf_width, len(args.domains), args.incumbent, kd)

    if not args.skip_leaves:
        jobs = [(gen, d, args.prior, args.frac, args.fresh_init) for d in args.domains]
        print(f"launching {len(jobs)} leaves, concurrency={args.concurrency}, frac={args.frac}", flush=True)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for d, rc in pool.map(run_leaf, jobs):
                print(f"=== leaf {d} exit={rc}", flush=True)

    leaves = [f"checkpoints/c{gen}_{d}/best.pt" for d in args.domains]
    with open(ROOT / f"logs/c{gen}_merge.log", "a") as fh:
        rc = subprocess.run([PY, "scripts/growth/pilot_latent_merge.py", "--leaves", *leaves,
                             "--shared-init", args.prior, "--out", f"checkpoints/c{gen}_merged"],
                            stdout=fh, stderr=fh, cwd=ROOT).returncode
    if rc != 0:
        print("MERGE FAILED", flush=True)
        return 1

    with open(ROOT / f"logs/c{gen}_ft.log", "a") as fh:
        rc = subprocess.run([PY, "scripts/train.py", "--config", f"configs/c{gen}_root_ft.yaml"],
                            stdout=fh, stderr=fh, cwd=ROOT).returncode
    if rc != 0:
        print("FT FAILED", flush=True)
        return 1

    with open(ROOT / f"logs/c{gen}_endmerge.log", "a") as fh:
        subprocess.run([PY, "scripts/growth/end_merge.py", "--config", f"configs/c{gen}_root_ft.yaml",
                        "--checkpoint-dir", f"checkpoints/c{gen}_root_ft"], stdout=fh, stderr=fh, cwd=ROOT)
    print(f"cycle {gen} complete (merge+ft+end-merge); certify: certify.py --tag c{gen}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
