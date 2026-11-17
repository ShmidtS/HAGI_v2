#!/usr/bin/env python3
"""HAGI self-rewrite: the model writes its own programs and rewrites its
own state, then continues growing from what it wrote.

Three capabilities, all under the certified-gate discipline:

  1. PROGRAM SYNTHESIS -> PYTHON  `transpile()` renders every expert's
     exact ExprV9 program as a standalone Python function (no runtime
     deps). `export_programs()` writes them to generated_programs.py and
     validates each emitted function against its expert over the
     expert's full certified support (exact integer equality).

  2. SELF-REWRITE  after each run the model rewrites its own persistent
     state: self_state.json (round counter, floor-lever ladder, the
     accumulated proposal stream, expert/waste stats). This is the model
     editing the artifact that governs its own future learning.

  3. RESUME = AUTONOMY  `--resume` boots from exactly what the model
     last wrote about itself (no external config) and keeps growing:
     the recorded stream is replayed, new rounds append new segments,
     new experts, new programs, and the state is rewritten again.

CLI:
  python -m hagi.rewrite --rounds N [--state-dir d]
  python -m hagi.rewrite --resume --rounds N [--state-dir d]
"""
from __future__ import annotations

import os
import re

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import hagi
from hagi.selfmodel import TracedProcessorV9
from hagi.loop import run_self_development

VAR_NAMES = ("x", "y", "z", "u", "v", "w")


# ------------------------------------------------------------ transpiler ----

_BIN = {"add": "+", "sub": "-", "mul": "*", "floordiv": "//",
        "mod": "%", "bit_and": "&", "bit_or": "|", "bit_xor": "^"}
_CMP = {"lt": "<", "le": "<=", "eq": "==", "ge": ">=", "ne": "!=", "gt": ">"}
_SIMPLE1 = {"neg": "(-({0}))", "abs": "abs({0})", "sign": "_hagi_sign({0})",
            "bit_not8": "((~{0}) & 0xFF)", "square": "({0} * {0})",
            "cube": "({0} * {0} * {0})"}
_SIMPLE2 = {"absdiff": "abs({0} - {1})", "gcd": "_hagi_gcd({0}, {1})",
            "lcm": "_hagi_lcm({0}, {1})", "max": "max({0}, {1})",
            "min": "min({0}, {1})",
            "bit_nand8": "((~({0} & {1})) & 0xFF)",
            "bit_nor8": "((~({0} | {1})) & 0xFF)",
            "bit_xnor8": "((~({0} ^ {1})) & 0xFF)",
            "pow_int": "_hagi_pow({0}, {1})"}


def _fn_hash(name: str) -> str:
    """Stable short hash name for an exported function (deterministic
    on the name — same expert always maps to the same hash)."""
    import hashlib
    return "f" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]


def transpile(e, names: tuple[str, ...] = VAR_NAMES) -> str:
    """ExprV9 -> standalone Python expression (helpers _hagi_* are emitted
    at the top of the generated module)."""
    op = e.op
    if op == "var":
        i = int(e.value)
        return names[i] if i < len(names) else f"x{i}"
    if op == "const":
        return repr(int(e.value))
    if op == "call":
        callee = _fn_hash(e.value) if re.fullmatch(
            r"[A-Z][A-Za-z0-9_]*", str(e.value)) else str(e.value)
        return f"{callee}({', '.join(transpile(a, names) for a in e.args)})"
    if op == "anf":
        n_vars, coeffs = e.value
        if not coeffs:
            return "0"
        cs = ", ".join(f"({int(s)}, {int(c)})" for s, c in coeffs)
        xs = ", ".join(names[:n_vars])
        return f"_hagi_anf(({cs},), ({xs},))"
    if op == "if_cond":
        return (f"({transpile(e.args[1], names)} if "
                f"{transpile(e.args[0], names)} else "
                f"{transpile(e.args[2], names)})")
    if op == "poly":
        coeffs = tuple(e.value or ())
        if not coeffs:
            return "0"
        cs = ", ".join(repr(int(c)) for c in coeffs)
        return f"_hagi_poly(({cs},), {transpile(e.args[0], names)})"
    if op == "poly_nd":
        terms = tuple(e.value or ())
        if not terms:
            return "0"
        ts = ", ".join(f"({int(c)}, ({', '.join(str(int(p)) for p in pw)}))"
                        for c, pw in terms)
        ns = ", ".join(names)
        return f"_hagi_poly_nd(({ts},), ({ns},))"
    a = [transpile(x, names) for x in e.args]
    if op in _BIN:
        return f"({a[0]} {_BIN[op]} {a[1]})"
    if op in _CMP:
        return f"(1 if {a[0]} {_CMP[op]} {a[1]} else 0)"
    if op in _SIMPLE1:
        return _SIMPLE1[op].format(a[0])
    if op in _SIMPLE2:
        return _SIMPLE2[op].format(a[0], a[1])
    if op == "fma":
        return f"({a[0]} * {a[1]} + {a[2]})"
    if op == "clamp":
        return f"min(max({a[0]}, {a[1]}), {a[2]})"
    if op == "clamp3":
        return f"min(max({a[0]}, -3), 3)"
    if op == "median3":
        return f"sorted(({a[0]}, {a[1]}, {a[2]}))[1]"
    if op == "mul3":
        return f"({a[0]} * {a[1]} * {a[2]})"
    raise ValueError(f"unsupported op for transpile: {op}")


RUNTIME_HELPERS = '''\
def _hagi_pow(x, y):
    if y < 0:
        raise ValueError("negative exponent")
    return x ** y


def _hagi_sign(v):
    return (v > 0) - (v < 0)


def _hagi_gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def _hagi_lcm(a, b):
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // _hagi_gcd(a, b)


def _hagi_anf(coeffs, xs):
    mask = 0
    for i, b in enumerate(xs):
        if b:
            mask |= 1 << (len(xs) - 1 - i)
    acc = 0
    for s, c in coeffs:
        if (s & mask) == s:
            acc ^= c
    return acc


def _hagi_poly(coeffs, x):
    acc = 0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _hagi_poly_nd(terms, xs):
    total = 0
    for coef, powers in terms:
        t = coef
        for x, p in zip(xs, powers):
            t *= x ** p
        total += t
    return total
'''


def _expert_domain(ex) -> list[tuple[int, ...]]:
    import itertools
    doms = [tuple(int(v) for v in d) for d in ex.domains]
    rows = []
    for row in itertools.product(*doms):
        if ex.valid_rule == "nonzero_y" and row[1] == 0:
            continue
        if ex.valid_rule == "ordered_yz" and row[1] > row[2]:
            continue
        rows.append(row)
    return rows


def export_programs(proc, out_dir: Path) -> dict[str, Any]:
    """Write every expert as a standalone Python function and validate
    exact equality against the expert over its full certified domain."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "generated_programs.py"
    lines = [
        '"""Programs written by HAGI (exact synthesis, gate-adopted).',
        '',
        'Auto-generated by hagi_selfrewrite.export_programs; every',
        "function is validated for exact integer equality against its",
        "certified expert over the expert's full support domain.",
        '"""',
        "", RUNTIME_HELPERS,
    ]
    for name, ex in proc.experts.items():
        expr = ex.program.expr
        if expr is None:
            continue
        params = ", ".join(VAR_NAMES[:ex.program.arity])
        lines.append(f"def {_fn_hash(name)}({params}):")
        lines.append(f"    return {transpile(expr, VAR_NAMES[:ex.program.arity])}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

    import importlib.util
    spec = importlib.util.spec_from_file_location("hagi_generated", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    checked = failed = 0
    for name, ex in proc.experts.items():
        expr = ex.program.expr
        fn = getattr(mod, _fn_hash(name), None)
        if expr is None or fn is None:
            if expr is not None:
                failed += 1
            continue
        for row in _expert_domain(ex):
            try:
                want = expr.eval(row, proc.experts)
            except Exception:
                break  # outside the expert's effective support
            if int(fn(*row)) != int(want):
                failed += 1
                break
            checked += 1
    return {"path": str(path), "functions": sum(
                1 for ex in proc.experts.values() if ex.program.expr),
            "evaluations_checked": checked, "failures": failed}


# ------------------------------------------------------------ self-rewrite --

def run(rounds: int, state_dir: Path, workers: int = 0, resume: bool = False,
        floor_ladder: list[int] | None = None) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "self_state.json"
    inc_path = state_dir / "incumbent.pkl"

    initial_segments: list[tuple[int, int]] = []
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        initial_segments = [tuple(s) for s in state["segments"]]
        ladder = state.get("floor_ladder") or floor_ladder or [200, 50, 20]
        booted_from = "self_state.json (the model's own rewrite)"
        prior = state
    else:
        ladder = floor_ladder or [200, 50, 20]
        booted_from = "cold start"
        prior = None

    # workers: threads share memory — no pickling MemoryError, no
    # degradation needed. Default = all CPU cores (user decision
    # 2026-11-17: maximum possible for the machine).
    eff_workers = workers or (os.cpu_count() or 4)
    # CertifiedLoop: gate-adopted knobs drive the production loop.
    try:
        from hagi import selfedit
        adopted_kwargs = selfedit.adopted_settings()
    except Exception:
        adopted_kwargs = {}
    if "floor_ladder" not in adopted_kwargs:
        adopted_kwargs["floor_ladder"] = ladder
    total = run_self_development(
        max_rounds=rounds, workers=eff_workers,
        initial_segments=initial_segments, **adopted_kwargs)
    final_proc = total.pop("_final_incumbent")

    report: dict[str, Any] = {
        "booted_from": booted_from,
        "resumed_segments": len(initial_segments),
        "prior_state": prior,
        "adopted_rounds": total["final"]["adopted_rounds"],
        "experts": total["final"]["experts"],
        "wasted": total["final"]["wasted_candidates_on_holds"],
        "seconds": total["final"]["seconds_total"],
    }

    # the rewrite: model persists itself + its programs
    inc_path.write_bytes(pickle.dumps(final_proc))
    # self-optimization (MDL): prune duplicates/pseudo-args, keep the
    # call DAG intact, re-validate everything exactly; full library
    # is kept as incumbent_full.pkl
    try:
        import mdl as _opt
        _res = _opt.prune_library(final_proc)
        _new, _fails = _opt.apply_prune(final_proc, _res["keep"])
        if _fails == 0 and len(_new.experts) < len(final_proc.experts):
            inc_full = state_dir / "incumbent_full.pkl"
            inc_full.write_bytes(pickle.dumps(final_proc))
            final_proc = _new
            inc_path.write_bytes(pickle.dumps(final_proc))
        report["self_optimize"] = _res["report"]
    except Exception as e:  # optimization must never break the loop
        report["self_optimize"] = {"skipped": repr(e)[:80]}
    report["programs"] = export_programs(final_proc, state_dir)
    state = {
        "rounds_done": (prior or {}).get("rounds_done", 0)
                       + total["final"]["adopted_rounds"],
        "segments": total["segments"],
        "floor_ladder": ladder,
        "rewrites": (prior or {}).get("rewrites", 0) + 1,
        "experts": len(final_proc.experts),
        "wasted": total["final"]["wasted_candidates_on_holds"],
    }
    # curriculum phase accounting (primes CurriculumCert, order-free)
    import curriculum as _curr
    adopted = total["final"]["adopted_rounds"] or 1
    relief = adopted / (adopted + total["final"]["wasted_candidates_on_holds"])
    prior_curr = (prior or {}).get("curriculum", [])
    prior_curr.append({"phase": "GROW->DERIVE->OPTIMIZE->GATE",
                       "delta": round(relief, 4),
                       "d_cert": 0.02, "steps": adopted})
    state["curriculum"] = prior_curr
    _curr.certificate(state)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    report["self_state"] = {k: (v if k != "segments" else len(v))
                            for k, v in state.items()}
    return report


def _self_edit_guarded(cycle: int) -> None:
    """Each cycle: Nadya proposes a patch to her own learning knobs;
    certified gate (benchmark no-regression) decides adoption."""
    try:
        from hagi import selfedit
        res = selfedit.self_edit_cycle(seed=cycle)
        print(f"[self-edit] {res.get('decision')}: {res.get('patch')}",
              flush=True)
    except Exception as e:  # self-edit failure never stops the loop
        print(f"[self-edit] skipped: {e!r}", flush=True)


def _archgenesis_guarded(cycle: int) -> None:
    """Architecture Genesis: propose -> benchmark -> certified adopt
    of a loop-config change (relative to the throughput profile).
    Runs rarely (once per 20 cycles) — each probe costs a full
    micro-benchmark (FlopsEconomy: probes are not free)."""
    if cycle % 20 != 7:  # offset so it never collides with self-edit
        return
    try:
        from hagi import genesis
        res = genesis.genesis_cycle(seed=cycle,
                                             profile="throughput")
        print(f"[archgenesis] {res.get('decision')}: "
              f"{res.get('mutation')} gain={res.get('rel_gain')}",
              flush=True)
    except Exception as e:  # genesis failure never stops the loop
        print(f"[archgenesis] skipped: {e!r}", flush=True)


def _ec_evolution_guarded(cycle: int) -> None:
    """Evolutionary EC-processor genesis: every 30th cycle evolve the
    EC-processor genome one generation batch (certified-first fitness:
    group-law gate, then modmul cost). Profile rotates by cycle parity:
    even -> throughput, odd -> security (relative optimum, no universal
    config)."""
    if cycle % 30 != 13:
        return
    try:
        from hagi import genesis as ec
        profile = "throughput" if cycle % 60 < 30 else "security"
        res = ec.evolve(pop_size=8, generations=3, profile=profile,
                        seed=cycle)
        print(f"[ec-evolution] {profile}: best {res['best_fitness']} "
              f"modmul/mul, genome={res['best_genome']}", flush=True)
    except Exception as e:  # evolution failure never stops the loop
        print(f"[ec-evolution] skipped: {e!r}", flush=True)


def _retrain_lm_guarded(state_dir: Path) -> None:
    """Each forever cycle: the byte LM retrains on the model's own
    artifacts (self-referential neural growth, §3/§9 guarded)."""
    try:
        from hagi import mind
        print("[lm]", mind.retrain(), flush=True)
    except Exception as e:  # neural body failure never stops the loop
        print(f"[lm] retrain skipped: {e!r}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="HAGI self-rewrite loop")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--forever", action="store_true",
                   help="infinite autonomous self-development: grow-"
                        "rewrite-resume cycles until the process is "
                        "stopped; every cycle rewrites self_state.json "
                        "and generated_programs.py")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--state-dir", type=Path, default=Path("checkpoints/self"))
    a = p.parse_args()
    if a.forever:
        import time as _time
        cycle = 0
        while True:
            cycle += 1
            rep = run(a.rounds, a.state_dir, workers=a.workers, resume=True)
            print(f"[forever cycle {cycle}] rounds={rep['self_state']['rounds_done']} "
                  f"experts={rep['experts']} funcs={rep['programs']['functions']} "
                  f"checked={rep['programs']['evaluations_checked']} "
                  f"failures={rep['programs']['failures']} "
                  f"rewrites={rep['self_state']['rewrites']} "
                  f"{rep['seconds']}s", flush=True)
            _retrain_lm_guarded(a.state_dir)
            _self_edit_guarded(cycle)
            _archgenesis_guarded(cycle)
            _ec_evolution_guarded(cycle)
            _time.sleep(1)
    rep = run(a.rounds, a.state_dir, workers=a.workers, resume=a.resume)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


def rename_programs_hash(path: Path) -> dict[str, str]:
    """Rewrite an existing generated_programs.py: every top-level
    `def NAME(...)` and all its call sites get a short stable hash
    name f<sha1:10>. Helper functions (_hagi_*) keep their names.
    Returns the old->new mapping."""
    src = path.read_text(encoding="utf-8")
    defs = re.findall(r"^def ([A-Z][A-Za-z0-9_]*)\(", src, flags=re.M)
    mapping = {old: _fn_hash(old) for old in defs}
    def sub(m):
        n = m.group(1)
        return ("def " + mapping[n] + "(") if n in mapping else m.group(0)
    src2 = re.sub(r"^def ([A-Za-z0-9_]+)\(", sub, src, flags=re.M)
    for old, new in mapping.items():
        src2 = re.sub(rf"{re.escape(old)}", new, src2)
    path.write_text(src2, encoding="utf-8")
    return mapping
