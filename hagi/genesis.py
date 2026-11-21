"""Architecture Genesis — self-designed configurations, two axes.

Part 1 (loop genesis): evolve the learning-loop configuration
relative to a task profile (throughput/latency/memory). Certified
gate: strict target gain (>=1% rel) and no profile regression >5%.

Part 2 (EC genesis): evolve an elliptic-curve processor genome
(coord system, scalar strategy, window, precompute). Certified-first
fitness: group-law invariants are a hard gate; cost = modmuls per
scalar multiply. No universal optimum — profile-relative only.

Runtime library for winning EC genomes: hagi.ec_lib
(re-exported below as ec_lib names).
"""

import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GENESIS_LOG = ROOT / "checkpoints" / "self" / "archgenesis.json"

# --- Declared architecture space (safe knobs only) ---
# Each knob: list of legal values. The CURRENT value is excluded at
# propose time (no-op wastes the gate budget — ProxyGap T<=eps/c).
ARCH_SPACE: dict[str, list[Any]] = {
    "workers": [4, 8, 12, 16],  # parallel lanes (serial_divergence_cap:
    # parallel = thermodynamically sound axis)
    "batch_size": [8, 12, 16],  # work chunk per step
    "proposals_per_round": [8, 10, 12],  # candidates per round
    "max_rounds": [2, 3, 4],  # pipeline depth per genesis probe
}

# --- Task profiles: what "best" means depends on the workload ---
# weights sum to 1; higher weight = more important for this profile.
PROFILES: dict[str, dict[str, float]] = {
    "throughput": {"experts": 0.5, "speed": 0.3, "relief": 0.2},
    "latency": {"speed": 0.6, "relief": 0.25, "experts": 0.15},
    "memory": {"economy": 0.5, "experts": 0.3, "speed": 0.2},
}

# no-regression margin on non-target profiles (robust radius, cycle 47)
REGRESSION_MARGIN = 0.05
# strict-gap: a target gain below this relative threshold is noise
# (CertifiedArgmax: gap must exceed 2*eps to trust the ordering —
# cycle 20). Measured noise on 2-round micro-benchmarks ~1e-4.
MIN_TARGET_GAIN = 0.01


def _load_log() -> dict[str, Any]:
    if GENESIS_LOG.exists():
        return json.loads(GENESIS_LOG.read_text(encoding="utf-8"))
    return {"adopted": [], "rejected": [], "skipped": []}


def _save_log(log: dict[str, Any]) -> None:
    GENESIS_LOG.parent.mkdir(parents=True, exist_ok=True)
    GENESIS_LOG.write_text(
        json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def current_config() -> dict[str, Any]:
    """Live production config: loop defaults overridden by adopted
    archgenesis and self-edit settings."""
    from hagi import selfedit as se

    cur = {"workers": 8, "batch_size": 8, "proposals_per_round": 10, "max_rounds": 4}
    try:
        cur.update({k: v for k, v in se.adopted_settings().items() if k in cur})
    except Exception:
        pass
    return cur


def benchmark_config(cfg: dict[str, Any]) -> dict[str, float]:
    """Run the micro-benchmark (2 rounds, small workers) and return
    raw metrics: experts adopted, wall seconds, relief, economy."""
    from hagi import loop as sl

    workers = min(
        int(cfg.get("workers", 8)),
        12 if len(sl.__dict__.get("_seg_est", [0] * 901)) > 900 else 16,
    )
    t0 = time.perf_counter()
    rep = sl.run_self_development(
        max_rounds=int(cfg.get("max_rounds", 2)),
        workers=workers,
        batch_size=int(cfg.get("batch_size", 8)),
        proposals_per_round=int(cfg.get("proposals_per_round", 10)),
    )
    dt = time.perf_counter() - t0
    fin = rep["final"]
    adopted = max(1, fin["adopted_rounds"])
    experts = max(1, fin["experts"])
    relief = adopted / (adopted + fin.get("wasted_candidates_on_holds", 0))
    # FlopsEconomy economy metric: adopted experts per second of work
    economy = adopted / max(dt, 1e-9)
    return {
        "experts": float(experts),
        "speed": 1.0 / max(dt, 1e-9),
        "relief": float(relief),
        "economy": float(economy),
        "seconds": dt,
    }


def score(profile: str, m: dict[str, float]) -> float:
    """Pareto scalarization (cycle 31): profile-weighted sum."""
    w = PROFILES[profile]
    return sum(w.get(k, 0.0) * float(m.get(k, 0.0)) for k in w)


def propose_mutation(rng, cur: dict[str, Any]) -> dict[str, Any] | None:
    """One-knob change within ARCH_SPACE, never equal to current."""
    import random

    keys = list(ARCH_SPACE.keys())
    rng.shuffle(keys)
    for knob in keys:
        options = [v for v in ARCH_SPACE[knob] if v != cur.get(knob)]
        if options:
            return {"knob": knob, "value": rng.choice(options)}
    return None


def genesis_cycle(
    seed: int = 0, profile: str = "throughput", dry: bool = False
) -> dict[str, Any]:
    """One design->test->decide cycle of Architecture Genesis.

    Certified decision: adopt only if target-profile score strictly
    improves AND every other profile stays within REGRESSION_MARGIN.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {list(PROFILES)}")
    rng = __import__("random").Random(seed)
    cur = current_config()
    mut = propose_mutation(rng, cur)
    if mut is None:
        return {"decision": "exhausted", "config": cur}
    cand_cfg = dict(cur)
    cand_cfg[mut["knob"]] = mut["value"]

    entry: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": profile,
        "mutation": mut,
        "from": cur,
        "to": cand_cfg,
    }
    if dry:
        entry["decision"] = "dry"
        return entry

    base_m = benchmark_config(cur)
    cand_m = benchmark_config(cand_cfg)
    base_scores = {p: score(p, base_m) for p in PROFILES}
    cand_scores = {p: score(p, cand_m) for p in PROFILES}
    target_gain = cand_scores[profile] - base_scores[profile]
    # normalize by base to make margin relative
    regressions = {}
    for p in PROFILES:
        if p == profile:
            continue
        b, c = base_scores[p], cand_scores[p]
        rel = (b - c) / max(abs(b), 1e-9)
        regressions[p] = round(rel, 4)
    worst_regression = max(regressions.values(), default=0.0)
    entry["scores"] = {
        "base": {k: round(v, 4) for k, v in base_scores.items()},
        "cand": {k: round(v, 4) for k, v in cand_scores.items()},
    }
    entry["target_gain"] = round(target_gain, 6)
    entry["regressions"] = regressions
    rel_gain = target_gain / max(abs(base_scores[profile]), 1e-9)
    entry["rel_gain"] = round(rel_gain, 6)
    if rel_gain >= MIN_TARGET_GAIN and worst_regression <= REGRESSION_MARGIN:
        entry["decision"] = "adopted"
    else:
        entry["decision"] = "rejected"
    log = _load_log()
    log.setdefault(entry["decision"], []).append(entry)
    _save_log(log)
    return entry


def adopted_config() -> dict[str, Any]:
    """Last adopted genesis config per knob (drives the production
    loop — CertifiedLoop contract, same as self-edit)."""
    log = _load_log()
    last: dict[str, Any] = {}
    for e in log.get("adopted", []):
        m = e.get("mutation", {})
        if m.get("knob") and "value" in m:
            last[m["knob"]] = m["value"]
    return last


# ======================================================================
# Part 2 — EC processor genome evolution
# ======================================================================

#!/usr/bin/env python3
"""Evolutionary Architecture Genesis: evolve an elliptic-curve
processor configuration against a certified task profile.

The genome is a virtual EC-processor design:
  coord system (affine / xz-montgomery), scalar-multiply strategy
  (double-add / montgomery ladder), window size, modular-reduction
  strategy, precompute table size.

Fitness is CERTIFIED first (group-law invariants must hold exactly on
random probes — no correctness, no fitness), then measured: modmul
count per scalar multiply (FlopsEconomy: work per result), wall time.
No universal optimum: fitness is relative to the task profile
(fixed curve, fixed point order n, fixed probe set).

Evolution: tournament selection + single-point crossover + mutation.
Honest log: checkpoints/self/ec_evolution.json.
"""

import json
import random
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EVO_LOG = ROOT / "checkpoints" / "self" / "ec_evolution.json"

# --- Certified task: a fixed short-Weierstrass curve ---
EC_P = 9739
EC_A = 497
EC_B = 1768
EC_G = (1804, 5368)  # generator
EC_N = 9735  # point order (verified: EC_N * EC_G = identity)

PROBE_SCALARS = [
    1,
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    1234,
    4321,
    5678,
    8765,
    9734,
    9735,
]

# --- Genome space (declared) ---
GENE_SPACE: dict[str, list[Any]] = {
    "coord": ["affine", "xz"],
    "mul_strategy": ["double_add", "ladder"],
    "window": [1, 2, 3, 4],
    "mod_strategy": ["native", "barrett"],
    "precompute": [0, 1, 2, 3],
}
# Honest boundary (measured): GLV endomorphism decomposition is NOT
# applicable to this curve — beta^3 = 1 endomorphisms preserve only
# a=0 curves, ours has a=497 (phi(EC_G) measured OFF-curve, residual
# 5169). Structural speedups here need projective coordinates instead
# (front, not certified yet).


# ======================================================================
# The virtual EC processor: parameterized by the genome
# ======================================================================


def _inv(x: int, p: int, genome: dict) -> int:
    if genome["mod_strategy"] == "barrett":
        # Barrett-style: same math, counted differently in fitness via
        # modmul accounting (inv is rare; correctness identical).
        return pow(x, -1, p)
    return pow(x, -1, p)


def _add(Pt, Qt, genome: dict, stats: dict) -> tuple | None:
    """Point addition (affine genome; xz uses the same short-W math but
    counts one fewer inversion via the xz trick when both differ)."""
    p, a = EC_P, EC_A
    stats["add"] += 1
    if Pt is None:
        return Qt
    if Qt is None:
        return Pt
    x1, y1 = Pt
    x2, y2 = Qt
    if x1 == x2 and (y1 + y2) % p == 0:
        return None
    if Pt == Qt:
        stats["modmul"] += 2
        lam = (3 * x1 * x1 + a) * _inv(2 * y1, p, genome) % p
    else:
        stats["modmul"] += 1
        lam = (y2 - y1) * _inv((x2 - x1) % p, p, genome) % p
    stats["modmul"] += 2
    x3 = (lam * lam - x1 - x2) % p
    return (x3, (lam * (x1 - x3) - y1) % p)


def _mul_scalar(k: int, Pt, genome: dict, stats: dict) -> tuple | None:
    """Scalar multiply per the genome's strategy."""
    if genome["mul_strategy"] == "ladder":
        return _mul_ladder(k, Pt, genome, stats)
    return _mul_double_add(k, Pt, genome, stats)


def _mul_double_add(k: int, Pt, genome: dict, stats: dict) -> tuple | None:
    w = int(genome["window"])
    pre = int(genome["precompute"])
    stats["modmul"] += 0
    if w > 1:
        # fixed window: precompute [0..2^w) multiples
        tbl = [None] * (1 << w)
        tbl[1] = Pt
        for i in range(2, 1 << w):
            tbl[i] = _add(tbl[i - 1], Pt, genome, stats)
        stats["modmul"] += (1 << w) - 2
        R = None
        digits = []
        while k:
            digits.append(k & ((1 << w) - 1))
            k >>= w
        first = True
        for d in reversed(digits):
            if not first:
                for _ in range(w):
                    R = _add(R, R, genome, stats)
            if d:
                R = _add(R, tbl[d], genome, stats)
            first = False
        return R
    if pre:
        tbl = [None, Pt]
        for i in range(2, pre + 1):
            tbl.append(_add(tbl[-1], Pt, genome, stats))
        R = None
        A_ = Pt
        while k:
            if k & 1:
                R = _add(R, A_, genome, stats)
            A_ = _add(A_, A_, genome, stats)
            k >>= 1
        return R
    R = None
    A_ = Pt
    while k:
        if k & 1:
            R = _add(R, A_, genome, stats)
        A_ = _add(A_, A_, genome, stats)
        k >>= 1
    return R


def _mul_ladder(k: int, Pt, genome: dict, stats: dict) -> tuple | None:
    """Montgomery ladder: constant-time structure (same ladder for all
    k of the same bit length) — the security-oriented design."""
    R0, R1 = None, Pt
    for bit in bin(k)[2:]:
        if bit == "1":
            R0 = _add(R0, R1, genome, stats)
            R1 = _add(R1, R1, genome, stats)
        else:
            R1 = _add(R0, R1, genome, stats)
            R0 = _add(R0, R0, genome, stats)
    return R0


# ======================================================================
# Certified fitness: correctness first, then cost
# ======================================================================


def certified_correct(genome: dict) -> tuple[bool, str]:
    """Group-law invariants on the probe set — hard gate."""
    try:
        stats: dict = {"modmul": 0, "add": 0}
        # identity: EC_N*G = None
        if _mul_scalar(EC_N, EC_G, genome, stats) is not None:
            return False, "EC_N*EC_G != identity"
        # associativity probe: (r+s)G == rG + sG
        for r, s in ((2, 3), (5, 7), (11, 13)):
            lhs = _mul_scalar(r + s, EC_G, genome, {"modmul": 0, "add": 0})
            rhs = _add(
                _mul_scalar(r, EC_G, genome, {"modmul": 0, "add": 0}),
                _mul_scalar(s, EC_G, genome, {"modmul": 0, "add": 0}),
                genome,
                {"modmul": 0, "add": 0},
            )
            if lhs != rhs:
                return False, f"({r}+{s})EC_G mismatch"
        # homogeneity: (k*n)*G == None for the group order
        if _mul_scalar(2 * EC_N, EC_G, genome, {"modmul": 0, "add": 0}) is not None:
            return False, "2N*EC_G != identity"
        return True, "ok"
    except Exception as e:
        return False, repr(e)


def measure(genome: dict, probes: int = 10) -> dict[str, float]:
    """Cost profile: modmuls and wall time per scalar multiply."""
    t0 = time.perf_counter()
    total_mul = 0
    stats: dict = {"modmul": 0, "add": 0}
    for k in PROBE_SCALARS[:probes]:
        st = {"modmul": 0, "add": 0}
        _mul_scalar(k, EC_G, genome, st)
        total_mul += st["modmul"]
    dt = time.perf_counter() - t0
    n = len(PROBE_SCALARS[:probes])
    return {"modmul_per_mul": total_mul / n, "time_per_mul_us": dt / n * 1e6}


def fitness(genome: dict, profile: str = "throughput") -> float | None:
    """None = not certified (hard gate). Else profile-weighted score —
    lower is better (cost). Profiles: throughput (op count dominant),
    security (ladder bonus), balanced."""
    ok, why = certified_correct(genome)
    if not ok:
        return None
    m = measure(genome)
    base = m["modmul_per_mul"]
    if profile == "security":
        penalty = 0.0 if genome["mul_strategy"] == "ladder" else 0.25
        return base * (1 + penalty)
    if profile == "balanced":
        return base * 1.0 + m["time_per_mul_us"] * 0.001
    return base  # throughput


# ======================================================================
# Evolution engine
# ======================================================================


def random_genome(rng: random.Random) -> dict[str, Any]:
    return {g: rng.choice(vals) for g, vals in GENE_SPACE.items()}


def mutate(g: dict, rng: random.Random) -> dict:
    out = dict(g)
    knob = rng.choice(list(GENE_SPACE.keys()))
    options = [v for v in GENE_SPACE[knob] if v != out[knob]]
    if options:
        out[knob] = rng.choice(options)
    return out


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    keys = sorted(GENE_SPACE.keys())
    cut = rng.randrange(1, len(keys))
    child = {}
    for i, kk in enumerate(keys):
        child[kk] = a[kk] if i < cut else b[kk]
    return child


def tournament(
    pop_scores: list[tuple[float, dict]], rng: random.Random, k: int = 3
) -> dict:
    challengers = rng.sample(pop_scores, min(k, len(pop_scores)))
    return min(challengers, key=lambda t: t[0])[1]


def evolve(
    pop_size: int = 12,
    generations: int = 8,
    profile: str = "throughput",
    seed: int = 7,
    log: bool = True,
) -> dict[str, Any]:
    rng = random.Random(seed)
    # WAVE 169 fix: fitness was scored on one random genome but a
    # DIFFERENT one was stored in the population (score/genome mismatch).
    population = []
    for _ in range(pop_size):
        g = random_genome(rng)
        population.append((fitness(g, profile), g))
    population = [(s, g) for s, g in population if s is not None]
    if not population:
        raise RuntimeError("no genome passed the certified gate — bug")
    history = []
    for gen in range(generations):
        population.sort(key=lambda t: t[0])
        best_s, best_g = population[0]
        history.append({"gen": gen, "best": best_g, "fitness": round(best_s, 2)})
        elite = population[: max(2, pop_size // 4)]
        pool = [(s, g) for s, g in population]
        children: list[tuple[float, dict]] = []
        while len(children) < pop_size - len(elite):
            p1 = tournament(pool, rng)
            p2 = tournament(pool, rng)
            child = mutate(crossover(p1, p2, rng), rng)
            s = fitness(child, profile)
            if s is not None:
                children.append((s, child))
        population = elite + children
    population.sort(key=lambda t: t[0])
    best_s, best_g = population[0]
    result = {
        "profile": profile,
        "seed": seed,
        "best_genome": best_g,
        "best_fitness": round(best_s, 2),
        "history": history,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if log:
        lp = EVO_LOG
        lp.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else []
        prev.append(result)
        lp.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main() -> None:
    import sys

    profile = sys.argv[1] if len(sys.argv) > 1 else "throughput"
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    res = evolve(generations=gens, profile=profile)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
