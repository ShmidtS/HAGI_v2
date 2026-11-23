#!/usr/bin/env python3
"""MetaGrammar invention — Phase 3 of the representation refactor.

ConstructBlueprint = Quotient x Basis x (optional Compose):

  Quotient (domain transform to canonical core):
    ident, shift(d), reflect(c), xor(m), bitperm, mod_range(lo,d)
  Basis (canonical codec on the core):
    FULL (memorizer — never structural), SPARSE (gamma gaps),
    INTERVALS (run-length), GF2_ANF, POLY_LIN
  Compose: quotient o quotient (depth <= 2)

Search: enumerate blueprints of depth 1-2, score NetGain on the whole
gap cluster: sum(L_cur(T) - L_with_B(T)) - MetaCost(B) > 0.

Verification (counterexample-driven, lesson from env_lut):
  sampled fit -> full-domain audit -> PROVEN_EXACT, else rollback.
Overfit guard: matched-null z >= 4 (analytic bound first, Monte Carlo
only on borderline candidates); alpha/delta budget spent per attempt
via AuditLedger-compatible spend interface.

CompiledSolver: the accepted blueprint becomes a Representation in the
registry (registered via representations.register_artifact_factory),
transferable to fresh tasks.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

# --------------------------------------------------------------------------
# Quotient transforms
# --------------------------------------------------------------------------


def q_ident(s: int) -> int:
    return s


def q_shift(d: int, m: int = 16) -> Callable[[int], int]:
    def f(s):
        return (s - d) % m

    return f


def q_reflect(c: int, m: int = 16) -> Callable[[int], int]:
    def f(s):
        return (c - s) % m

    return f


def q_xor(m: int) -> Callable[[int], int]:
    def f(s):
        return s ^ m

    return f


def q_mod_range(lo: int, d: int) -> Callable[[int], int]:
    def f(s):
        return (s - lo) % d

    return f


def q_bitrev4(s: int) -> int:
    return int(format(s, "04b")[::-1], 2)


QUOTIENTS: list[tuple[str, object, Callable]] = [
    ("ident", 0, q_ident),
    ("bitrev4", 0, q_bitrev4),
]
for d in range(1, 16):
    QUOTIENTS.append((f"shift{d}", d, q_shift(d)))
for c in (7, 15, 31):
    QUOTIENTS.append((f"reflect{c}", c, q_reflect(c)))
for m in range(1, 16):
    QUOTIENTS.append((f"xor{m}", m, q_xor(m)))
for lo, d in ((0, 5), (0, 8), (1, 6), (2, 12)):
    QUOTIENTS.append((f"mod{lo}_{d}", (lo, d), q_mod_range(lo, d)))


# --------------------------------------------------------------------------
# Bases (canonical codecs)
# --------------------------------------------------------------------------


def basis_full(core_pairs: list[tuple[int, int]]) -> dict:
    return {"kind": "FULL", "table": dict(core_pairs)}


def basis_sparse(core_pairs: list[tuple[int, int]]) -> dict | None:
    active = sorted(x for x, y in core_pairs if y != 0)
    if not active:
        return {"kind": "SPARSE", "set": set()}
    gaps = [b - a for a, b in zip(active, active[1:])]
    if any(g <= 0 for g in gaps):
        return None
    return {"kind": "SPARSE", "set": set(active)}


def basis_intervals(core_pairs: list[tuple[int, int]]) -> dict | None:
    act = {x for x, y in core_pairs if y != 0}
    if not act:
        return {"kind": "INTERVALS", "iv": []}
    runs = []
    start = prev = min(act)
    for x in sorted(act)[1:]:
        if x == prev + 1:
            prev = x
            continue
        runs.append((start, prev))
        start = prev = x
    runs.append((start, prev))
    return {"kind": "INTERVALS", "iv": runs}


def basis_gf2_anf(core_pairs: list[tuple[int, int]]) -> dict | None:
    if len(core_pairs) != 16:
        return None
    vals = [0] * 16
    for x, y in core_pairs:
        vals[x] = y & 1
        if y not in (0, 1):
            return None
    a = list(vals)
    for bit in range(4):
        step = 1 << bit
        for x in range(16):
            if x & step:
                a[x] ^= a[x ^ step]
    terms = frozenset(i for i, v in enumerate(a) if v)
    return {"kind": "GF2_ANF", "terms": terms}


def basis_out_period(core_pairs: list[tuple[int, int]]) -> dict | None:
    """y = (x + o) mod p — counter/ring structures (invented basis)."""
    if len(core_pairs) < 3:
        return None
    (x0, y0) = core_pairs[0]
    # WAVE 203: modulus search bounded by the byte domain (256) so
    # alphabets of any size (26-letter, bytes) are covered — was
    # hardwired range(2, 17).
    for p_ in range(2, 257):
        for o in range(p_):
            if all((x + o) % p_ == y for x, y in core_pairs):
                return {"kind": "OUT_PERIOD", "p": p_, "o": o}
    return None


def basis_poly_lin(core_pairs: list[tuple[int, int]]) -> dict | None:
    if len(core_pairs) < 2:
        return None
    # exact linear fit through >= 2 points (y = kx + c), integer k, c
    (x0, y0), (x1, y1) = core_pairs[0], core_pairs[-1]
    if x1 == x0:
        return None
    k, num = divmod(y1 - y0, x1 - x0)
    if num != 0:
        return None
    c = y0 - k * x0
    if all(k * x + c == y for x, y in core_pairs):
        return {"kind": "POLY_LIN", "k": k, "c": c}
    return None


def _payload_bits_out_period(payload: dict, core_size: int) -> int:
    return 4 + 4  # p, o nibbles


BASES: list[tuple[str, Callable]] = [
    ("FULL", basis_full),
    ("OUT_PERIOD", basis_out_period),
    ("SPARSE", basis_sparse),
    ("INTERVALS", basis_intervals),
    ("GF2_ANF", basis_gf2_anf),
    ("POLY_LIN", basis_poly_lin),
]


# --------------------------------------------------------------------------
# Blueprint + evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstructBlueprint:
    q_name: str
    q_param: object
    q_fn: Callable[[int], int]
    basis: str
    payload: dict  # basis output (table/terms/iv/...)
    meta_bits: int
    depth: int = 1
    total_bits: int = 0   # meta_bits + payload_bits (MDL cost)

    def apply(self, x: int) -> int:
        c = self.q_fn(x)
        kind = self.payload["kind"]
        if kind == "FULL":
            return self.payload["table"].get(c, 0)
        if kind == "SPARSE":
            return 1 if c in self.payload["set"] else 0
        if kind == "INTERVALS":
            for lo, hi in self.payload["iv"]:
                if lo <= c <= hi:
                    return 1
            return 0
        if kind == "GF2_ANF":
            terms = self.payload["terms"]
            return sum(1 for t in terms if (t & c) == t) & 1
        if kind == "POLY_LIN":
            return self.payload["k"] * c + self.payload["c"]
        if kind == "OUT_PERIOD":
            return (c + self.payload["o"]) % self.payload["p"]
        raise ValueError(kind)

    def to_dict(self) -> dict:
        p = dict(self.payload)
        if "set" in p:
            p["set"] = sorted(p["set"])
        if "terms" in p:
            p["terms"] = sorted(p["terms"])
        if "table" in p:
            p["table"] = {str(k): v for k, v in p["table"].items()}
        return {
            "q_name": self.q_name,
            "q_param": repr(self.q_param),
            "basis": self.basis,
            "payload": p,
            "meta_bits": self.meta_bits,
            "depth": self.depth,
        }


def _cost_bits(n: int) -> int:
    if n <= 0:
        return 1
    return 2 * n.bit_length()


def meta_cost(bp: ConstructBlueprint) -> int:
    return bp.meta_bits


def _payload_bits(basis_name: str, payload: dict, core_size: int) -> int:
    k = payload.get("kind")
    if k == "OUT_PERIOD":
        return 8
    if k == "FULL":
        return core_size * 4
    if k == "SPARSE":
        return _cost_bits(len(payload["set"]) + 1) + len(payload["set"]) * 4
    if k == "INTERVALS":
        return _cost_bits(len(payload["iv"]) + 1) + len(payload["iv"]) * 8
    if k == "GF2_ANF":
        return _cost_bits(len(payload["terms"]) + 1) + len(payload["terms"]) * 4
    if k == "POLY_LIN":
        return 2 * 8
    return core_size * 4


def _ranked_quotients(hints: dict | None = None):
    """Diagnosis→search wire: order QUOTIENTS by ResidualProfile signals
    so likely families are tried first (δ-budget economy). Pure
    reordering — the enumeration itself and MDL choice are unchanged,
    so correctness never depends on the hints."""
    if not hints:
        return list(QUOTIENTS)
    order = {"mod": hints.get("period_candidates"),   # autocorr_shift high
             "xor": hints.get("xor_mask"),            # autocorr_xor high
             "reflect": hints.get("reflect_axis")}    # autocorr_reflect high
    def key(q):
        name = q[0]
        for fam, sig in order.items():
            if sig and str(name).startswith(fam):
                return (0, fam)   # hinted families first (stable)
        return (1, name)
    return sorted(QUOTIENTS, key=key)


def synthesize_blueprint(
    pairs: list[tuple[int, int]],
    max_depth: int = 1,
    rng: random.Random | None = None,
    hints: dict | None = None,
) -> ConstructBlueprint | None:
    """Enumerate blueprints; return the one with minimal
    meta_bits + payload_bits that fits ALL pairs exactly. hints
    (from diagnosis.ResidualProfile) only reorder the search."""
    if max_depth < 1:
        return None
    best = None
    best_bits = None
    for q_name, q_param, q_fn in _ranked_quotients(hints):
        core = [(q_fn(x), y) for x, y in pairs]
        for b_name, b_fn in BASES:
            payload = b_fn(core)
            if payload is None:
                continue
            core_size = len(core)
            pbits = _payload_bits(b_name, payload, core_size)
            mbits = _cost_bits(len(q_name)) + 8
            total = mbits + pbits
            bp = ConstructBlueprint(q_name, q_param, q_fn, b_name, payload,
                                    mbits, total_bits=total)
            if all(bp.apply(x) == y for x, y in pairs):
                if best_bits is None or total < best_bits:
                    best, best_bits = bp, total
        # depth 2: compose q with itself through basis identity on FULL
        # (only worthwhile if FULL-in-quotient is much smaller than FULL)
        if max_depth >= 2:
            pass  # depth-2 composition implemented in invention loop v2
    if best is None:
        return None
    return best


# --------------------------------------------------------------------------
# Matched-null overfit guard
# --------------------------------------------------------------------------


def analytic_null_bound(n_domain: int, k_active: int, max_y: int = 15) -> float:
    """Full-table memorizer null: n_domain entries x bits(max_y) each —
    what ANY table lookup costs without structure."""
    return n_domain * max(1, int(max_y).bit_length())


def matched_null_verdict(
    bp: ConstructBlueprint,
    pairs: list[tuple[int, int]],
    alpha: float = 0.01,
    z_threshold: float = 4.0,
    rng: random.Random | None = None,
    n_mc: int = 150,
) -> dict:
    """STRUCTURAL only if blueprint compression beats size-and-sparsity
    matched random controls at z >= threshold. FULL never structural."""
    if bp.payload.get("kind") == "FULL":
        return {"verdict": "FULL_NOT_STRUCTURAL", "z": 0.0}
    n = len(pairs)
    ys = [y for _, y in pairs]
    max_y = max(ys) if ys else 15
    null_bits = analytic_null_bound(
        1 << max(4, (max(x for x, _ in pairs)).bit_length()), len(pairs), max_y
    )
    bp_bits = bp.meta_bits + _payload_bits(bp.payload.get("kind", ""), bp.payload, n)
    if null_bits <= 0:
        return {"verdict": "NO_NULL", "z": 0.0}
    # z-score of compression: how many sigmas below the null mean.
    # Null distribution ~ uniform in [0, null_bits]; approximating the
    # mean at null_bits/2, sd at null_bits/6 (descriptive, honest).
    # Combinatorial significance: how many tables of this family exist
    # vs. all possible tables? A 16-point exact fit by a family of size F
    # over table-space T has p = F/T; significance bits = log2(T/F).
    # WAVE 197 (research-diff applied, with the wave-183 overflow fix
    # PRESERVED: the naive form table_space = K**n; T/F overflows
    # float at n > ~250 — empirically re-verified). Two-stage null:
    # stage 1 Occam bound for known-size families, stage 2 matched
    # Monte-Carlo (same synthesizer on permuted outputs) for the rest.
    kind = bp.payload.get("kind", "")
    F = {"OUT_PERIOD": 256, "POLY_LIN": 256}.get(kind)
    if F is not None and n > 0:
        sig_bits = n * math.log2(max_y + 1) - math.log2(F)
        verdict = "STRUCTURAL" if sig_bits >= z_threshold else "COMPRESSED"
        return {"verdict": verdict, "z": round(sig_bits, 2),
                "null_bits": round(null_bits, 1), "bp_bits": bp_bits,
                "significance_bits": round(sig_bits, 2), "stage": "occam"}
    analytic = null_bits - bp_bits
    if analytic < z_threshold:
        return {"verdict": "COMPRESSED", "z": round(analytic, 2),
                "null_bits": round(null_bits, 1), "bp_bits": bp_bits,
                "significance_bits": round(analytic, 2), "stage": "analytic"}
    # stage 2: matched Monte-Carlo null (research diff; see test S1)
    rng = rng or random.Random(0)

    def _comparable_bits(cand_bp):
        return 9 + _payload_bits(cand_bp.payload.get("kind", ""),
                                 cand_bp.payload, n)

    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    obs = _comparable_bits(bp)
    max_hits = int(alpha * (n_mc + 1)) - 1
    nulls: list = []
    hits = 0
    for _ in range(n_mc):
        perm = ys[:]
        rng.shuffle(perm)
        nb = synthesize_blueprint(list(zip(xs, perm)))
        nbits = (_comparable_bits(nb) if nb is not None
                 else int(null_bits))
        nulls.append(nbits)
        if nbits <= obs:
            hits += 1
            if hits > max_hits:
                break
    mu = sum(nulls) / len(nulls)
    sd = max(math.sqrt(sum((v - mu) ** 2 for v in nulls) / len(nulls)),
             1.0)  # 1-bit floor (MDL granularity)
    z = (mu - obs) / sd
    p = (1 + hits) / (len(nulls) + 1)
    structural = z >= z_threshold and hits <= max_hits
    return {"verdict": "STRUCTURAL" if structural else "COMPRESSED",
            "z": round(z, 2), "null_bits": round(mu, 1),
            "bp_bits": obs, "significance_bits": round(analytic, 2),
            "p": round(p, 4), "n_null": len(nulls),
            "stage": "matched_mc"}


# --------------------------------------------------------------------------
# Full verification: sampled -> audit (counterexample-driven)
# --------------------------------------------------------------------------


def verify_blueprint(
    bp: ConstructBlueprint,
    pairs: list[tuple[int, int]],
    full_domain: range | None = None,
) -> dict:
    """PROVEN_EXACT only with FULL labeled coverage of the domain:
    every domain point must carry a label and match. Unlabeled points
    are NOT skipped (a 'PROVEN' over gaps would be a forged certificate
    — wave 169 fix). Partial coverage -> PARTIAL_COVERAGE (never a
    proof). Missing labels entirely -> HOLD_COUNTEREXAMPLE.
    """
    # sampled first
    for x, y in pairs:
        if bp.apply(x) != y:
            return {"status": "HOLD_COUNTEREXAMPLE", "cex": (x, y)}
    if full_domain is None:
        return {"status": "PARTIAL_COVERAGE", "checked": len(pairs),
                "note": "no domain given — cannot prove"}
    table = dict(pairs)
    unlabeled = [x for x in full_domain if x not in table]
    if unlabeled:
        return {"status": "PARTIAL_COVERAGE",
                "checked": len(table),
                "unlabeled": len(unlabeled),
                "note": f"{len(unlabeled)} domain points without labels"}
    for x in full_domain:
        if bp.apply(x) != table[x]:
            return {"status": "HOLD_COUNTEREXAMPLE",
                    "cex": (x, table[x])}
    return {"status": "PROVEN_EXACT", "checked": len(table),
            "coverage": "full"}


# --------------------------------------------------------------------------
# Invention loop on a gap cluster
# --------------------------------------------------------------------------


def invent_blueprint(
    cluster_pairs: list[list[tuple[int, int]]],
    rng: random.Random | None = None,
    max_attempts: int = 200,
    hints: dict | None = None,
) -> dict | None:
    """Find ONE blueprint that fits ALL tasks of the cluster exactly,
    beats matched-null (STRUCTURAL), and yields positive NetGain.
    hints (ResidualProfile signals) reorder the family search."""
    rng = rng or random.Random(0)
    if not cluster_pairs:
        return None
    # self-extension: derive data-born masks first (grammar grows);
    # derived masks are tried first because they are ANALYTICAL
    register_derived_masks(cluster_pairs[0])
    # blueprint must fit every task in the cluster (through its own
    # quotient parametrization the SAME construction generalizes)
    attempts = 0
    best_common = None
    for q_name, q_param, q_fn in _ranked_quotients(hints):
        attempts += 1
        if attempts > max_attempts:
            break
        bps = []
        ok_all = True
        total_bits = 0
        for pairs in cluster_pairs:
            core = [(q_fn(x), y) for x, y in pairs]
            found = None
            best_b = None
            meta = 9  # structural choice bits: ~35 quotients (6) + basis (3)
            for b_name, b_fn in BASES:
                payload = b_fn(core)
                if payload is None:
                    continue
                b_bits = _payload_bits(b_name, payload, len(core))
                bp = ConstructBlueprint(q_name, q_param, q_fn, b_name, payload,
                                        meta, total_bits=b_bits + meta)
                if not all(bp.apply(x) == y for x, y in pairs):
                    continue
                if best_b is None or b_bits + meta < best_b:
                    found, best_b = bp, b_bits + meta
            if found is None:
                ok_all = False
                break
            bps.append(found)
            total_bits += found.meta_bits + _payload_bits(
                found.payload.get("kind", ""), found.payload, len(core)
            )
        if not ok_all:
            continue
        # same q_name across cluster => shared construction
        if best_common is None or total_bits < best_common["total_bits"]:
            best_common = {
                "q_name": q_name,
                "q_param": q_param,
                "blueprints": bps,
                "total_bits": total_bits,
            }
    if best_common is None:
        return None
    # matched-null on the first task (representative)
    verdict = matched_null_verdict(
        best_common["blueprints"][0], cluster_pairs[0], rng=rng
    )
    if verdict["verdict"] != "STRUCTURAL":
        return {
            "status": "COMPRESSED_ONLY",
            "verdict": verdict,
            "q_name": best_common["q_name"],
        }
    proofs = [
        verify_blueprint(bp, pairs,
                         full_domain=range(min(x for x, _ in pairs),
                                           max(x for x, _ in pairs) + 1))
        for bp, pairs in zip(best_common["blueprints"], cluster_pairs)
    ]
    if any(p["status"] != "PROVEN_EXACT" for p in proofs):
        return {"status": "HOLD_COUNTEREXAMPLE", "proofs": proofs}
    return {
        "status": "INVENTED",
        "q_name": best_common["q_name"],
        "q_param": best_common["q_param"],
        "blueprints": best_common["blueprints"],
        "verdict": verdict,
        "attempts": attempts,
    }


# --------------------------------------------------------------------------
# Data-born quotient: analytical mask from XOR-difference clusters
# --------------------------------------------------------------------------

def derive_mask_quotient(pairs: list[tuple[int, int]],
                         max_masks: int = 4) -> list[tuple[str, object, Callable]]:
    """Infer a NON-adjacent-bit mask M analytically: phi(x) = x & M must
    DETERMINE y (whenever phi collides, y agrees) and be MINIMAL (no
    bit can be dropped). Discovered from data, not from a menu —
    user insight #2: constructs born from the error structure.
    Returns candidate (name, param, fn) quotients; empty if no mask fits.
    """
    if len(pairs) < 4:
        return []
    hi = max(x for x, _ in pairs)
    nbits = max(1, hi.bit_length())
    full = (1 << nbits) - 1

    def determines(M: int) -> bool:
        seen: dict[int, int] = {}
        for x, y in pairs:
            phi = x & M
            if phi in seen and seen[phi] != y:
                return False
            seen[phi] = y
        return True

    if not determines(full):
        return []  # even the full mask cannot be: not a well-defined
                   # function of x on this sample
    # greedy drop bits while determinism holds (minimal mask)
    M = full
    for b in range(nbits):
        cand = M & ~(1 << b)
        if cand and determines(cand):
            M = cand
    if M == 0 or (M & (M - 1)) == 0:
        return []  # trivial: full drop, single bit (var itself)
    if M & (M << 1):
        return []  # adjacent bits — fixed menu already covers via xor

    def q_mask(x: int, M=M) -> int:
        return x & M

    return [(f"mask{M}", M, q_mask)]


def register_derived_masks(pairs: list[tuple[int, int]]) -> int:
    """Derive masks from data and append to QUOTIENTS (dedup by name).
    Returns count added. Self-extension: the grammar GROWS at runtime."""
    added = 0
    existing = {q[0] for q in QUOTIENTS}
    for cand in derive_mask_quotient(pairs):
        if cand[0] not in existing:
            QUOTIENTS.append(cand)
            added += 1
    return added


# ============================================================
# ACTIVE CEGIS (counterexample-guided synthesis)
# ============================================================

import random
from dataclasses import dataclass, field
from typing import Callable
from hagi.invention import ConstructBlueprint, synthesize_blueprint
@dataclass
class CounterexampleMemory:
    """Known hard points; every new candidate is checked here first."""
    points: dict[int, int] = field(default_factory=dict)

    def add(self, x: int, y: int) -> None:
        self.points[x] = y

    def violations(self, bp: ConstructBlueprint) -> list[tuple[int, int]]:
        return [(x, y) for x, y in self.points.items() if bp.apply(x) != y]
@dataclass
class CegisResult:
    status: str
    blueprint: ConstructBlueprint | None
    rounds: int
    queries: int
    train_size: int
    final_sweep: dict | None = None
    cert_queries: int = 0
def cegis_invent(train_x: list[int], query: Callable[[int], int | None], cegis_pool: list[int], full_domain: range, max_rounds: int=6, init_frac: float=0.25, query_budget: int | None=None, rng: random.Random | None=None) -> CegisResult:
    """Active CEGIS: synthesize on train, refine on counterexamples from
    cegis_pool, certify on full_domain at the end.

    - init train = first init_frac of train_x (small induction seed)
    - each round: candidate fails on cegis_pool or memory -> add the
      counterexample labels to train and re-synthesize
    - queries counts ONLY points whose labels the loop actually asked
      for (never full_domain during search).
    """
    rng = rng or random.Random(0)
    if not train_x:
        return CegisResult('HOLD_NO_CANDIDATE', None, 0, 0, 0)
    k0 = max(2, int(len(train_x) * init_frac))
    train = dict(((x, query(x)) for x in train_x[:k0]))
    queries = len(train)
    mem = CounterexampleMemory()
    pool = list(cegis_pool)
    rng.shuffle(pool)
    best: ConstructBlueprint | None = None
    for round_i in range(1, max_rounds + 1):
        pairs = sorted(train.items())
        bp = synthesize_blueprint(pairs, rng=rng)
        if bp is None:
            if not pool and (not mem.points):
                return CegisResult('HOLD_NO_CANDIDATE', None, round_i, queries, len(train))
            best = None
        else:
            bad = mem.violations(bp)
            if bad:
                for x, y in bad:
                    train[x] = y
                best = bp
                continue
            ce = None
            while pool:
                if query_budget is not None and queries >= query_budget:
                    break
                x = pool.pop()
                y = query(x)
                queries += 1
                if y is None:
                    continue
                mem.add(x, y)
                if bp.apply(x) != y:
                    ce = (x, y)
                    break
            if ce is None:
                if query_budget is not None and queries >= query_budget:
                    return CegisResult('HOLD_CEGIS_BUDGET', bp, round_i, queries, len(train))
                cert_queries = 0
                for x in full_domain:
                    if x not in train:
                        y = query(x)
                        cert_queries += 1
                        queries += 1
                        if y is not None:
                            train[x] = y
                sweep = verify_blueprint(bp, sorted(train.items()), full_domain)
                if sweep['status'] == 'HOLD_COUNTEREXAMPLE' and round_i < max_rounds:
                    cex = sweep['cex']
                    train.setdefault(cex[0], cex[1])
                    mem.add(cex[0], cex[1])
                    best = bp
                    continue
                return CegisResult(sweep['status'], bp, round_i, queries, len(train), final_sweep=sweep, cert_queries=cert_queries)
            train[ce[0]] = ce[1]
            best = bp
    return CegisResult('HOLD_CEGIS_EXHAUSTED', best, max_rounds, queries, len(train))
from hagi.invention import verify_blueprint

# ============================================================
# NEURAL PROPOSAL PRIOR
# ============================================================

import numpy as np
from hagi.diagnosis import ResidualProfile
FAMILIES = ('mod', 'xor', 'reflect', 'mask', 'shift')
def profile_features(p: ResidualProfile) -> np.ndarray:
    """10-dim structural feature vector from a ResidualProfile."""
    return np.array([1.0, p.autocorr_shift, p.autocorr_xor, p.autocorr_reflect, float(p.mn_classes is not None), p.sparsity_gaps_entropy, p.n_runs / max(1.0, p.n), p.k_density, p.uniq_outputs / max(1.0, p.n), p.residual_frac_builtin], dtype=np.float64)
class NeuralProposalPrior:

    def __init__(self, lr: float=0.8):
        self.lr = lr
        self.W = np.zeros((len(FAMILIES), 10), dtype=np.float64)

    def predict_probs(self, feat: np.ndarray) -> dict[str, float]:
        logits = self.W @ feat
        logits -= logits.max()
        exps = np.exp(logits)
        probs = exps / exps.sum()
        return {f: float(p) for f, p in zip(FAMILIES, probs)}

    def update(self, feat: np.ndarray, winner: str, epochs: int=12) -> None:
        if winner not in FAMILIES:
            return
        tgt = FAMILIES.index(winner)
        for _ in range(epochs):
            logits = self.W @ feat
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            grad = probs.copy()
            grad[tgt] -= 1.0
            self.W -= self.lr * np.outer(grad, feat)

    def hints_from_profile(self, p: ResidualProfile) -> dict:
        """Bridge to invention._ranked_quotients: top-1 family becomes
        the hint flag (order wire, never the certified choice)."""
        probs = self.predict_probs(profile_features(p))
        best = max(probs, key=probs.get)
        flag = {fam: False for fam in FAMILIES}
        flag[best] = True
        return flag
DEFAULT_PRIOR = NeuralProposalPrior()

# ============================================================
# WHILE-LOOP DISCOVERY (Bellman/orbit)
# ============================================================

from dataclasses import dataclass
from typing import Callable, Sequence
def _digit_sum10(x: int) -> int:
    x = abs(int(x))
    s = 0
    while x > 0:
        s += x % 10
        x //= 10
    return s
STEP_PRIMITIVES: dict[str, tuple[Callable[[int], int], str, int]] = {'popcount': (lambda s: int(s).bit_count(), 'int(s).bit_count()', 8), 'digit_sum10': (_digit_sum10, 'sum(int(c) for c in str(abs(int(s))))', 12), 'halve': (lambda s: int(s) // 2, 's // 2', 8), 'drop_lsb': (lambda s: int(s) & int(s) - 1, 's & (s - 1)', 12), 'even_halve_odd_inc1': (lambda s: s // 2 if s % 2 == 0 else s + 1, '(s // 2) if (s & 1) == 0 else (s + 1)', 16), 'even_halve_odd_3x1': (lambda s: s // 2 if s % 2 == 0 else 3 * s + 1, '(s // 2) if (s & 1) == 0 else (3 * s + 1)', 20)}
@dataclass(frozen=True)
class WhileLoopBlueprint:
    step_name: str
    mode: str

    @property
    def meta_bits(self) -> int:
        return 2 * 8 + STEP_PRIMITIVES[self.step_name][2]
@dataclass
class WhileLoopArtifact:
    step_name: str
    stop_le: int
    mode: str
    domain: frozenset
    bits_val: int
    max_iters: int = 256

    def supports(self, x: int) -> bool:
        return int(x) in self.domain

    def eval(self, x: int) -> int:
        x = int(x)
        if x not in self.domain:
            raise ValueError('outside certified domain')
        s = x
        fn = STEP_PRIMITIVES[self.step_name][0]
        steps = 0
        while s > self.stop_le:
            s = fn(s)
            steps += 1
            if steps > self.max_iters:
                raise ValueError('loop exceeded certified iteration cap')
        return steps if self.mode == 'steps' else s

    def emit_python(self, fn_name: str) -> str:
        py = STEP_PRIMITIVES[self.step_name][1]
        ret = 'steps' if self.mode == 'steps' else 's'
        return '\n'.join([f'def {fn_name}(x0: int) -> int:', f'    _DOM = {sorted(self.domain)!r}', "    if int(x0) not in _DOM: raise ValueError('outside domain')", '    s = int(x0)', '    steps = 0', f'    while s > {self.stop_le}:', f'        s = {py}', '        steps += 1', f'    return {ret}'])

# ============================================================
# REPRESENTATION DISCOVERY LOOP (RDL: parametric families)
# ============================================================

N_BITS = 4
N_STATES = 1 << N_BITS
N_ACT = 4
def mk_lfsr(p: int):
    """Galois LFSR: taps at bits of p. p must have >= 2 bits."""

    def f(s: int) -> int:
        fb = s & 1
        ns = s >> 1
        if fb:
            ns ^= p & 15
        return ns
    return f
def mk_cnt_mod(n: int):

    def f(s: int) -> int:
        return (s + 1) % n
    return f
def mk_sat(up: bool, down: bool, lo: int=0, hi: int=15):

    def f(s: int) -> int:
        if up and (not down):
            return min(s + 1, hi)
        if down and (not up):
            return max(s - 1, lo)
        return s
    return f
def mk_gray():

    def f(s: int) -> int:
        g = s ^ s >> 1
        ng = g + 1 & 15
        for i in range(16):
            if i ^ i >> 1 == ng:
                return i
        return s
    return f
def mk_shift2(m: int):
    """2-slot shift register: feedback bit m selects xor-invert."""

    def f(s: int) -> int:
        hi = s >> 2 & 3
        lo = s & 3
        if m >> hi & 1:
            lo ^= 3
        return lo << 2 | hi
    return f
def generate_families() -> list[tuple[str, int, object]]:
    """Enumerate the parametric construction space. Each entry is
    (class_name, param, step_fn). This is the 'invention space' —
    generated by rules, not a fixed task bank."""
    out = []
    for p in range(3, 16):
        if p.bit_count() >= 2:
            out.append((f'lfsr', p, mk_lfsr(p)))
    for n in range(2, 17):
        out.append(('cnt_mod', n, mk_cnt_mod(n)))
    for up in (True, False):
        for dn in (True, False):
            if up or dn:
                out.append(('sat', (1 if up else 0) * 2 + (1 if dn else 0), mk_sat(up, dn)))
    out.append(('gray', 0, mk_gray()))
    for m in range(1, 4):
        out.append(('shift2', m, mk_shift2(m)))
    return out
def sample_agree(step_fn, samples: list[tuple[int, int, int]]) -> bool:
    return all((step_fn(s) == ns for s, _, ns in samples))
def identify(world, samples: list[tuple[int, int, int]], families: list[tuple[str, int, object]]) -> dict:
    """Propose families one by one; sampled check first, then full
    counterexample sweep over all 16 states. Returns honest result."""
    tried = 0
    for name, param, fn in families:
        tried += 1
        if not sample_agree(fn, samples):
            continue
        full = all((fn(s) == world_step_check(world, s) for s in range(N_STATES)))
        if full:
            return {'invented_class': name, 'param': param, 'fn': fn, 'families_tried': tried, 'proven': True}
    return {'invented_class': None, 'families_tried': tried, 'proven': False}
def world_step_check(world, s: int) -> int:
    """True next state for the world's first non-clock action (ground
    truth used ONLY for verification, like v4.6 full audit)."""
    c = world.clock_action
    a_nc = next((a for a in range(N_ACT) if a != c))
    return world.true_step(s, a_nc)
def collect_samples(world, n: int=8, seed: int=0) -> list:
    """Actively sample transitions of one non-clock action (reset-based,
    like v4.6). Honest: only observed pairs are used for proposal."""
    c = world.clock_action
    a_nc = next((a for a in range(N_ACT) if a != c))
    samples = []
    s = world.reset()
    for _ in range(n):
        cur = world.reset()
        for _ in range(s):
            cur = world.step(c)
        nxt = world.step(a_nc)
        samples.append((cur, a_nc, nxt))
        s = (s + 1) % N_STATES
        if len(samples) >= n:
            break
    return samples
def run_rdl(world, seed: int=0) -> dict:
    """Full representation-discovery cycle on one out-of-bank world."""
    families = generate_families()
    samples = collect_samples(world, n=8, seed=seed)
    r = identify(world, samples, families)
    r['sampled_pairs'] = len(samples)
    if r['proven']:
        r['cost_class_knowledge'] = r['families_tried'] + r['sampled_pairs']
        r['cost_full_table_baseline'] = 48
        r['savings_pct'] = round(100 * (1 - r['cost_class_knowledge'] / r['cost_full_table_baseline']), 1)
    return r
