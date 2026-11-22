"""Measurement suite (WAVE 195): merged from 5 modules (transfer_eval, capability, budget, oracle, active)."""
from __future__ import annotations


# ============================================================
# TRANSFER GATE METER
# ============================================================

from dataclasses import dataclass, field
from typing import Callable
@dataclass
class TaskResult:
    task_name: str
    solved: bool
    structural: bool
    program: bool
    bits: int
    probes: int
@dataclass
class MeterVerdict:
    action: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
class CapabilityMeter:

    def __init__(self, alpha_fdr: float=0.0):
        self.alpha_fdr = alpha_fdr

    def evaluate(self, held_out_results: list[TaskResult], matched_control_results: list[TaskResult], pre_solve_count: int, pre_program_count: int, pre_avg_bits: float, post_avg_bits: float, pre_avg_probes: float, post_avg_probes: float) -> MeterVerdict:
        reasons: list[str] = []
        m: dict = {}
        solved = [r for r in held_out_results if r.solved]
        structural = [r for r in solved if r.structural]
        programs = [r for r in solved if r.program]
        if len(solved) < pre_solve_count:
            reasons.append(f'REGRESSION: solved {len(solved)} < {pre_solve_count}')
        if len(programs) < pre_program_count:
            reasons.append(f'REGRESSION: programs {len(programs)} < {pre_program_count}')
        m['solved'] = len(solved)
        m['structural'] = len(structural)
        m['programs'] = len(programs)
        delta = len(structural) + len(programs) - pre_solve_count
        m['delta_capacity'] = delta
        if delta <= 0:
            reasons.append(f'NO_CAPACITY_GAIN: delta={delta}')
        false_structural = sum((1 for r in matched_control_results if r.structural))
        m['false_structural_controls'] = false_structural
        if false_structural > self.alpha_fdr:
            reasons.append(f'FALSE_STRUCTURAL on controls: {false_structural}')
        bits_win = post_avg_bits < pre_avg_bits
        probes_win = post_avg_probes < pre_avg_probes
        m['bits_win'] = bits_win
        m['probes_win'] = probes_win
        m['avg_bits'] = (pre_avg_bits, post_avg_bits)
        m['avg_probes'] = (pre_avg_probes, post_avg_probes)
        if not (bits_win or probes_win):
            reasons.append('NO_COST_WIN')
        if reasons:
            return MeterVerdict('ROLLBACK', reasons, m)
        return MeterVerdict('COMMIT', [], m)
def rollback(registry_remove: Callable[[str], None], repr_id: str, restore_snapshot: Callable[[], None]) -> MeterVerdict:
    """Honest rollback: remove the registry entry, restore the snapshot."""
    try:
        registry_remove(repr_id)
    finally:
        restore_snapshot()
    return MeterVerdict('ROLLBACK', ['rolled back'], {'repr_id': repr_id})

# ============================================================
# CAPABILITY SNAPSHOT
# ============================================================

from dataclasses import dataclass
from typing import Callable
@dataclass
class CapabilitySnapshot:
    version: int
    library_meta_bits: int
    experts_bits: int
    unsolved_penalty_bits: int
    total_mdl_bits: int
    heldout_solve_rate: float
    mean_samples_to_solve: float
    noise_false_accepts: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()
def total_mdl(library_meta_bits: int, expert_bits: list[int], unsolved: int, penalty_per_unsolved: int=64) -> int:
    return int(library_meta_bits) + sum((int(b) for b in expert_bits)) + int(unsolved) * int(penalty_per_unsolved)
def capability_delta(before: CapabilitySnapshot, after: CapabilitySnapshot) -> dict:
    d_mdl = after.total_mdl_bits - before.total_mdl_bits
    d_rate = after.heldout_solve_rate - before.heldout_solve_rate
    d_samples = after.mean_samples_to_solve - before.mean_samples_to_solve
    return {'delta_total_mdl': d_mdl, 'delta_heldout_solve_rate': d_rate, 'delta_mean_samples': d_samples, 'noise_false_accepts': after.noise_false_accepts, 'REAL_GAIN': d_mdl < 0 and d_rate > 0 and (after.noise_false_accepts == 0)}
def heldout_suite(tasks: list[tuple[str, Callable, range, int]], solver: Callable[[str, range, int], tuple[int, int] | None]) -> tuple[int, float, float]:
    """Run held-out transfer panel. solver(name, domain, target) returns
    (bits_cost | None-if-HOLD, samples_used). Returns (solved, rate,
    mean_samples_to_solve over solved only)."""
    solved = 0
    samples = []
    for name, fn, dom, target in tasks:
        res = solver(name, dom, target)
        if res is not None and res[0] is not None:
            solved += 1
            samples.append(res[1])
    rate = solved / max(1, len(tasks))
    mean_s = sum(samples) / len(samples) if samples else 0.0
    return (solved, rate, mean_s)
def noise_panel(noise_tables: list[dict[int, int]], inventor: Callable[[dict[int, int]], object]) -> int:
    """Feed randomized tables; count accepted STRUCTURAL constructs.
    Honest system must return 0."""
    accepts = 0
    for tbl in noise_tables:
        if inventor(tbl) is not None:
            accepts += 1
    return accepts

# ============================================================
# ANYTIME ALPHA BUDGET
# ============================================================

import math
from typing import Sequence
def weighted_exp_budget(kappa: Sequence[float], w: Sequence[float]) -> float:
    return sum((wi / ki for wi, ki in zip(w, kappa)))
def weighted_exp_log_coeff(c: Sequence[float], kappa: Sequence[float], w: Sequence[float]) -> float:
    return sum((wi / ki * math.log(ci * ki / wi) for ci, ki, wi in zip(c, kappa, w)))
def weighted_exp_allocation(c: Sequence[float], kappa: Sequence[float], w: Sequence[float], B: float) -> list[float]:
    """Exact closed form (Budget/WeightedExpAlloc.lean, C428):
    alloc_i = (log(c_i·κ_i/w_i) − (logCoeff − B)/budget) / κ_i."""
    if not len(c) == len(kappa) == len(w) or not c:
        raise ValueError('c/kappa/w must be same non-empty length')
    if any((k <= 0 for k in kappa)) or any((wi <= 0 for wi in w)) or any((ci <= 0 for ci in c)):
        raise ValueError('all c, kappa, w must be positive')
    budget = weighted_exp_budget(kappa, w)
    coeff = weighted_exp_log_coeff(c, kappa, w)
    shift = (coeff - B) / budget
    return [(math.log(ci * ki / wi) - shift) / ki for ci, ki, wi in zip(c, kappa, w)]
def fisher_tolerances(col_curv: Sequence[float], B: float, w: Sequence[float] | None=None) -> list[float]:
    """C472 bridge: per-coordinate quantization tolerance ε_j from the
    Gershgorin column curvature c_j (hess_le_colCurv). Flat coordinates
    (small c_j) tolerate more error — tolerance is MONOTONE decreasing
    in curvature (fisher_tolerance_mono)."""
    n = len(col_curv)
    ww = w if w is not None else [1.0] * n
    if any((cj <= 0 for cj in col_curv)):
        raise ValueError('column curvature must be positive')
    return weighted_exp_allocation([1.0] * n, list(col_curv), ww, B)

# ============================================================
# CACHED ORACLE
# ============================================================

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable
class OracleQueryBudgetExceeded(RuntimeError):
    """Raised before a teacher call would exceed the unique-query budget."""
@dataclass(frozen=True)
class OracleStats:
    unique_queries: int
    requests: int
    cache_hits: int
    unique_by_purpose: tuple[tuple[str, int], ...]
    cache_hits_by_purpose: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {'unique_queries': self.unique_queries, 'requests': self.requests, 'cache_hits': self.cache_hits, 'unique_by_purpose': dict(self.unique_by_purpose), 'cache_hits_by_purpose': dict(self.cache_hits_by_purpose)}
class CachedOracle:
    """A per-task memoizing wrapper around a deterministic teacher callable.

    Args:
        teacher: Callable that accepts one positional value per input axis.
        arity: Optional expected number of input coordinates.
        max_unique_queries: Optional hard cap on distinct teacher calls.

    The wrapper stores integer labels because the current HAGI symbolic task
    interface is integer-valued. It is intentionally ignorant of task
    semantics, train/validation splits and active acquisition policy.
    """

    def __init__(self, teacher: Callable[..., int], *, arity: int | None=None, max_unique_queries: int | None=None) -> None:
        if not callable(teacher):
            raise TypeError('teacher must be callable')
        if arity is not None and arity < 1:
            raise ValueError('arity must be positive')
        if max_unique_queries is not None and max_unique_queries < 0:
            raise ValueError('max_unique_queries must be non-negative')
        self._teacher = teacher
        self.arity = arity
        self.max_unique_queries = max_unique_queries
        self._labels: dict[tuple[int, ...], int] = {}
        self._requests = 0
        self._cache_hits = 0
        self._unique_by_purpose: Counter[str] = Counter()
        self._cache_hits_by_purpose: Counter[str] = Counter()

    def query(self, inputs: Iterable[int], *, purpose: str='unspecified') -> int:
        """Return a cached label or call the teacher once for this tuple."""
        key = tuple((int(value) for value in inputs))
        if self.arity is not None and len(key) != self.arity:
            raise ValueError(f'oracle input arity {len(key)} != expected {self.arity}')
        if not purpose or not isinstance(purpose, str):
            raise ValueError('purpose must be a non-empty string')
        self._requests += 1
        if key in self._labels:
            self._cache_hits += 1
            self._cache_hits_by_purpose[purpose] += 1
            return self._labels[key]
        if self.max_unique_queries is not None and len(self._labels) >= self.max_unique_queries:
            raise OracleQueryBudgetExceeded(f'unique oracle query budget exhausted ({self.max_unique_queries})')
        label = int(self._teacher(*key))
        self._labels[key] = label
        self._unique_by_purpose[purpose] += 1
        return label

    @property
    def unique_queries(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> dict[tuple[int, ...], int]:
        """Return a defensive snapshot for fingerprints and audit reports."""
        return dict(self._labels)

    def stats(self) -> OracleStats:
        return OracleStats(unique_queries=len(self._labels), requests=self._requests, cache_hits=self._cache_hits, unique_by_purpose=tuple(sorted(self._unique_by_purpose.items())), cache_hits_by_purpose=tuple(sorted(self._cache_hits_by_purpose.items())))

# ============================================================
# QUERY-BY-COMMITTEE
# ============================================================

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Hashable, Sequence, TypeVar
H = TypeVar('H')
X = TypeVar('X')
Y = TypeVar('Y', bound=Hashable)
@dataclass(frozen=True)
class QueryChoice:
    point: object
    information_gain_bits: float
    outcome_probabilities: tuple[tuple[Hashable, float], ...]
def select_query_by_committee(candidates: Sequence[H], pool: Sequence[X], predict: Callable[[H, X], Y], weights: Sequence[float] | None=None) -> QueryChoice | None:
    """Choose the pool point with greatest predictive-label entropy.

    ``weights`` are non-negative posterior/prior masses for candidates. The
    function is deterministic: ties preserve the supplied pool order. It
    returns ``None`` when fewer than two hypotheses remain or no pool point
    separates them. A point is never queried here; this is pure selection.
    """
    if len(candidates) < 2 or not pool:
        return None
    if weights is None:
        masses = [1.0] * len(candidates)
    else:
        if len(weights) != len(candidates):
            raise ValueError('weights must match the candidate count')
        masses = [float(value) for value in weights]
        if any((not math.isfinite(value) or value < 0 for value in masses)):
            raise ValueError('weights must be finite and non-negative')
    total_mass = sum(masses)
    if total_mass <= 0:
        raise ValueError('at least one candidate weight must be positive')
    masses = [value / total_mass for value in masses]
    best: QueryChoice | None = None
    for point in pool:
        outcome_mass: dict[Hashable, float] = defaultdict(float)
        for candidate, mass in zip(candidates, masses):
            if mass <= 0:
                continue
            outcome = predict(candidate, point)
            outcome_mass[outcome] += mass
        if len(outcome_mass) < 2:
            continue
        entropy = -sum((prob * math.log2(prob) for prob in outcome_mass.values() if prob > 0))
        if best is None or entropy > best.information_gain_bits + 1e-12:
            best = QueryChoice(point=point, information_gain_bits=entropy, outcome_probabilities=tuple(sorted(outcome_mass.items(), key=lambda item: repr(item[0]))))
    return best

def oe_first_representatives(programs, probes, evaluate):
    """WAVE 211 (HAGI_v2_final theory_ports, C540 OE-дедуп):
    keep the FIRST program per finite probe-observation vector —
    observational equivalence pruning without global equivalence
    (OEComplete.lean: дедуп не теряет решений)."""
    seen = set()
    out = []
    for program in programs:
        key = tuple(evaluate(program, x) for x in probes)
        if key not in seen:
            seen.add(key)
            out.append(program)
    return out
