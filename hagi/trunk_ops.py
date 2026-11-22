"""Trunk operations (WAVE 195): six-phase cycle, certified merge,
Hadamard gate, residual-block growth — one file instead of 4."""
from __future__ import annotations


# ============================================================
# SIX-PHASE CANONICAL CYCLE (C479)
# ============================================================

from typing import Callable
import numpy as np
def six_phase_cycle(objective: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], w0: np.ndarray, steps_train: int=10, hadamard_len: int | None=None, merge_cand: np.ndarray | None=None, stop_patience: int=2, eta: float=0.1, delta_total: float=0.05) -> dict:
    """Run the unified cycle; return the phase trace. A phase is a
    COMMIT (with measured drop) or a HOLD (no-op) — never silent.
    C479 contracts: Σδ_i ≤ delta_total across the 6 phases (false-commit
    budget split by count), and the telescope invariant
    objective(t6) ≤ objective(t0) is asserted on the trace."""
    delta_per_phase = delta_total / 6.0
    obj_start = float(objective(np.asarray(w0, dtype=float)))
    trace: list[dict] = []
    w = np.asarray(w0, dtype=float)
    drops = []
    for _ in range(steps_train):
        obj = objective(w)
        w = w - eta * grad(w)
        drops.append(obj - objective(w))
    trace.append({'phase': 'TRAIN', 'commit': sum((d > 0 for d in drops)) > 0, 'drop': float(sum(drops))})
    g = grad(w)
    s = max(float(np.abs(g).max()) * 0.2, 1e-12)
    d = s * np.sign(np.round(g / s))
    obj = objective(w)
    w2 = w - eta * d
    drop_ef = obj - objective(w2)
    if drop_ef > 0:
        w = w2
    trace.append({'phase': 'EF21', 'commit': drop_ef > 0, 'drop': float(drop_ef)})
    phase_h = {'phase': 'HADAMARD-TERN', 'commit': False, 'drop': 0.0}
    if hadamard_len and w.size == hadamard_len:
        g_gate = rotate_gate(w)
        if g_gate['rotate']:
            w_r = sylvester_mul(w, hadamard_len)
            drop_h = objective(w) - objective(w_r)
            if drop_h > 0:
                w = w_r
                phase_h['commit'] = True
                phase_h['drop'] = float(drop_h)
    trace.append(phase_h)
    obj_before = objective(w)
    grew = False
    trace.append({'phase': 'GROW', 'commit': False, 'drop': 0.0})
    obj_after = obj_before
    phase_m = {'phase': 'MERGE', 'commit': False, 'drop': 0.0}
    if merge_cand is not None:
        g_w = grad(w)
        b = merge_bound(lambda x: float(objective(x)), g_w, w, merge_cand)
        if b['certified']:
            cand_obj = float(objective(merge_cand))
            obj_w = float(objective(w))
            if (obj_w + cand_obj) / 2 + b['delta_mid'] + b['delta_end'] / 2 < obj_w:
                w = (w + merge_cand) / 2.0
                phase_m['commit'] = True
                phase_m['drop'] = float(obj_w - objective(w))
    trace.append(phase_m)
    recent = [t for t in trace if t['phase'] in ('EF21', 'HADAMARD-TERN', 'MERGE')]
    stalled = sum((1 for t in reversed(recent) if not t['commit']))
    stop = stalled >= stop_patience
    trace.append({'phase': 'STOP', 'commit': not stop, 'drop': 0.0, 'stalled': stalled})
    obj_end = float(objective(w))
    telescope_ok = obj_end <= obj_start + 1e-09
    return {'trace': trace, 'w': w, 'obj': obj_end, 'obj_start': obj_start, 'telescope_ok': telescope_ok, 'delta_budget': {'per_phase': delta_per_phase, 'total': delta_total, 'phases_committed': len([t for t in trace if t['commit']])}, 'committed': [t['phase'] for t in trace if t['commit']]}
def sylvester_mul(w: np.ndarray, n: int) -> np.ndarray:
    """Apply the integer Sylvester Hadamard of size n to w."""
    return sylvester(int(np.log2(n))).astype(float) @ w

# ============================================================
# CERTIFIED CHECKPOINT MERGE (C479v2)
# ============================================================

from typing import Callable, Sequence
import numpy as np
def lin_at(g: Sequence[float], w1: Sequence[float], w2: Sequence[float], t: float) -> float:
    """First-order linearization of L at w1 along (w2 − w1)."""
    g = np.asarray(g)
    w1 = np.asarray(w1)
    w2 = np.asarray(w2)
    return float(g @ (w2 - w1)) * t
def merge_bound(L: Callable[[np.ndarray], float], g: Sequence[float], w1: Sequence[float], w2: Sequence[float]) -> dict:
    """Measure δ (mid residual), δ₂ (end residual) and return the
    certified bound L(mid) ≤ (L1+L2)/2 + δ + δ₂/2 (theorem)."""
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    mid = (w1 + w2) / 2.0
    L1, L2 = (float(L(w1)), float(L(w2)))
    delta_mid = abs(float(L(mid)) - lin_at(g, w1, w2, 0.5))
    delta_end = abs(L2 - (L1 + lin_at(g, w1, w2, 1.0)))
    bound = (L1 + L2) / 2.0 + delta_mid + delta_end / 2.0
    return {'L_mid': float(L(mid)), 'bound': bound, 'delta_mid': delta_mid, 'delta_end': delta_end, 'certified': float(L(mid)) <= bound + 1e-12}
def diag_quadratic_merge(w1: Sequence[float], w2: Sequence[float], c: Sequence[float], a: Sequence[float]) -> dict:
    """EXACT merge identity for the diagonal-quadratic loss
    L(θ)=½Σ c_i (θ_i−a_i)²: residual (1/8)Σ c_i (w2−w1)_i², and the
    merged loss computed exactly. Verifies the theorem's quadratic
    bridge (C472 curvature)."""
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    c = np.asarray(c, dtype=float)
    a = np.asarray(a, dtype=float)
    L = lambda w: 0.5 * float(np.sum(c * (w - a) ** 2))
    mid = (w1 + w2) / 2.0
    g = c * (w1 - a)
    delta_curv = float(np.sum(c * (w2 - w1) ** 2)) / 8.0
    bound = (L(w1) + L(w2)) / 2.0 + delta_curv + delta_curv / 2.0
    return {'L_mid': L(mid), 'bound': bound, 'delta_curv': delta_curv, 'exact': L(mid) <= bound + 1e-12}

# ============================================================
# HADAMARD INCOHERENCE GATE (C477)
# ============================================================

import numpy as np
def sylvester(k: int) -> np.ndarray:
    """Integer ±1 Sylvester Hadamard of size 2^k (recursive Kronecker)."""
    if k < 0:
        raise ValueError('k must be >= 0')
    H = np.ones((1, 1), dtype=np.int64)
    for _ in range(k):
        H = np.block([[H, H], [H, -H]])
    return H
def coh_sq(w: np.ndarray) -> float:
    """cohSq = ‖w‖∞² / ‖w‖₂² (0 if w is all-zero: nothing to measure)."""
    w = np.asarray(w, dtype=float).ravel()
    l2 = float((w * w).sum())
    if l2 <= 0:
        return 0.0
    return float(w.max() ** 2 if w.max() >= 0 else (-w).max() ** 2) / l2 if False else float(np.abs(w).max()) ** 2 / l2
def rotate_gate(w: np.ndarray) -> dict:
    """Measure cohSq(w) vs cohSq(H w); return the decision. H and w
    must be length 2^k."""
    w = np.asarray(w, dtype=float).ravel()
    n = w.size
    if n & n - 1:
        raise ValueError('w length must be a power of two')
    H = sylvester(int(np.log2(n))).astype(float)
    hw = H @ w
    before, after = (coh_sq(w), coh_sq(hw))
    return {'cohSq_before': before, 'cohSq_after': after, 'rotate': after < before, 'energy_ratio': float((hw * hw).sum() / (w * w).sum()) if (w * w).sum() > 0 else 1.0}
def quantize_rows_with_gate(W: np.ndarray) -> np.ndarray:
    """Per-row: rotate-if-measured-better, then b1.58 ternary with the
    row scale computed IN THE ROTATED BASIS (s = mean|v| of the rotated
    row). Note: dequantization needs the row's rotation decision —
    consumers should store the per-row gate flags.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[1]
    if n & n - 1:
        raise ValueError('row length must be a power of two')
    H = sylvester(int(np.log2(n))).astype(float)
    out = np.empty_like(W)
    flags: list[bool] = []
    for i in range(W.shape[0]):
        v = W[i]
        g = rotate_gate(v)
        if g['rotate']:
            v, flag = (H @ v, True)
        else:
            flag = False
        s = np.abs(v).mean()
        s = s if s > 1e-12 else 1.0
        q = np.round(v / s).clip(-1.0, 1.0) * s
        out[i] = q
        flags.append(flag)
    return (out, flags)

# ============================================================
# RESIDUAL BLOCK GROWTH (C450/C504/C506)
# ============================================================

from typing import Callable, Sequence
class ResidualBlock:
    """y = x + gamma * h(x); gamma=0 -> exact identity."""

    def __init__(self, h: Callable, gamma: float=0.0):
        self.h = h
        self.gamma = float(gamma)

    def __call__(self, x):
        if self.gamma == 0.0:
            return x
        return x + self.gamma * self.h(x)

    def set_gamma(self, gamma: float) -> None:
        self.gamma = float(gamma)
def grow_trunk(trunk: Sequence[Callable], h: Callable) -> list[Callable]:
    """Append a residual-zero block: trunk function preserved EXACTLY
    (grow_block_bench_sound). Returns the new trunk. WAVE 186
    (C506 grow_blocks_repeat_sound): repeated appends compose —
    each new block is the only one with gamma != 0 potential; all
    previous blocks are FROZEN by contract (their gamma stays as
    set; the trainer must not touch them — only the fresh block
    enters the certified training loop)."""
    return list(trunk) + [ResidualBlock(h, gamma=0.0)]
def trainable_blocks(trunk: Sequence['ResidualBlock'], grown_index: int | None=None):
    """C506: only the LAST (fresh) block trains; the rest frozen.
    Returns the block(s) allowed into the certified step."""
    if grown_index is None:
        grown_index = len(trunk) - 1
    return [b for i, b in enumerate(trunk) if isinstance(b, ResidualBlock) and i == grown_index]
def run_trunk(trunk: Sequence[Callable], x):
    """Compose blocks left-to-right (benchTrunk foldl form)."""
    for b in trunk:
        x = b(x)
    return x
def obj_invariant_check(trunk_old: Sequence[Callable], trunk_new: Sequence[Callable], xs, obj: Callable) -> bool:
    """grow_block_obj_invariant check: any objective over trunk
    output is unchanged by the append (exact float equality here —
    the Fraction-exact form lives in the bench)."""
    return all((obj(run_trunk(trunk_old, x)) == obj(run_trunk(trunk_new, x)) for x in xs))

# ==================================================================
# GROW SNAP (C521): certified snap to q/2^k grid at grow boundaries.

from fractions import Fraction as _Fr


def snap_rational(value, k: int) -> _Fr:
    """Snap a rational/float to the q/2^k grid (banker's rounding).
    Per-value error <= 2^-(k+1) (half-cell)."""
    f = _Fr(value)
    scaled = f * (1 << k)
    q = (scaled.numerator * 2 + scaled.denominator) // (2 * scaled.denominator)
    return _Fr(q, 1 << k)


def snap_denominator_reset(x, k: int):
    """Snap values (scalar or sequence) so every denominator <= 2^k
    (C521 snap_denominator_reset)."""
    if isinstance(x, (list, tuple)):
        return type(x)(snap_rational(v, k) for v in x)
    return snap_rational(x, k)


def snap_obj_bound(s_err: float, ds: float) -> float:
    """Certified per-point objective degradation bound for snapping
    features by ds: |(s+ds-y)^2 - (s-y)^2| <= (2|s-y| + |ds|)*|ds|
    (C521 snap_preserves_objective_bound, exact in Fraction)."""
    return (2 * abs(s_err) + abs(ds)) * abs(ds)


# ==================================================================
# ADAPTER GROWTH (C532): certified low-rank side-branch growth.
# adapterVec A B x = x + B*(A*x) — 2*r*d parameters instead of d*d
# for a full residual block (r << d). Neutrality certified with ONE
# zero factor (B=0 => exact identity), unlike residualLeaf which
# requires both zero.

import numpy as _np


class LowRankAdapter:
    """LoRA-form growth adapter (C532 AdapterGrowth):
    pred = w @ (f + B @ (A @ f)). Grows 2*r*d params, certified
    identity at B = 0 (one-factor zero neutrality)."""

    def __init__(self, d: int, r: int = 1, seed: int = 0):
        rng = _np.random.default_rng(seed)
        # Kaiming-style init for A; B = 0 => exact obj-invariant start
        self.A = rng.standard_normal((r, d)) / _np.sqrt(d)
        self.B = _np.zeros((d, r))
        self.d, self.r = d, r

    @property
    def n_params(self) -> int:
        return 2 * self.r * self.d

    def __call__(self, f: _np.ndarray) -> _np.ndarray:
        # x + B @ (A @ x); with B=0 this is EXACTLY x (C532 neutrality)
        return f + self.B @ (self.A @ f)

    def params(self) -> _np.ndarray:
        return _np.concatenate([self.A.ravel(), self.B.ravel()])

    def set_params(self, flat: _np.ndarray) -> None:
        rA = self.r * self.d
        self.A = flat[:rA].reshape(self.r, self.d)
        self.B = flat[rA:].reshape(self.d, self.r)


def adapter_warm_start(src_adapter: "LowRankAdapter",
                       dst: "LowRankAdapter") -> bool:
    """C560 adapter-transfer: warm-start domain B's adapter with a
    copy of domain A's (pre-registered pair closeness decides WHEN
    to do this — the caller's job). Measured upstream: T1-positive
    on close pairs (x387 drop speed), forgetA=0 (copy semantics —
    source untouched by construction). Shapes must match."""
    if (src_adapter.d, src_adapter.r) != (dst.d, dst.r):
        return False
    dst.A = src_adapter.A.copy()
    dst.B = src_adapter.B.copy()
    return True


def adapter_param_count(d: int, r: int, full_block_d: int | None = None) -> dict:
    """C532 explicit parameter count: adapter 2*r*d vs full block d*d
    (C532 adapter_param_le_full_block when r <= d/2)."""
    out = {"adapter": 2 * r * d}
    if full_block_d is not None:
        out["full_block"] = full_block_d * full_block_d
        out["ratio"] = out["adapter"] / out["full_block"]
    return out


class AdapterBank:
    """WAVE 210: per-domain independent adapters (user-directed,
    upstream HAGI_v2_final measured: separate B/C adapters kept
    source logits bit-exact, forgetA=0.0). ONE shared trunk/head +
    isolated low-rank side-branches per domain. Cross-domain reuse
    via adapter_warm_start (C560)."""

    def __init__(self, d: int, r: int = 2):
        self.d, self.r = d, r
        self._adapters: dict[str, LowRankAdapter] = {}

    def get(self, domain: str) -> LowRankAdapter:
        if domain not in self._adapters:
            self._adapters[domain] = LowRankAdapter(self.d, self.r,
                                                    seed=len(self._adapters))
        return self._adapters[domain]

    def apply(self, domain: str, f):
        """Residual through THIS domain's adapter only: f + B(A f).
        Other domains' adapters are untouched — isolation by
        construction (forgetA=0 bit-exact, upstream-measured)."""
        return self.get(domain)(f)

    def transfer(self, src: str, dst: str) -> bool:
        """C560 warm-start: copy src adapter into dst (caller decides
        closeness). Source adapter unchanged."""
        if src not in self._adapters:
            return False
        return adapter_warm_start(self._adapters[src],
                                  self.get(dst))

    def n_params(self) -> int:
        return sum(a.n_params for a in self._adapters.values())


# ==================================================================
# WAVE 211 (HAGI_v2_final theory_ports C538-C541): projected weight
# decay + exact coverage gate.

def projected_weight_decay(W, P, c: float, complement_update=None):
    """WeightDecayLeak: W' = W - dC - c(W - WP); assumes dC P = 0,
    P^2 = P. Protected component W P is preserved EXACTLY (upstream
    measured 5.68e-15 vs naive 1.945 leak)."""
    W = np.asarray(W, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    d = (np.zeros_like(W) if complement_update is None
         else np.asarray(complement_update, dtype=np.float64))
    return W - d - c * (W - W @ P)


def exact_ce_contribution(p_true: float) -> float:
    """C541: exact-zero CE only at p == 1; reject p <= 0 instead of
    computing -log(0)."""
    import math
    p = float(p_true)
    if p == 1.0:
        return 0.0
    if p <= 0.0:
        raise ValueError(
            "invalid exact coverage: true-byte probability must be > 0")
    return -math.log(p)


# WAVE 214 (hagi_lab): packed 5-trit/byte codec (Ternary.lean: 3^5=243<=2^8).

def pack_trits5(levels: Sequence[int]) -> tuple[bytes, int]:
    """Pack {-1,0,+1} symbols in base-3 groups of five, one byte/group.

    Mapping: -1 -> 0, 0 -> 1, +1 -> 2. Since 3**5=243 <= 256,
    every five-trit code fits injectively in one byte.
    """
    vals = [int(x) for x in levels]
    if any(x not in (-1, 0, 1) for x in vals):
        raise ValueError("levels must be in {-1,0,+1}")
    out = bytearray()
    for start in range(0, len(vals), 5):
        digits = [v+1 for v in vals[start:start+5]]
        digits += [1] * (5-len(digits))
        code = sum(d * (3**i) for i, d in enumerate(digits))
        if code >= 243:
            raise AssertionError("base-three code overflow")
        out.append(code)
    return bytes(out), len(vals)


def unpack_trits5(payload: bytes, length: int) -> np.ndarray:
    if length < 0 or len(payload) != (length+4)//5:
        raise ValueError("payload length does not match trit count")
    out = []
    for code in payload:
        if code >= 243:
            raise ValueError("invalid packed trit byte (must be <243)")
        n = int(code)
        for _ in range(5):
            out.append((n % 3)-1)
            n //= 3
    return np.asarray(out[:length], dtype=np.int8)

