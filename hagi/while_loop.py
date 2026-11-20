"""WhileLoopIR — open iterative-algorithm discovery (Bellman consistency
& orbit fixed-point). Closes the loop-algorithm class: popcount-to-zero,
digit sums, Collatz-like walks. Analytical: no threshold brute force.

Discovery: y(x) - y(step(x)) == 1 for step-counting (Bellman), or
y(x) == y(step(x)) until the fixed point for final-state loops.
Stop threshold inferred analytically from the observed zeros/fixeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence


def _digit_sum10(x: int) -> int:
    x = abs(int(x))
    s = 0
    while x > 0:
        s += x % 10
        x //= 10
    return s


STEP_PRIMITIVES: dict[str, tuple[Callable[[int], int], str, int]] = {
    # name: (fn, python-expr, meta cost bits)
    "popcount": (lambda s: int(s).bit_count(), "int(s).bit_count()", 8),
    "digit_sum10": (_digit_sum10,
                    "sum(int(c) for c in str(abs(int(s))))", 12),
    "halve": (lambda s: int(s) // 2, "s // 2", 8),
    "drop_lsb": (lambda s: int(s) & (int(s) - 1), "s & (s - 1)", 12),
    "even_halve_odd_inc1": (
        lambda s: s // 2 if s % 2 == 0 else s + 1,
        "(s // 2) if (s & 1) == 0 else (s + 1)", 16),
    "even_halve_odd_3x1": (
        lambda s: s // 2 if s % 2 == 0 else 3 * s + 1,
        "(s // 2) if (s & 1) == 0 else (3 * s + 1)", 20),
}


@dataclass(frozen=True)
class WhileLoopBlueprint:
    step_name: str
    mode: str          # "steps" | "final"

    @property
    def meta_bits(self) -> int:
        return 2 * 8 + STEP_PRIMITIVES[self.step_name][2]


@dataclass
class WhileLoopArtifact:
    step_name: str
    stop_le: int          # loop while s > stop_le (analytically inferred)
    mode: str
    domain: frozenset     # certified support
    bits_val: int
    max_iters: int = 256

    def supports(self, x: int) -> bool:
        return int(x) in self.domain

    def eval(self, x: int) -> int:
        x = int(x)
        if x not in self.domain:
            raise ValueError("outside certified domain")
        s = x
        fn = STEP_PRIMITIVES[self.step_name][0]
        steps = 0
        while s > self.stop_le:
            s = fn(s)
            steps += 1
            if steps > self.max_iters:
                raise ValueError("loop exceeded certified iteration cap")
        return steps if self.mode == "steps" else s

    def emit_python(self, fn_name: str) -> str:
        py = STEP_PRIMITIVES[self.step_name][1]
        ret = "steps" if self.mode == "steps" else "s"
        return "\n".join([
            f"def {fn_name}(x0: int) -> int:",
            f"    _DOM = {sorted(self.domain)!r}",
            "    if int(x0) not in _DOM: raise ValueError('outside domain')",
            "    s = int(x0)",
            "    steps = 0",
            f"    while s > {self.stop_le}:",
            f"        s = {py}",
            "        steps += 1",
            f"    return {ret}",
        ])


def discover_while_loop_candidates(
    xs: Sequence[int], ys: Sequence[int]
) -> list[tuple[float, WhileLoopBlueprint]]:
    """Bellman / orbit-invariance detector: walk ONE step from each
    observed x; a family qualifies only if the relation holds for ALL
    observed transition pairs (analytical, no threshold search)."""
    obs = {int(x): int(y) for x, y in zip(xs, ys)}
    if any(v < 0 for v in obs):
        return []
    scored = []
    for name, (sfn, _, _) in STEP_PRIMITIVES.items():
        bell_ok = bell_tot = fp_ok = fp_tot = 0
        for x, y in obs.items():
            nxt = sfn(x)
            if nxt == x or nxt not in obs:
                continue
            if y > 0:
                bell_tot += 1
                if y - obs[nxt] == 1:
                    bell_ok += 1
            if x != nxt:
                fp_tot += 1
                if y == obs[nxt]:
                    fp_ok += 1
        if bell_tot >= 4 and bell_ok == bell_tot and len(set(ys)) >= 3:
            scored.append((1.0 + 0.01 * bell_tot,
                           WhileLoopBlueprint(name, "steps")))
        if fp_tot >= 4 and fp_ok == fp_tot and len(set(ys)) >= 3:
            if any(x == y for x, y in obs.items()):
                scored.append((1.0 + 0.01 * fp_tot,
                               WhileLoopBlueprint(name, "final")))
    scored.sort(key=lambda t: (-t[0], t[1].meta_bits))
    return scored


def synthesize_while_loop(
    bp: WhileLoopBlueprint,
    pairs: list[tuple[int, int]],
    domain: Sequence[int],
) -> Optional[WhileLoopArtifact]:
    """Infer stop_le analytically from the boundary between zeros/fixed
    points and the rest; then verify exactly on all pairs."""
    obs = {int(x): int(y) for x, y in pairs}
    if not obs or any(v < 0 for v in obs):
        return None
    if bp.mode == "steps":
        anchors = [x for x, y in obs.items() if y == 0]
        others = [x for x, y in obs.items() if y > 0]
    else:
        anchors = [x for x, y in obs.items() if x == y]
        others = [x for x, y in obs.items() if x != y]
    if not anchors or not others:
        cands = [0, 1]
    else:
        lo, hi = max(anchors), min(others) - 1
        if lo > hi:
            return None
        cands = list(range(lo, min(hi + 1, lo + 16)))
    for stop_le in cands:
        bits = bp.meta_bits + 2 * (max(1, stop_le + 1).bit_length() - 1) + 1
        art = WhileLoopArtifact(bp.step_name, stop_le, bp.mode,
                                frozenset(int(v) for v in domain), bits)
        try:
            if all(art.eval(x) == y for x, y in pairs):
                return art
        except (ValueError, RecursionError):
            continue
    return None
