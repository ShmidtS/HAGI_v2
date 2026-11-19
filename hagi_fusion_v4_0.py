#!/usr/bin/env python3
"""HAGI Fusion v4.0 — Autonomous Proof-Carrying Universal Temporal Skill Engine (single-file).

Core mathematical & algorithmic subsystems:
  1. GF(2) Algebraic Normal Form (ANF) learner with DegreeContracts & full-cube audit
  2. Factored GF(2) logic synthesis (Positive/Negative Davio, affine divisors,
     1-gate XOR/AND functional resubstitution) + proof-carrying netlist compaction
  3. Exact GF(2) RREF don't-care ANF annihilation + optimal algebraic state encoding
  4. Hopcroft Mealy quotient minimization + exact bisimulation verification
  5. Semantic & structural global TemporalDAG + Modular Assume-Guarantee (nvars > 12)
  6. Surgical output-cone repair + Lee-Yannakakis Adaptive Distinguishing Trees (Wp-ADT)
  7. Active Stochastic Mealy L* over PrefixTrie (longest-first batching, Rivest-Schapire
     binary-search CEX, exact sequential Binomial PAC thresholds, eval_match early exit)
  8. Epistemic SelfModel, 3-checkpoint save/restore, zero-import replay verifier, & CLI
"""
from __future__ import annotations

import argparse, functools, hashlib, heapq, itertools, json, math, random, time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_ISSUER = "HAGI_TEMPORAL_AGENT_V40"
GLOBAL_N = 24   # S0..S21 (0..21), U0..U1 (22..23)
AUDIT_CAP = 12  # <=12: full-cube audit + quotient; >12: Assume-Guarantee + compaction proof


# ---------------------------------------------------------------------------
# 1. GF(2) Polynomials, Netlist Metrics & Symbolic Verification
# ---------------------------------------------------------------------------

def xor(a: frozenset[int], b: frozenset[int]) -> frozenset[int]:
    return a.symmetric_difference(b)

def mul(a: frozenset[int], b: frozenset[int]) -> frozenset[int]:
    z: set[int] = set()
    for x in a:
        for y in b:
            t = x | y
            if t in z: z.remove(t)
            else: z.add(t)
    return frozenset(z)

def ev(p: frozenset[int], x: int) -> int:
    return sum(1 for t in p if (t & x) == t) & 1

def deg(p: frozenset[int]) -> int:
    return max((t.bit_count() for t in p), default=0)

def digest(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def binom_p_ge(m: int, k: int) -> float:
    return sum(math.comb(m, j) for j in range(k, m + 1)) / (1 << m)

@functools.lru_cache(maxsize=8)
def _binom_thresh_table(alpha_step: float, min_m: int, max_m: int) -> tuple[int, ...]:
    tbl = [max_m + 1] * (max_m + 1)
    for m_tr in range(min_m, max_m + 1):
        req = m_tr + 1
        for k in range(m_tr, m_tr // 2, -1):
            if binom_p_ge(m_tr, k) <= alpha_step: req = k
            else: break
        tbl[m_tr] = req
    return tuple(tbl)

def mobius(rows: list[tuple[int, ...]], n: int) -> tuple[frozenset[int], ...]:
    out = []
    for j in range(len(rows[0])):
        a = [int(r[j]) & 1 for r in rows]
        for bit in range(n):
            step = 1 << bit
            for x in range(1 << n):
                if x & step: a[x] ^= a[x ^ step]
        out.append(frozenset(i for i, v in enumerate(a) if v))
    return tuple(out)

def substitute(p: frozenset[int], subs: tuple[frozenset[int], ...]) -> frozenset[int]:
    out: frozenset[int] = frozenset()
    for term in p:
        prod: frozenset[int] = frozenset({0})
        for i, q in enumerate(subs):
            if (term >> i) & 1: prod = mul(prod, q)
        out = xor(out, prod)
    return out

def remap(p: frozenset[int], mapping: tuple[int, ...]) -> frozenset[int]:
    return frozenset(sum(1 << mapping[i] for i in range(len(mapping)) if (t >> i) & 1) for t in p)

def _finalize_net(n: int, gates: list[dict], outs: list[int]) -> dict:
    nw = n + len(gates); fan = [0] * nw; depth = [0] * nw
    for i, g in enumerate(gates):
        depth[n + i] = 1 + max(depth[g["a"]], depth[g["b"]])
        fan[g["a"]] += 1; fan[g["b"]] += 1
    for o in outs: fan[o] += 1
    wb = max(1, (nw - 1).bit_length()); ob = max(1, nw.bit_length())
    return {"n_inputs": n, "gates": gates, "outs": outs,
            "metrics": {"gate_count": len(gates), "depth": max((depth[o] for o in outs), default=0),
                        "max_fanout": max(fan, default=0), "program_bits": len(gates) * (4 + 2 * wb) + len(outs) * ob}}

def eval_net(net: dict, inputs: tuple[int, ...]) -> tuple[int, ...]:
    n = net["n_inputs"]
    if len(inputs) != n: raise ValueError("input arity mismatch")
    w = [int(x) & 1 for x in inputs]
    for i, g in enumerate(net["gates"]):
        a, b, m = g["a"], g["b"], g["mask"]
        if not (0 <= a < n + i and 0 <= b < n + i): raise ValueError(f"forward edge {i}")
        w.append(0 if m == 0 else (1 if m == 15 else (w[a] ^ w[b] if m == 6 else (w[a] & w[b] if m == 8 else -1))))
        if w[-1] == -1: raise ValueError(f"unsupported mask {m}")
    if any(o < 0 or o >= len(w) for o in net["outs"]): raise ValueError("bad output")
    return tuple(w[o] for o in net["outs"])

def symbolic_eval_net_general(net: dict, input_polys=None) -> tuple[frozenset[int], ...]:
    n = net["n_inputs"]
    w = [frozenset({1 << i}) for i in range(n)] if input_polys is None else list(input_polys)
    for i, g in enumerate(net["gates"]):
        a, b, m = g["a"], g["b"], g["mask"]
        if not (0 <= a < n + i and 0 <= b < n + i): raise ValueError(f"forward edge {i}")
        if m == 0: z: frozenset[int] = frozenset()
        elif m == 15: z = frozenset({0})
        elif m == 6: z = xor(w[a], w[b])
        elif m == 8: z = mul(w[a], w[b])
        else: raise ValueError(f"unsupported mask {m}")
        w.append(z)
    if any(o < 0 or o >= len(w) for o in net["outs"]): raise ValueError("bad output")
    return tuple(w[o] for o in net["outs"])

def symbolic_verify_general(net: dict, expected: tuple[frozenset[int], ...]) -> bool:
    try: return symbolic_eval_net_general(net) == expected
    except Exception: return False


# ---------------------------------------------------------------------------
# 2. Proof-Carrying Compaction, Factored GF(2) Synthesis & ANF Front-End
# ---------------------------------------------------------------------------

def compact_netlist_with_proof(net: dict) -> tuple[dict, dict]:
    n = net["n_inputs"]; val: list[int | None] = [None] * n; alias = list(range(n))
    struct: dict[tuple[int, int, int], int] = {}; new_gates: list[dict] = []
    proof_steps = [{"raw_wire": i, "rule": "INPUT", "opt_wire": i} for i in range(n)]

    def get_const(c: int, raw_w: int, rule: str) -> int:
        sig = (15 if c else 0, 0, 0)
        if sig in struct:
            w = struct[sig]; proof_steps.append({"raw_wire": raw_w, "rule": rule + "_CSE", "opt_wire": w}); return w
        w = n + len(new_gates); new_gates.append({"a": 0, "b": 0, "mask": 15 if c else 0})
        struct[sig] = w; val.append(c); proof_steps.append({"raw_wire": raw_w, "rule": rule + "_NEW", "opt_wire": w}); return w

    for idx, g in enumerate(net["gates"]):
        raw_w = n + idx; a, b, m = alias[g["a"]], alias[g["b"]], g["mask"]
        if m == 0: alias.append(get_const(0, raw_w, "CONST0")); continue
        if m == 15: alias.append(get_const(1, raw_w, "CONST1")); continue
        va, vb = val[a], val[b]
        if m == 6:
            if a == b: alias.append(get_const(0, raw_w, "XOR_SELF")); continue
            if va == 0: alias.append(b); proof_steps.append({"raw_wire": raw_w, "rule": "XOR_ZERO_L", "opt_wire": b}); continue
            if vb == 0: alias.append(a); proof_steps.append({"raw_wire": raw_w, "rule": "XOR_ZERO_R", "opt_wire": a}); continue
            if va == 1 and vb == 1: alias.append(get_const(0, raw_w, "XOR_ONE_ONE")); continue
            if b < a: a, b = b, a
        elif m == 8:
            if a == b: alias.append(a); proof_steps.append({"raw_wire": raw_w, "rule": "AND_IDEM", "opt_wire": a}); continue
            if va == 0 or vb == 0: alias.append(get_const(0, raw_w, "AND_ZERO")); continue
            if va == 1: alias.append(b); proof_steps.append({"raw_wire": raw_w, "rule": "AND_ONE_L", "opt_wire": b}); continue
            if vb == 1: alias.append(a); proof_steps.append({"raw_wire": raw_w, "rule": "AND_ONE_R", "opt_wire": a}); continue
            if b < a: a, b = b, a
        sig = (m, a, b)
        if sig in struct:
            w = struct[sig]; alias.append(w); proof_steps.append({"raw_wire": raw_w, "rule": "CSE_REUSE", "opt_wire": w})
        else:
            w = n + len(new_gates); new_gates.append({"a": a, "b": b, "mask": m})
            struct[sig] = w; val.append(None); alias.append(w)
            proof_steps.append({"raw_wire": raw_w, "rule": "EMIT_GATE", "opt_wire": w})

    outs_pre = [alias[o] for o in net["outs"]]
    used = {o for o in outs_pre if o >= n}; stack = list(used)
    while stack:
        w = stack.pop(); g = new_gates[w - n]
        for ch in (g["a"], g["b"]):
            if ch >= n and ch not in used: used.add(ch); stack.append(ch)
    remap_w = {i: i for i in range(n)}; final_gates = []
    for idx, old_w in enumerate(sorted(used)):
        g = new_gates[old_w - n]; remap_w[old_w] = n + idx
        final_gates.append({"a": remap_w[g["a"]], "b": remap_w[g["b"]], "mask": g["mask"]})
    opt_net = _finalize_net(n, final_gates, [remap_w[o] for o in outs_pre])
    return opt_net, {"proof_steps": proof_steps, "pre_dce_gates": new_gates, "pre_dce_outs": outs_pre, "dce_remap": {str(k): v for k, v in remap_w.items()}}

def compact_netlist(net: dict) -> dict:
    return compact_netlist_with_proof(net)[0]

def verify_compaction_proof(raw_net: dict, opt_net: dict, proof: dict) -> tuple[bool, str]:
    n = raw_net["n_inputs"]
    if opt_net["n_inputs"] != n or len(opt_net["outs"]) != len(raw_net["outs"]): return False, "arity"
    pre_gates, steps = proof["pre_dce_gates"], proof["proof_steps"]
    if len(steps) != n + len(raw_net["gates"]): return False, "step count"
    const_val: list[int | None] = [None] * n
    for i, g in enumerate(pre_gates):
        if not (0 <= g["a"] < n + i and 0 <= g["b"] < n + i): return False, f"pre_dce forward {i}"
        const_val.append(0 if g["mask"] == 0 else (1 if g["mask"] == 15 else (None if g["mask"] in (6, 8) else -1)))
        if const_val[-1] == -1: return False, "bad mask"
    rep = [0] * (n + len(raw_net["gates"]))
    for i in range(n):
        st = steps[i]
        if st["raw_wire"] != i or st["rule"] != "INPUT" or st["opt_wire"] != i: return False, f"input step {i}"
        rep[i] = i
    for idx, g in enumerate(raw_net["gates"]):
        rw = n + idx; st = steps[rw]; ow, rule = st["opt_wire"], st["rule"]
        if st["raw_wire"] != rw or not (0 <= ow < n + len(pre_gates)): return False, f"step bounds {rw}"
        ra, rb, m = rep[g["a"]], rep[g["b"]], g["mask"]
        if rule.startswith("CONST0"): ok = (m == 0 and const_val[ow] == 0)
        elif rule.startswith("CONST1"): ok = (m == 15 and const_val[ow] == 1)
        elif rule.startswith("XOR_SELF"): ok = (m == 6 and ra == rb and const_val[ow] == 0)
        elif rule == "XOR_ZERO_L": ok = (m == 6 and const_val[ra] == 0 and ow == rb)
        elif rule == "XOR_ZERO_R": ok = (m == 6 and const_val[rb] == 0 and ow == ra)
        elif rule.startswith("XOR_ONE_ONE"): ok = (m == 6 and const_val[ra] == 1 and const_val[rb] == 1 and const_val[ow] == 0)
        elif rule == "AND_IDEM": ok = (m == 8 and ra == rb and ow == ra)
        elif rule.startswith("AND_ZERO"): ok = (m == 8 and (const_val[ra] == 0 or const_val[rb] == 0) and const_val[ow] == 0)
        elif rule == "AND_ONE_L": ok = (m == 8 and const_val[ra] == 1 and ow == rb)
        elif rule == "AND_ONE_R": ok = (m == 8 and const_val[rb] == 1 and ow == ra)
        elif rule in ("CSE_REUSE", "EMIT_GATE"):
            ok = (ow >= n and pre_gates[ow - n]["mask"] == m and sorted((pre_gates[ow - n]["a"], pre_gates[ow - n]["b"])) == sorted((ra, rb)))
        else: ok = False
        if not ok: return False, f"{rule} {rw}"
        rep[rw] = ow
    if [rep[o] for o in raw_net["outs"]] != proof["pre_dce_outs"]: return False, "pre_dce outs"
    dce_map = {int(k): v for k, v in proof["dce_remap"].items()}
    if any(dce_map.get(i) != i for i in range(n)): return False, "dce inputs"
    for idx, og in enumerate(opt_net["gates"]):
        if not (0 <= og["a"] < n + idx and 0 <= og["b"] < n + idx): return False, f"opt forward {idx}"
        matched = [k for k, v in dce_map.items() if v == n + idx and k >= n]
        if len(matched) != 1: return False, f"dce map {idx}"
        pg = pre_gates[matched[0] - n]
        if pg["mask"] != og["mask"] or og["a"] != dce_map.get(pg["a"]) or og["b"] != dce_map.get(pg["b"]): return False, f"dce mismatch {idx}"
    if [dce_map.get(o) for o in proof["pre_dce_outs"]] != opt_net["outs"]: return False, "dce outs"
    return True, "verified"

@functools.lru_cache(maxsize=512)
def compile_factored_gf2(n: int, outputs: tuple[frozenset[int], ...]) -> dict:
    """Multi-output Davio + Affine Divisor + 1-gate Functional Resubstitution over GF(2)."""
    gates: list[dict] = []
    poly_wire: dict[frozenset[int], int] = {frozenset({1 << i}): i for i in range(n)}
    wire_poly: list[frozenset[int]] = [frozenset({1 << i}) for i in range(n)]

    def emit(a: int, b: int, mask: int, poly: frozenset[int]) -> int:
        if poly in poly_wire: return poly_wire[poly]
        if mask in (6, 8) and b < a: a, b = b, a
        w = n + len(gates); gates.append({"a": a, "b": b, "mask": mask})
        poly_wire[poly] = w; wire_poly.append(poly)
        return w

    def get_const(c: int) -> int:
        p = frozenset({0}) if c else frozenset()
        return poly_wire[p] if p in poly_wire else emit(0, 0, 15 if c else 0, p)

    def try_resub(p: frozenset[int]) -> int | None:
        if p in poly_wire: return poly_wire[p]
        for pa, wa in poly_wire.items():
            rem = xor(p, pa)
            if rem in poly_wire: return emit(wa, poly_wire[rem], 6, p)
        if len(wire_poly) <= 32:
            items = [(pa, wa, deg(pa)) for pa, wa in poly_wire.items()]
            dp = deg(p)
            for i in range(len(items)):
                pa, wa, da = items[i]
                if da == 0: continue
                for j in range(i + 1, len(items)):
                    pb, wb, db = items[j]
                    if db == 0 or (da == 1 and db == 1 and len(pa) == 1 and len(pb) == 1): continue
                    if da + db >= dp and mul(pa, pb) == p: return emit(wa, wb, 8, p)
        return None

    def build_xor_list(items: list[tuple[frozenset[int], int]]) -> int:
        if not items: return get_const(0)
        while len(items) > 1:
            best = None
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    xp = xor(items[i][0], items[j][0])
                    if xp in poly_wire: best = (i, j, xp, poly_wire[xp]); break
                if best: break
            if best:
                i, j, xp, xw = best; items.pop(j); items.pop(i); items.append((xp, xw))
            else:
                (p1, w1), (p2, w2) = items.pop(0), items.pop(0)
                xp = xor(p1, p2)
                items.append((xp, emit(w1, w2, 6, xp) if w1 != w2 else get_const(0)))
        return items[0][1]

    def build_poly(p: frozenset[int]) -> int:
        if p in poly_wire: return poly_wire[p]
        if not p: return get_const(0)
        if p == frozenset({0}): return get_const(1)
        rs = try_resub(p)
        if rs is not None: return rs
        if max(t.bit_count() for t in p) <= 1:
            return build_xor_list([(frozenset({0}), get_const(1)) if t == 0 else (frozenset({t}), poly_wire[frozenset({t})]) for t in sorted(p)])
        if len(p) == 1:
            t = next(iter(p)); bits = [1 << i for i in range(n) if (t >> i) & 1]
            cur = [(frozenset({b}), poly_wire[frozenset({b})]) for b in bits]
            while len(cur) > 1:
                found = None
                for i in range(len(cur)):
                    for j in range(i + 1, len(cur)):
                        mp = frozenset({next(iter(cur[i][0])) | next(iter(cur[j][0]))})
                        if mp in poly_wire: found = (i, j, mp, poly_wire[mp]); break
                    if found: break
                if found:
                    i, j, mp, mw = found; cur.pop(j); cur.pop(i); cur.append((mp, mw))
                else:
                    p1, w1 = cur.pop(0); p2, w2 = cur.pop(0)
                    mp = frozenset({next(iter(p1)) | next(iter(p2))})
                    cur.append((mp, emit(w1, w2, 8, mp)))
            return cur[0][1]

        cm = (1 << n) - 1
        for t in p: cm &= t
        if cm:
            b = cm & -cm
            return emit(poly_wire[frozenset({b})], build_poly(frozenset(t ^ b for t in p)), 8, p)
        for v in range(n):
            vb = 1 << v
            p0 = frozenset(t for t in p if not (t & vb))
            p1 = frozenset(t ^ vb for t in p if (t & vb))
            if p0 == p1 and p1: return emit(build_poly(frozenset({0, vb})), build_poly(p0), 8, p)
        for a in range(n):
            for b in range(a + 1, n):
                ab, bb = 1 << a, 1 << b
                sub_same: set[int] = set(); sub_inv: set[int] = set()
                for t in p:
                    nt = (t ^ ab) | bb if (t & ab) else t
                    if nt in sub_same: sub_same.remove(nt)
                    else: sub_same.add(nt)
                    if t & ab:
                        for nti in (t ^ ab, (t ^ ab) | bb):
                            if nti in sub_inv: sub_inv.remove(nti)
                            else: sub_inv.add(nti)
                    else:
                        if t in sub_inv: sub_inv.remove(t)
                        else: sub_inv.add(t)
                if not sub_same:
                    return emit(build_poly(frozenset({ab, bb})), build_poly(frozenset(t ^ ab for t in p if (t & ab))), 8, p)
                if not sub_inv:
                    return emit(build_poly(frozenset({0, ab, bb})), build_poly(frozenset(t ^ ab for t in p if (t & ab))), 8, p)

        if len(p) >= 3 and len(wire_poly) <= 64:
            for pa, wa in list(poly_wire.items()):
                if len(pa) <= 1 and pa != frozenset({0}): continue
                rem = xor(p, pa)
                if len(rem) <= 2 and len(rem) < len(p) - 1: return emit(wa, build_poly(rem), 6, p)

        nl_vars = sorted({i for t in p if t.bit_count() >= 2 for i in range(n) if (t >> i) & 1}) or [0]
        best_var = best_kind = best_score = None
        def comp_cost(q: frozenset[int]) -> int:
            if not q or q == frozenset({0}) or q in poly_wire: return 0
            if max(t.bit_count() for t in q) <= 1: return 1
            return len(q) + sum(t.bit_count() for t in q if t.bit_count() >= 2)
        for v in nl_vars:
            vb = 1 << v
            p0 = frozenset(t for t in p if not (t & vb))
            p1 = frozenset(t ^ vb for t in p if (t & vb))
            p_neg = xor(p0, p1)
            sc_pos = (comp_cost(p0) + comp_cost(p1), len(p0) + len(p1), v)
            if best_score is None or sc_pos < best_score: best_score, best_var, best_kind = sc_pos, v, "POS"
            c_notv = 0 if frozenset({0, vb}) in poly_wire else 1
            sc_neg = (comp_cost(p_neg) + comp_cost(p1) + c_notv, len(p_neg) + len(p1), v)
            if sc_neg < best_score: best_score, best_var, best_kind = sc_neg, v, "NEG"
        vb = 1 << best_var
        p0 = frozenset(t for t in p if not (t & vb))
        p1 = frozenset(t ^ vb for t in p if (t & vb))
        if best_kind == "POS":
            w_prod = emit(poly_wire[frozenset({vb})], build_poly(p1), 8, frozenset(t | vb for t in p1))
            return w_prod if not p0 else emit(build_poly(p0), w_prod, 6, p)
        p_neg = xor(p0, p1)
        w_prod = emit(build_poly(frozenset({0, vb})), build_poly(p1), 8, mul(frozenset({0, vb}), p1))
        return w_prod if not p_neg else emit(build_poly(p_neg), w_prod, 6, p)

    order = sorted(range(len(outputs)), key=lambda idx: (deg(outputs[idx]), len(outputs[idx]), idx))
    outs_map = {idx: build_poly(outputs[idx]) for idx in order}
    return compact_netlist({"n_inputs": n, "gates": gates, "outs": [outs_map[i] for i in range(len(outputs))]})

@dataclass(frozen=True)
class DegreeContract:
    n_inputs: int; max_anf_degree: int; issuer: str; version: int = 1
    def validate(self, trusted_issuers: set[str]) -> tuple[bool, str]:
        if self.n_inputs < 1: return False, "n_inputs must be positive"
        if not 0 <= self.max_anf_degree <= self.n_inputs: return False, "degree bound outside [0,n]"
        if self.issuer not in trusted_issuers: return False, "untrusted degree-contract issuer"
        return True, "trusted contract"

@dataclass
class ANFCandidate:
    n_inputs: int; outputs: tuple[frozenset[int], ...]; netlist: dict; query_count: int; mode: str; certificate: dict; metrics: dict
    def semantic_digest(self) -> str:
        return hashlib.sha256(json.dumps({"n": self.n_inputs, "outputs": [sorted(x) for x in self.outputs]}, sort_keys=True).encode()).hexdigest()

class ProofCarryingANFLearner:
    def __init__(self, trusted_issuers: Iterable[str] = (), max_full_support_inputs: int = 14):
        self.trusted_issuers = set(trusted_issuers); self.max_full_support_inputs = max_full_support_inputs

    @staticmethod
    def subsets_upto(n: int, d: int) -> list[int]:
        return [sum(1 << i for i in c) for k in range(d + 1) for c in itertools.combinations(range(n), k)]

    @staticmethod
    def _interpolate(n: int, d: int, oracle: Callable[[int], tuple[int, ...]], n_outputs: int):
        terms = [set() for _ in range(n_outputs)]; labels = []
        for s in ProofCarryingANFLearner.subsets_upto(n, d):
            y = tuple(int(v) & 1 for v in oracle(s)); labels.append((s, *y))
            t = (s - 1) & s if s else -1; proper = []
            while t >= 0:
                proper.append(t)
                if t == 0: break
                t = (t - 1) & s
            for o in range(n_outputs):
                v = y[o]
                for sub in proper:
                    if sub in terms[o]: v ^= 1
                if v: terms[o].add(s)
        return tuple(frozenset(t) for t in terms), labels

    @staticmethod
    def _full_mobius(n: int, oracle: Callable[[int], tuple[int, ...]], n_outputs: int):
        N = 1 << n; observed = [tuple(int(v) & 1 for v in oracle(x)) for x in range(N)]
        if any(len(row) != n_outputs for row in observed): raise ValueError("oracle output arity mismatch")
        return mobius(observed, n), N

    @staticmethod
    def compile(n: int, outputs: tuple[frozenset[int], ...]) -> dict:
        gates: list[dict] = []; depth = [0] * n; cache: dict[int, int] = {}; const1 = const0 = None
        def add(a: int, b: int, mask: int) -> int:
            w = n + len(gates); gates.append({"a": a, "b": b, "mask": mask}); depth.append(1 + max(depth[a], depth[b])); return w
        def monomial(s: int) -> int:
            nonlocal const1
            if s in cache: return cache[s]
            if s == 0:
                if const1 is None: const1 = add(0, 0, 15)
                cache[s] = const1; return const1
            if s & (s - 1) == 0: w = s.bit_length() - 1; cache[s] = w; return w
            bits = [1 << i for i in range(n) if (s >> i) & 1]; mid = len(bits) // 2
            w = add(monomial(sum(bits[:mid])), monomial(sum(bits[mid:])), 8); cache[s] = w; return w
        outs = []
        for out_terms in outputs:
            leaves = [monomial(s) for s in sorted(out_terms)]
            if not leaves:
                if const0 is None: const0 = add(0, 0, 0)
                outs.append(const0)
            elif len(leaves) == 1: outs.append(leaves[0])
            else:
                heap = [(depth[w], w) for w in leaves]; heapq.heapify(heap)
                while len(heap) > 1:
                    da, a = heapq.heappop(heap); db, b = heapq.heappop(heap)
                    heapq.heappush(heap, (1 + max(da, db), add(a, b, 6)))
                outs.append(heap[0][1])
        return _finalize_net(n, gates, outs)

    @staticmethod
    def compile_optimal(n: int, outputs: tuple[frozenset[int], ...]) -> dict:
        sop = compact_netlist(ProofCarryingANFLearner.compile(n, outputs))
        if n <= 12:
            fac = compile_factored_gf2(n, outputs)
            if symbolic_verify_general(fac, outputs):
                return min([sop, fac], key=lambda nt: (len(nt["gates"]), nt["metrics"]["depth"], nt["metrics"]["program_bits"]))
        return sop

    @staticmethod
    def symbolic_verify(netlist: dict, expected: tuple[frozenset[int], ...]) -> bool:
        return symbolic_verify_general(netlist, expected)

    def fit(self, n_inputs: int, n_outputs: int, oracle: Callable[[int], tuple[int, ...]],
            contract: DegreeContract | None = None, allow_full_support: bool = True,
            audit_full_support: bool = False) -> tuple[ANFCandidate | None, dict]:
        if n_inputs < 1 or n_outputs < 1: raise ValueError("input/output arity must be positive")
        if contract is not None:
            valid, reason = contract.validate(self.trusted_issuers)
            if valid and contract.n_inputs == n_inputs:
                d = contract.max_anf_degree
                outputs, labels = self._interpolate(n_inputs, d, oracle, n_outputs)
                net = self.compile_optimal(n_inputs, outputs); basis = [x[0] for x in labels]
                cert = {"kind": "trusted-degree-unisolvent", "issuer": contract.issuer, "contract_version": contract.version,
                        "max_anf_degree": d, "basis_size": len(basis),
                        "basis_points_sha256": hashlib.sha256(",".join(map(str, basis)).encode()).hexdigest(),
                        "labels_sha256": hashlib.sha256(bytes(v for row in labels for v in row[1:])).hexdigest(), "unit_triangular": True}
                if not self.symbolic_verify(net, outputs): return None, {"action": "HOLD", "reason": "symbolic netlist verification failed"}
                audit_checks = 0
                if audit_full_support and n_inputs <= self.max_full_support_inputs:
                    mismatch = False
                    for x in range(1 << n_inputs):
                        if tuple(ev(terms, x) for terms in outputs) != tuple(int(v) & 1 for v in oracle(x)):
                            mismatch = True; break
                        audit_checks += 1
                    if mismatch:
                        if not allow_full_support: return None, {"action": "HOLD", "reason": "degree contract failed full-support audit", "audit_checks": audit_checks}
                        full_outputs, q = self._full_mobius(n_inputs, oracle, n_outputs)
                        full_net = self.compile_optimal(n_inputs, full_outputs)
                        if not self.symbolic_verify(full_net, full_outputs): return None, {"action": "HOLD", "reason": "fallback symbolic verification failed"}
                        return ANFCandidate(n_inputs, full_outputs, full_net, q, "full-support-escalated",
                                            {"kind": "full-support-mobius", "full_support": True, "queries": q}, full_net["metrics"]), \
                               {"action": "COMMIT_CANDIDATE", "mode": "full-support-escalated", "query_count": q, "support_size": q, "degree_contract_rejected": True}
                cert["audit_full_support"] = bool(audit_full_support and n_inputs <= self.max_full_support_inputs)
                cert["audit_rows_checked"] = audit_checks
                return ANFCandidate(n_inputs, outputs, net, len(labels), "degree-contract", cert, net["metrics"]), \
                       {"action": "COMMIT_CANDIDATE", "mode": "degree-contract", "query_count": len(labels), "support_size": 1 << n_inputs, "audit_rows_checked": audit_checks}
            reason = reason if not valid else "contract input arity mismatch"
        else:
            reason = "no degree contract"
        if not allow_full_support or n_inputs > self.max_full_support_inputs:
            return None, {"action": "HOLD", "reason": reason + "; exact full-support fallback unavailable"}
        outputs, q = self._full_mobius(n_inputs, oracle, n_outputs)
        net = self.compile_optimal(n_inputs, outputs)
        if not self.symbolic_verify(net, outputs): return None, {"action": "HOLD", "reason": "full-support symbolic netlist verification failed"}
        return ANFCandidate(n_inputs, outputs, net, q, "full-support", {"kind": "full-support-mobius", "full_support": True, "queries": q}, net["metrics"]), \
               {"action": "COMMIT_CANDIDATE", "mode": "full-support", "query_count": q, "support_size": q}

def surgical_repair_netlist(corrupted_net: dict, expected_anf: tuple[frozenset[int], ...],
                            oracle_rows: list[tuple[int, ...]], learner: ProofCarryingANFLearner,
                            issuer: str = DEFAULT_ISSUER) -> tuple[dict, list[int], int]:
    n = corrupted_net["n_inputs"]; got_anf = symbolic_eval_net_general(corrupted_net)
    bad = [idx for idx, (g, e) in enumerate(zip(got_anf, expected_anf)) if g != e]
    if not bad: return corrupted_net, [], 0
    cand, dec = learner.fit(n, len(bad), lambda x: tuple(oracle_rows[x][idx] for idx in bad),
                            contract=DegreeContract(n, max(deg(expected_anf[idx]) for idx in bad), issuer, 2),
                            allow_full_support=True, audit_full_support=False)
    if cand is None: raise RuntimeError(f"surgical repair failed: {dec}")
    new_gates = [dict(g) for g in corrupted_net["gates"]]; new_outs = list(corrupted_net["outs"])
    offset = n + len(new_gates); rep_net = cand.netlist
    for g in rep_net["gates"]:
        new_gates.append({"a": g["a"] if g["a"] < n else g["a"] - n + offset,
                          "b": g["b"] if g["b"] < n else g["b"] - n + offset, "mask": g["mask"]})
    for pos, out_idx in enumerate(bad):
        ro = rep_net["outs"][pos]; new_outs[out_idx] = ro if ro < n else ro - n + offset
    repaired = compact_netlist({"n_inputs": n, "gates": new_gates, "outs": new_outs})
    if not symbolic_verify_general(repaired, expected_anf): raise RuntimeError("surgical repair verification failed")
    return repaired, bad, cand.query_count


# ---------------------------------------------------------------------------
# 3. Hopcroft Quotient Minimization, GF(2) Don't-Care RREF & State Encoding
# ---------------------------------------------------------------------------

def quotient_to_rows(k: int, ni: int, no: int, T) -> tuple[int, list[tuple[int, ...]]]:
    ns_min = 0 if k <= 1 else (k - 1).bit_length(); rows = []
    for x in range(1 << (ns_min + ni)):
        c = x & ((1 << ns_min) - 1) if ns_min > 0 else 0; u = x >> ns_min
        nxt_c, out_u = T[c][u] if c < k else (0, (0,) * no)
        rows.append(tuple([(nxt_c >> i) & 1 for i in range(ns_min)] + list(out_u)))
    return ns_min, rows

def minimize_quotient_table(k_raw: int, ni: int, no: int, raw_T, reset: int = 0):
    seen = {reset}; q = deque([reset])
    while q:
        s = q.popleft()
        for u in range(1 << ni):
            nxt, _ = raw_T[s][u]
            if nxt not in seen: seen.add(nxt); q.append(nxt)
    reach = sorted(seen); cls: dict[int, int] = {}; sigs: dict = {}
    for s in reach: cls[s] = sigs.setdefault(tuple(raw_T[s][u][1] for u in range(1 << ni)), len(sigs))
    while True:
        nsigs: dict = {}; ncls: dict[int, int] = {}
        for s in reach: ncls[s] = nsigs.setdefault((cls[s], tuple(cls[raw_T[s][u][0]] for u in range(1 << ni))), len(nsigs))
        if len(nsigs) == len(sigs): break
        cls, sigs = ncls, nsigs
    canon_id = {cls[reset]: 0}; rep = {0: reset}; bq = deque([reset])
    while bq:
        s = bq.popleft()
        for u in range(1 << ni):
            nxt, _ = raw_T[s][u]; nc = cls[nxt]
            if nc not in canon_id: cid = len(canon_id); canon_id[nc] = cid; rep[cid] = nxt; bq.append(nxt)
    k = len(canon_id); phi = {s: canon_id[cls[s]] for s in reach}
    T = tuple(tuple((phi[raw_T[rep[c]][u][0]], raw_T[rep[c]][u][1]) for u in range(1 << ni)) for c in range(k))
    return k, digest({"ni": ni, "no": no, "k": k, "T": [[list(x[1]) + [x[0]] for x in row] for row in T]}), T, phi, reach

def minimize_mealy_from_rows(ns: int, ni: int, no: int, rows: list[tuple[int, ...]], reset: int = 0):
    raw_T = tuple(tuple((sum(v << i for i, v in enumerate(rows[s | (u << ns)][:ns])), tuple(rows[s | (u << ns)][ns:])) for u in range(1 << ni)) for s in range(1 << ns))
    k, bdig, T, phi, reach = minimize_quotient_table(1 << ns, ni, no, raw_T, reset)
    ns_min, min_rows = quotient_to_rows(k, ni, no, T)
    for s in reach:
        c = phi[s]
        for u in range(1 << ni):
            ro = rows[s | (u << ns)]; rm = min_rows[c | (u << ns_min)]
            if tuple(ro[ns:]) != tuple(rm[ns_min:]) or phi[sum(v << i for i, v in enumerate(ro[:ns]))] != sum(v << i for i, v in enumerate(rm[:ns_min])):
                raise AssertionError("bisimulation verification failed")
    return ns_min, k, bdig, T, min_rows, len(reach)

def minimize_dont_care_rows_f2(ns: int, ni: int, no: int, rows: list[tuple[int, ...]], care_states: set[int]):
    """Exact GF(2) RREF Don't-Care ANF Degree & Monomial Annihilation on unreachable state codes."""
    n = ns + ni; N = 1 << n; ncols = ns + no
    dc_xs = [s | (u << ns) for s in range(1 << ns) if s not in care_states for u in range(1 << ni)]
    anf_base = mobius(rows, n)
    if not dc_xs or n > 10: return rows, anf_base
    monomials = sorted(range(N), key=lambda t: (t.bit_count(), t), reverse=True)
    mono_pos = {t: idx for idx, t in enumerate(monomials)}; pivots = []
    for idx_d, d in enumerate(dc_xs):
        vec = 0
        for pos, t in enumerate(monomials):
            if (d & t) == d: vec |= 1 << (N - 1 - pos)
        dc_combo = 1 << idx_d; cur = vec
        for p_bit, p_vec, p_combo in pivots:
            if (cur >> p_bit) & 1: cur ^= p_vec; dc_combo ^= p_combo
        if cur: pivots.append((cur.bit_length() - 1, cur, dc_combo)); pivots.sort(key=lambda x: -x[0])
    new_rows = [list(r) for r in rows]
    for j in range(ncols):
        avec = 0
        for t in anf_base[j]: avec |= 1 << (N - 1 - mono_pos[t])
        dc_used = 0
        for p_bit, p_vec, p_combo in pivots:
            if (avec >> p_bit) & 1: avec ^= p_vec; dc_used ^= p_combo
        for idx_d, d in enumerate(dc_xs):
            if (dc_used >> idx_d) & 1: new_rows[d][j] ^= 1
    fin_rows = [tuple(r) for r in new_rows]
    return fin_rows, mobius(fin_rows, n)

@functools.lru_cache(maxsize=256)
def optimal_encode_quotient(k: int, ni: int, no: int, T) -> tuple[int, list[tuple[int, ...]], tuple[frozenset[int], ...], dict]:
    """Algebraic state assignment pi: {0..k-1}->{0..2^ns-1} (pi(0)=0) + GF(2) RREF + factored netlist."""
    ns = 0 if k <= 1 else (k - 1).bit_length()
    if k <= 1 or ns + ni > 10:
        ns_m, rows = quotient_to_rows(k, ni, no, T); anf = mobius(rows, ns_m + ni)
        return ns_m, rows, anf, ProofCarryingANFLearner.compile_optimal(ns_m + ni, anf)
    has_dc = k < (1 << ns)

    def eval_perm(perm_codes: tuple[int, ...], refine_dc: bool = False):
        rows = [(0,) * (ns + no)] * (1 << (ns + ni)); care = set(perm_codes)
        for c in range(k):
            sc = perm_codes[c]
            for u in range(1 << ni):
                nc, out_u = T[c][u]; nsc = perm_codes[nc]
                rows[sc | (u << ns)] = tuple([(nsc >> i) & 1 for i in range(ns)] + list(out_u))
        opt_rows, opt_anf = minimize_dont_care_rows_f2(ns, ni, no, rows, care) if (has_dc and refine_dc) else (rows, mobius(rows, ns + ni))
        return (sum(len(p) for p in opt_anf), sum(deg(p) for p in opt_anf)), opt_rows, opt_anf

    if k <= 4 and ns <= 2:
        top = sorted([(eval_perm((0,) + p, True), (0,) + p) for p in itertools.permutations(range(1, 1 << ns), k - 1)], key=lambda x: x[0][0])[:3]
        cands = [(sc, codes, r, a) for (sc, r, a), codes in top]
    else:
        gray_seq = [i ^ (i >> 1) for i in range(1 << ns)]
        seeds = [tuple(range(k)), tuple([0] + [g for g in gray_seq if g != 0][:k - 1])]
        best_cands = []
        for init_codes in seeds:
            cur = list(init_codes); cur_sc, cur_r, cur_a = eval_perm(tuple(cur), True)
            best_cands.append((cur_sc, tuple(cur), cur_r, cur_a))
            if k <= 12:
                used = set(cur); unused = [c for c in range(1, 1 << ns) if c not in used]
                for i in range(1, k):
                    for j in range(i + 1, k):
                        cur[i], cur[j] = cur[j], cur[i]; sc, r, a = eval_perm(tuple(cur), False)
                        if sc < cur_sc: cur_sc, cur_r, cur_a = sc, r, a; best_cands.append((cur_sc, tuple(cur), cur_r, cur_a))
                        else: cur[i], cur[j] = cur[j], cur[i]
                    for idx_u, uc in enumerate(list(unused)):
                        old = cur[i]; cur[i] = uc; sc, r, a = eval_perm(tuple(cur), False)
                        if sc < cur_sc: cur_sc, cur_r, cur_a = sc, r, a; unused[idx_u] = old; best_cands.append((cur_sc, tuple(cur), cur_r, cur_a))
                        else: cur[i] = old
        best_cands.sort(key=lambda x: x[0]); cands = []; seen_c = set()
        for item in best_cands:
            if item[1] not in seen_c:
                seen_c.add(item[1]); sc, r, a = eval_perm(item[1], True); cands.append((sc, item[1], r, a))
            if len(cands) >= 3: break

    best_net = best_rows = best_anf = None
    for _, _, r, a in cands:
        net = ProofCarryingANFLearner.compile_optimal(ns + ni, a)
        if best_net is None or (len(net["gates"]), net["metrics"]["depth"], net["metrics"]["program_bits"]) < (len(best_net["gates"]), best_net["metrics"]["depth"], best_net["metrics"]["program_bits"]):
            best_net, best_rows, best_anf = net, r, a
    return ns, best_rows, best_anf, best_net


# ---------------------------------------------------------------------------
# 4. Skill, Typed Composition Algebra, Semantic TemporalDAG, SelfModel, Library
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    name: str; ns: int; ni: int; no: int; reset: int
    rows: list[tuple[int, ...]] | None; anf: tuple[frozenset[int], ...] | None; net: dict; route: str; op: str
    parents: tuple[str, ...] = (); queries: int = 0; compile_mode: str = "FLAT"; k_states: int = 1; ns_min: int = 0
    behav_digest: str = ""; quotient_T: tuple | None = None; bisim_verified: bool = True; ag_cert: dict | None = None; generation: int = 0

    def step(self, s: int, u: int, network: bool = False) -> tuple[int, tuple[int, ...]]:
        x = (s & ((1 << self.ns) - 1)) | ((u & ((1 << self.ni) - 1)) << self.ns)
        y = eval_net(self.net, tuple((x >> i) & 1 for i in range(self.ns + self.ni))) if (network or self.rows is None) else self.rows[x]
        return sum(v << i for i, v in enumerate(y[:self.ns])), tuple(y[self.ns:])

    @property
    def nvars(self) -> int: return self.ns + self.ni

    def semantic_digest(self) -> str:
        if self.anf is not None: return digest({"ns": self.ns, "ni": self.ni, "no": self.no, "anf": [sorted(p) for p in self.anf]})
        return digest({"ns": self.ns, "ni": self.ni, "no": self.no, "ag_net": self.net})

    def export(self) -> dict:
        return {"name": self.name, "ns": self.ns, "ni": self.ni, "no": self.no, "reset": self.reset,
                "rows": [list(r) for r in self.rows] if self.rows is not None else None,
                "anf": [sorted(p) for p in self.anf] if self.anf is not None else None,
                "net": self.net, "route": self.route, "op": self.op, "parents": list(self.parents),
                "queries": self.queries, "compile_mode": self.compile_mode, "k_states": self.k_states, "ns_min": self.ns_min,
                "behav_digest": self.behav_digest,
                "quotient_T": [[[x[0], list(x[1])] for x in row] for row in self.quotient_T] if self.quotient_T is not None else None,
                "bisim_verified": self.bisim_verified, "ag_cert": self.ag_cert, "generation": self.generation, "digest": self.semantic_digest()}

def lut4_anf(lut4: int) -> frozenset[int]: return mobius([((lut4 >> x) & 1,) for x in range(4)], 2)[0]
def lut4_name(lut4: int) -> str:
    return f"LUT_{ {0:'CONST0',1:'NOR',2:'NOT_A_AND_B',4:'A_AND_NOT_B',6:'XOR',8:'AND',9:'XNOR',11:'B_IMPLIES_A',13:'A_IMPLIES_B',14:'OR',15:'CONST1'}.get(lut4, f'{lut4:02d}') }"
def lut4_eval(lut4: int, ya: int, yb: int) -> int: return (lut4 >> (ya | (yb << 1))) & 1

def _emit_subcircuit(net: dict, wire_map: list[int], n_total: int, gates: list[dict]) -> list[int]:
    w = list(wire_map)
    for g in net["gates"]:
        nw = n_total + len(gates); gates.append({"a": w[g["a"]], "b": w[g["b"]], "mask": g["mask"]}); w.append(nw)
    return [w[o] for o in net["outs"]]

def compose_net_cascade(net_a: dict, ns_a: int, ni_a: int, net_b: dict, ns_b: int, ni_b: int) -> dict:
    n = ns_a + ns_b + ni_a; gates: list[dict] = []
    oa = _emit_subcircuit(net_a, list(range(ns_a)) + list(range(ns_a + ns_b, n)), n, gates)
    ob = _emit_subcircuit(net_b, list(range(ns_a, ns_a + ns_b)) + oa[ns_a:], n, gates)
    return {"n_inputs": n, "gates": gates, "outs": oa[:ns_a] + ob}

def compose_net_parallel_pair(net_a: dict, ns1: int, net_b: dict, ns2: int, ni: int) -> dict:
    n = ns1 + ns2 + ni; gates: list[dict] = []
    oa = _emit_subcircuit(net_a, list(range(ns1)) + list(range(ns1 + ns2, n)), n, gates)
    ob = _emit_subcircuit(net_b, list(range(ns1, ns1 + ns2)) + list(range(ns1 + ns2, n)), n, gates)
    return {"n_inputs": n, "gates": gates, "outs": oa[:ns1] + ob[:ns2] + oa[ns1:] + ob[ns2:]}

def compose_net_parallel_lut(net_a: dict, ns1: int, net_b: dict, ns2: int, ni: int, lut4: int) -> dict:
    n = ns1 + ns2 + ni; gates: list[dict] = []
    oa = _emit_subcircuit(net_a, list(range(ns1)) + list(range(ns1 + ns2, n)), n, gates)
    ob = _emit_subcircuit(net_b, list(range(ns1, ns1 + ns2)) + list(range(ns1 + ns2, n)), n, gates)
    oh = _emit_subcircuit(ProofCarryingANFLearner.compile(2, (lut4_anf(lut4),)), [oa[ns1], ob[ns2]], n, gates)
    return {"n_inputs": n, "gates": gates, "outs": oa[:ns1] + ob[:ns2] + oh}

def compose_net_mux_2in(net_a: dict, ns1: int, net_b: dict, ns2: int) -> dict:
    n = ns1 + ns2 + 2; u0, u1 = ns1 + ns2, ns1 + ns2 + 1; gates: list[dict] = []
    oa = _emit_subcircuit(net_a, list(range(ns1)) + [u0], n, gates)
    ob = _emit_subcircuit(net_b, list(range(ns1, ns1 + ns2)) + [u0], n, gates)
    ya, yb = oa[ns1], ob[ns2]
    gd = n + len(gates); gates.append({"a": ya, "b": yb, "mask": 6})
    gs = n + len(gates); gates.append({"a": u1, "b": gd, "mask": 8})
    gy = n + len(gates); gates.append({"a": ya, "b": gs, "mask": 6})
    return {"n_inputs": n, "gates": gates, "outs": oa[:ns1] + ob[:ns2] + [gy]}

def compose_net_moe_mux(net_s: dict, ns_s: int, net_a: dict, ns_a: int, net_b: dict, ns_b: int, ni: int) -> dict:
    n = ns_s + ns_a + ns_b + ni; off_u = ns_s + ns_a + ns_b; gates: list[dict] = []; u_wires = list(range(off_u, n))
    os_ = _emit_subcircuit(net_s, list(range(ns_s)) + u_wires, n, gates)
    oa = _emit_subcircuit(net_a, list(range(ns_s, ns_s + ns_a)) + u_wires, n, gates)
    ob = _emit_subcircuit(net_b, list(range(ns_s + ns_a, off_u)) + u_wires, n, gates)
    ys, ya, yb = os_[-1], oa[-1], ob[-1]
    gd = n + len(gates); gates.append({"a": ya, "b": yb, "mask": 6})
    gs = n + len(gates); gates.append({"a": ys, "b": gd, "mask": 8})
    gy = n + len(gates); gates.append({"a": ya, "b": gs, "mask": 6})
    return {"n_inputs": n, "gates": gates, "outs": os_[:-1] + oa[:-1] + ob[:-1] + [gy]}

def compose_net_delayed_feedback(net_m: dict, ns: int) -> dict:
    n = ns + 2; gates: list[dict] = []; mo = _emit_subcircuit(net_m, list(range(ns)) + [ns + 1, ns], n, gates)
    return {"n_inputs": n, "gates": gates, "outs": mo[:ns] + [mo[ns], mo[ns]]}

def compose_net_coupled_fb_2bit(net_a: dict, ns1: int, net_b: dict, ns2: int) -> dict:
    da, db, u0 = ns1 + ns2, ns1 + ns2 + 1, ns1 + ns2 + 2; n = u0 + 1; gates: list[dict] = []
    oa = _emit_subcircuit(net_a, list(range(ns1)) + [u0, db], n, gates)
    ob = _emit_subcircuit(net_b, list(range(ns1, ns1 + ns2)) + [u0, da], n, gates)
    return {"n_inputs": n, "gates": gates, "outs": oa[:ns1] + ob[:ns2] + [oa[ns1], ob[ns2], ob[ns2]]}

def verify_ag_raw(op: str, raw_net: dict, pa: Skill, pb: Skill | None = None, pc: Skill | None = None, lut4: int = 0) -> tuple[bool, str]:
    if op == "CASCADE": expected, ok_a = compose_net_cascade(pa.net, pa.ns, pa.ni, pb.net, pb.ns, pb.ni), pa.no == pb.ni
    elif op == "PARALLEL_PAIR": expected, ok_a = compose_net_parallel_pair(pa.net, pa.ns, pb.net, pb.ns, pa.ni), pa.ni == pb.ni
    elif op == "PARALLEL_LUT": expected, ok_a = compose_net_parallel_lut(pa.net, pa.ns, pb.net, pb.ns, pa.ni, lut4), (pa.ni == pb.ni and pa.no == 1 and pb.no == 1)
    elif op == "MUX_2IN": expected, ok_a = compose_net_mux_2in(pa.net, pa.ns, pb.net, pb.ns), (pa.ni == 1 and pb.ni == 1 and pa.no == 1 and pb.no == 1)
    elif op == "MOE_MUX": expected, ok_a = compose_net_moe_mux(pa.net, pa.ns, pb.net, pb.ns, pc.net, pc.ns, pa.ni), (pa.ni == pb.ni == pc.ni and pa.no == pb.no == pc.no == 1)
    elif op == "DELAYED_FEEDBACK": expected, ok_a = compose_net_delayed_feedback(pa.net, pa.ns), (pa.ni == 2 and pa.no == 1)
    elif op == "COUPLED_FEEDBACK_2BIT": expected, ok_a = compose_net_coupled_fb_2bit(pa.net, pa.ns, pb.net, pb.ns), (pa.ni == 2 and pb.ni == 2 and pa.no == 1 and pb.no == 1)
    else: return False, f"unsupported AG op {op}"
    if not ok_a: return False, "port arity mismatch"
    if any(raw_net[k] != expected[k] for k in ("n_inputs", "gates", "outs")): return False, "subcircuit mismatch"
    return True, f"exact modular assume-guarantee ({op})"

def _compose_anf_and_rows(op: str, pa: Skill, pb: Skill | None = None, pc: Skill | None = None, lut4: int = 0):
    if op == "CASCADE":
        ns1, ns2 = pa.ns, pb.ns
        ao = tuple(remap(p, tuple(list(range(ns1)) + [ns1 + ns2 + j for j in range(pa.ni)])) for p in pa.anf)
        anf = tuple(ao[:ns1]) + tuple(substitute(p, tuple([frozenset({1 << (ns1 + j)}) for j in range(ns2)] + list(ao[ns1:]))) for p in pb.anf)
        rows = []
        for x in range(1 << (ns1 + ns2 + pa.ni)):
            s1, s2, u = x & ((1 << ns1) - 1), (x >> ns1) & ((1 << ns2) - 1), x >> (ns1 + ns2)
            r1 = pa.rows[s1 | (u << ns1)]; r2 = pb.rows[s2 | (sum(v << i for i, v in enumerate(r1[ns1:])) << ns2)]
            rows.append(tuple(list(r1[:ns1]) + list(r2)))
        return anf, rows
    if op == "PARALLEL_PAIR":
        ns1, ns2, ni = pa.ns, pb.ns, pa.ni
        ao = tuple(remap(p, tuple(list(range(ns1)) + [ns1 + ns2 + j for j in range(ni)])) for p in pa.anf)
        bo = tuple(remap(p, tuple(list(range(ns1, ns1 + ns2)) + [ns1 + ns2 + j for j in range(ni)])) for p in pb.anf)
        anf = tuple(ao[:ns1]) + tuple(bo[:ns2]) + tuple(ao[ns1:]) + tuple(bo[ns2:])
        rows = [tuple(list(pa.rows[(x & ((1 << ns1) - 1)) | ((x >> (ns1 + ns2)) << ns1)][:ns1]) +
                      list(pb.rows[((x >> ns1) & ((1 << ns2) - 1)) | ((x >> (ns1 + ns2)) << ns2)][:ns2]) +
                      list(pa.rows[(x & ((1 << ns1) - 1)) | ((x >> (ns1 + ns2)) << ns1)][ns1:]) +
                      list(pb.rows[((x >> ns1) & ((1 << ns2) - 1)) | ((x >> (ns1 + ns2)) << ns2)][ns2:]))
                for x in range(1 << (ns1 + ns2 + ni))]
        return anf, rows
    if op == "PARALLEL_LUT":
        ns1, ns2, ni = pa.ns, pb.ns, pa.ni
        ao = tuple(remap(p, tuple(list(range(ns1)) + [ns1 + ns2 + j for j in range(ni)])) for p in pa.anf)
        bo = tuple(remap(p, tuple(list(range(ns1, ns1 + ns2)) + [ns1 + ns2 + j for j in range(ni)])) for p in pb.anf)
        anf = tuple(ao[:ns1]) + tuple(bo[:ns2]) + (substitute(lut4_anf(lut4), (ao[ns1], bo[ns2])),)
        rows = []
        for x in range(1 << (ns1 + ns2 + ni)):
            r1 = pa.rows[(x & ((1 << ns1) - 1)) | ((x >> (ns1 + ns2)) << ns1)]
            r2 = pb.rows[((x >> ns1) & ((1 << ns2) - 1)) | ((x >> (ns1 + ns2)) << ns2)]
            rows.append(tuple(list(r1[:ns1]) + list(r2[:ns2]) + [lut4_eval(lut4, r1[ns1], r2[ns2])]))
        return anf, rows
    if op == "MUX_2IN":
        ns1, ns2 = pa.ns, pb.ns; u0, u1 = ns1 + ns2, ns1 + ns2 + 1
        ao = tuple(remap(p, tuple(list(range(ns1)) + [u0])) for p in pa.anf)
        bo = tuple(remap(p, tuple(list(range(ns1, ns1 + ns2)) + [u0])) for p in pb.anf)
        anf = tuple(ao[:ns1]) + tuple(bo[:ns2]) + (xor(ao[ns1], mul(frozenset({1 << u1}), xor(ao[ns1], bo[ns2]))),)
        rows = []
        for x in range(1 << (ns1 + ns2 + 2)):
            r1 = pa.rows[(x & ((1 << ns1) - 1)) | (((x >> (ns1 + ns2)) & 1) << ns1)]
            r2 = pb.rows[((x >> ns1) & ((1 << ns2) - 1)) | (((x >> (ns1 + ns2)) & 1) << ns2)]
            rows.append(tuple(list(r1[:ns1]) + list(r2[:ns2]) + [r2[ns2] if ((x >> (ns1 + ns2 + 1)) & 1) else r1[ns1]]))
        return anf, rows
    if op == "MOE_MUX":
        ns_s, ns_a, ns_b, ni = pa.ns, pb.ns, pc.ns, pa.ni; off_a, off_b, off_u = ns_s, ns_s + ns_a, ns_s + ns_a + ns_b
        u_map = [off_u + j for j in range(ni)]
        so = tuple(remap(p, tuple(list(range(ns_s)) + u_map)) for p in pa.anf)
        ao = tuple(remap(p, tuple(list(range(off_a, off_a + ns_a)) + u_map)) for p in pb.anf)
        bo = tuple(remap(p, tuple(list(range(off_b, off_b + ns_b)) + u_map)) for p in pc.anf)
        anf = tuple(so[:ns_s]) + tuple(ao[:ns_a]) + tuple(bo[:ns_b]) + (xor(ao[ns_a], mul(so[ns_s], xor(ao[ns_a], bo[ns_b]))),)
        rows = []
        for x in range(1 << (ns_s + ns_a + ns_b + ni)):
            u = x >> off_u
            rs = pa.rows[(x & ((1 << ns_s) - 1)) | (u << ns_s)]
            ra = pb.rows[((x >> ns_s) & ((1 << ns_a) - 1)) | (u << ns_a)]
            rb = pc.rows[((x >> off_b) & ((1 << ns_b) - 1)) | (u << ns_b)]
            rows.append(tuple(list(rs[:ns_s]) + list(ra[:ns_a]) + list(rb[:ns_b]) + [rb[ns_b] if rs[ns_s] else ra[ns_a]]))
        return anf, rows
    if op == "DELAYED_FEEDBACK":
        ns = pa.ns; mo = tuple(remap(p, tuple(list(range(ns)) + [ns + 1, ns])) for p in pa.anf)
        anf = tuple(mo[:ns]) + (mo[ns], mo[ns])
        rows = []
        for x in range(1 << (ns + 2)):
            s, d, u0 = x & ((1 << ns) - 1), (x >> ns) & 1, (x >> (ns + 1)) & 1
            rm = pa.rows[s | ((u0 | (d << 1)) << ns)]; y = rm[ns]
            rows.append(tuple(list(rm[:ns]) + [y, y]))
        return anf, rows
    ns1, ns2 = pa.ns, pb.ns; da, db, u0 = ns1 + ns2, ns1 + ns2 + 1, ns1 + ns2 + 2
    ao = tuple(remap(p, tuple(list(range(ns1)) + [u0, db])) for p in pa.anf)
    bo = tuple(remap(p, tuple(list(range(ns1, ns1 + ns2)) + [u0, da])) for p in pb.anf)
    anf = tuple(ao[:ns1]) + tuple(bo[:ns2]) + (ao[ns1], bo[ns2], bo[ns2])
    rows = []
    for x in range(1 << (u0 + 1)):
        sa, sb = x & ((1 << ns1) - 1), (x >> ns1) & ((1 << ns2) - 1)
        va, vb, vu = (x >> da) & 1, (x >> db) & 1, x >> u0
        ra = pa.rows[sa | ((vu | (vb << 1)) << ns1)]; rb = pb.rows[sb | ((vu | (va << 1)) << ns2)]
        rows.append(tuple(list(ra[:ns1]) + list(rb[:ns2]) + [ra[ns1], rb[ns2], rb[ns2]]))
    return anf, rows

def compose_quotient(op: str, pa: Skill, pb: Skill | None = None, pc: Skill | None = None, lut4: int = 0):
    if op == "CASCADE":
        Ta, Tb = pa.quotient_T, pb.quotient_T; ka, kb, ni = len(Ta), len(Tb), pa.ni
        raw = [tuple((Ta[s1][u][0] * kb + Tb[s2][sum(v << i for i, v in enumerate(Ta[s1][u][1]))][0],
                      Tb[s2][sum(v << i for i, v in enumerate(Ta[s1][u][1]))][1]) for u in range(1 << ni))
               for s1 in range(ka) for s2 in range(kb)]
        return minimize_quotient_table(ka * kb, ni, pb.no, raw, 0)[:3]
    if op == "PARALLEL_PAIR":
        Ta, Tb = pa.quotient_T, pb.quotient_T; ka, kb, ni = len(Ta), len(Tb), pa.ni
        raw = [tuple((Ta[s1][u][0] * kb + Tb[s2][u][0], Ta[s1][u][1] + Tb[s2][u][1]) for u in range(1 << ni)) for s1 in range(ka) for s2 in range(kb)]
        return minimize_quotient_table(ka * kb, ni, pa.no + pb.no, raw, 0)[:3]
    if op == "PARALLEL_LUT":
        Ta, Tb = pa.quotient_T, pb.quotient_T; ka, kb, ni = len(Ta), len(Tb), pa.ni
        raw = [tuple((Ta[s1][u][0] * kb + Tb[s2][u][0], (lut4_eval(lut4, Ta[s1][u][1][0], Tb[s2][u][1][0]),)) for u in range(1 << ni)) for s1 in range(ka) for s2 in range(kb)]
        return minimize_quotient_table(ka * kb, ni, 1, raw, 0)[:3]
    if op == "MUX_2IN":
        Ta, Tb = pa.quotient_T, pb.quotient_T; ka, kb = len(Ta), len(Tb)
        raw = [tuple((Ta[s1][u & 1][0] * kb + Tb[s2][u & 1][0], Tb[s2][u & 1][1] if ((u >> 1) & 1) else Ta[s1][u & 1][1]) for u in range(4)) for s1 in range(ka) for s2 in range(kb)]
        return minimize_quotient_table(ka * kb, 2, 1, raw, 0)[:3]
    if op == "MOE_MUX":
        Ts, Ta, Tb = pa.quotient_T, pb.quotient_T, pc.quotient_T; ks, ka, kb, ni = len(Ts), len(Ta), len(Tb), pa.ni
        raw = [tuple((((Ts[ss][u][0] * ka) + Ta[sa][u][0]) * kb + Tb[sb][u][0], (Tb[sb][u][1][0] if Ts[ss][u][1][0] else Ta[sa][u][1][0],)) for u in range(1 << ni))
               for ss in range(ks) for sa in range(ka) for sb in range(kb)]
        return minimize_quotient_table(ks * ka * kb, ni, 1, raw, 0)[:3]
    if op == "DELAYED_FEEDBACK":
        Tm = pa.quotient_T; km = len(Tm)
        raw = [tuple((Tm[sm][u0 | (d << 1)][0] * 2 + Tm[sm][u0 | (d << 1)][1][0], (Tm[sm][u0 | (d << 1)][1][0],)) for u0 in (0, 1)) for sm in range(km) for d in (0, 1)]
        return minimize_quotient_table(km * 2, 1, 1, raw, 0)[:3]
    if op == "COUPLED_FEEDBACK_2BIT":
        Ta, Tb = pa.quotient_T, pb.quotient_T; ka, kb = len(Ta), len(Tb)
        raw = [tuple(((((Ta[sa][u0 | (db << 1)][0] * kb + Tb[sb][u0 | (da << 1)][0]) * 2 + Ta[sa][u0 | (db << 1)][1][0]) * 2 + Tb[sb][u0 | (da << 1)][1][0]),
                      (Tb[sb][u0 | (da << 1)][1][0],)) for u0 in (0, 1))
               for sa in range(ka) for sb in range(kb) for da in (0, 1) for db in (0, 1)]
        return minimize_quotient_table(ka * kb * 4, 1, 1, raw, 0)[:3]
    raise ValueError(op)

def canonical_poly(p: frozenset[int], ns: int, ni: int) -> frozenset[int]:
    return remap(p, tuple(list(range(ns)) + [22 + j for j in range(ni)]))

class TemporalDAG:
    """Canonical Semantic & Structural DAG over GF(2) polynomials."""
    def __init__(self):
        self.n = GLOBAL_N; self.gates: list[dict] = []; self.depth = [0] * GLOBAL_N
        self.struct: dict[tuple[int, int, int], int] = {}
        self.wire_polys: list[frozenset[int] | None] = [frozenset({1 << i}) for i in range(GLOBAL_N)]
        self.poly_struct: dict[frozenset[int], int] = {frozenset({1 << i}): i for i in range(GLOBAL_N)}
        self.semantic_merges = 0; self.outputs: dict[str, list[int]] = {}
        self.expected: dict[str, list[list[int]]] = {}; self.ag_skills: dict[str, dict] = {}

    def add(self, s: Skill):
        wires = list(range(s.ns)) + [22 + j for j in range(s.ni)]; track_poly = s.anf is not None
        for g in s.net["gates"]:
            a, b, m = wires[g["a"]], wires[g["b"]], g["mask"]
            if m in (0, 15): a = b = 0
            elif m in (6, 8) and b < a: a, b = b, a
            sig = (m, a, b)
            if sig in self.struct: z = self.struct[sig]
            else:
                gp: frozenset[int] | None = None
                if track_poly and self.wire_polys[a] is not None and self.wire_polys[b] is not None:
                    pa, pb = self.wire_polys[a], self.wire_polys[b]
                    gp = frozenset() if m == 0 else (frozenset({0}) if m == 15 else (xor(pa, pb) if m == 6 else mul(pa, pb)))
                if gp is not None and gp in self.poly_struct:
                    z = self.poly_struct[gp]; self.struct[sig] = z; self.semantic_merges += 1
                else:
                    z = self.n + len(self.gates); self.struct[sig] = z
                    self.gates.append({"a": a, "b": b, "mask": m}); self.depth.append(1 + max(self.depth[a], self.depth[b]))
                    self.wire_polys.append(gp)
                    if gp is not None: self.poly_struct[gp] = z
            wires.append(z)
        self.outputs[s.name] = [wires[o] for o in s.net["outs"]]
        if s.anf is not None: self.expected[s.name] = [sorted(canonical_poly(p, s.ns, s.ni)) for p in s.anf]
        else: self.ag_skills[s.name] = s.net

    def verify(self, skills_dict: dict[str, Skill] | None = None) -> tuple[bool, list[str]]:
        w: list[frozenset[int] | None] = [frozenset({1 << i}) for i in range(self.n)]; err: list[str] = []
        max_anf_gate = -1
        if self.expected:
            need = {o for nm in self.expected for o in self.outputs[nm] if o >= self.n}
            stack = list(need); max_anf_gate = max(need, default=-1); seen = set(need)
            while stack:
                gw = stack.pop(); g = self.gates[gw - self.n]
                for ch in (g["a"], g["b"]):
                    if ch >= self.n and ch not in seen: seen.add(ch); stack.append(ch); max_anf_gate = max(max_anf_gate, ch)
        for i, g in enumerate(self.gates):
            a, b, m = g["a"], g["b"], g["mask"]
            if not (0 <= a < self.n + i and 0 <= b < self.n + i): err.append(f"forward:{i}"); w.append(frozenset()); continue
            w.append((frozenset() if m == 0 else (frozenset({0}) if m == 15 else (xor(w[a], w[b]) if m == 6 else (mul(w[a], w[b]) if m == 8 else frozenset())))) if self.n + i <= max_anf_gate else None)
        for name, exp in self.expected.items():
            if tuple(w[o] for o in self.outputs[name]) != tuple(frozenset(x) for x in exp): err.append(f"output_anf:{name}")
        for name, net in self.ag_skills.items():
            sk = skills_dict.get(name) if skills_dict else None
            wires = (list(range(sk.ns)) + [22 + j for j in range(sk.ni)]) if sk else (list(range(net["n_inputs"] - 1)) + [22])
            for g in net["gates"]:
                a, b, m = wires[g["a"]], wires[g["b"]], g["mask"]
                if m in (0, 15): a = b = 0
                elif m in (6, 8) and b < a: a, b = b, a
                if (m, a, b) not in self.struct: err.append(f"ag_dag_missing:{name}"); break
                wires.append(self.struct[(m, a, b)])
            if [wires[o] for o in net["outs"]] != self.outputs[name]: err.append(f"ag_dag_output:{name}")
        return not err, err

    def metrics(self) -> dict:
        fan = [0] * (self.n + len(self.gates))
        for g in self.gates: fan[g["a"]] += 1; fan[g["b"]] += 1
        for outs in self.outputs.values():
            for x in outs: fan[x] += 1
        d = max((self.depth[o] for outs in self.outputs.values() for o in outs), default=0)
        wb = max(1, (self.n + len(self.gates) - 1).bit_length()); ob = max(1, (self.n + len(self.gates)).bit_length())
        return {"gate_count": len(self.gates), "depth": d, "max_fanout": max(fan, default=0),
                "program_bits": len(self.gates) * (4 + 2 * wb) + sum(len(o) * ob for o in self.outputs.values()), "semantic_merges": self.semantic_merges}

    def export(self) -> dict:
        return {"n_inputs": self.n, "gates": self.gates, "outputs": self.outputs, "expected_anf": self.expected, "ag_skill_names": sorted(self.ag_skills.keys()), "metrics": self.metrics()}

class SelfModel:
    def __init__(self, min_trials: int = 3):
        self.min_trials = min_trials; self.trials = 0; self.exact = 0; self.enabled = False
        self.history: list[dict] = []; self.epistemic_alerts: list[dict] = []; self.mdl_predictions: list[dict] = []; self.family_hits: dict[str, int] = {}
        self.adt_verifications = self.adt_baseline_wp_ticks = self.adt_actual_ticks = self.pac_lstar_solves = self.sop_baseline_gates = self.opt_chosen_gates = 0

    def observe(self, ok: bool, queries: int):
        self.trials += 1; self.exact += int(ok); self.history.append({"trial": self.trials, "exact": bool(ok), "warmup_queries": queries})
        if not ok: self.enabled = False
        elif len(self.history) >= self.min_trials and all(x["exact"] for x in self.history[-self.min_trials:]): self.enabled = True

    def log_alert(self, kind: str, detail: str, blocked: bool):
        self.epistemic_alerts.append({"kind": kind, "detail": detail, "blocked_or_escalated": bool(blocked)})

    def record_mdl(self, pred_hier: bool, actual_mode: str, flat_gates: int, hier_gates: int, min_gates: int):
        self.mdl_predictions.append({"pred_hier": bool(pred_hier), "actual_mode": actual_mode,
                                     "correct_hier_call": bool(pred_hier == (actual_mode == "HIERARCHICAL") or hier_gates == flat_gates),
                                     "flat_gates": flat_gates, "hier_gates": hier_gates, "min_gates": min_gates})

    def record_adt(self, wp_ticks: int, adt_ticks: int):
        self.adt_verifications += 1; self.adt_baseline_wp_ticks += wp_ticks; self.adt_actual_ticks += adt_ticks

    def record_synthesis(self, sop_gates: int, chosen_gates: int):
        self.sop_baseline_gates += sop_gates; self.opt_chosen_gates += chosen_gates

    def record_family_hit(self, family: str): self.family_hits[family] = self.family_hits.get(family, 0) + 1
    def ranked_families(self, default_order: list[str]) -> list[str]:
        return sorted(default_order, key=lambda f: (-self.family_hits.get(f, 0), default_order.index(f)))

    def export(self) -> dict:
        acc = (sum(x["correct_hier_call"] for x in self.mdl_predictions) / len(self.mdl_predictions)) if self.mdl_predictions else 1.0
        adt_saved = (1.0 - self.adt_actual_ticks / self.adt_baseline_wp_ticks) if self.adt_baseline_wp_ticks > 0 else 0.0
        synth_saved = (1.0 - self.opt_chosen_gates / self.sop_baseline_gates) if self.sop_baseline_gates > 0 else 0.0
        return {"min_trials": self.min_trials, "trials": self.trials, "exact": self.exact, "enabled": self.enabled,
                "history": self.history, "epistemic_alerts": self.epistemic_alerts, "mdl_routing_accuracy": round(acc, 4),
                "mdl_observations": len(self.mdl_predictions), "mdl_predictions": self.mdl_predictions, "family_hits": self.family_hits,
                "adt_verifications": self.adt_verifications, "adt_baseline_wp_ticks": self.adt_baseline_wp_ticks,
                "adt_actual_ticks": self.adt_actual_ticks, "adt_savings_pct": round(100.0 * adt_saved, 2),
                "pac_lstar_solves": self.pac_lstar_solves, "sop_baseline_gates": self.sop_baseline_gates,
                "opt_chosen_gates": self.opt_chosen_gates, "synthesis_compression_pct": round(100.0 * synth_saved, 2)}

class Library:
    def __init__(self):
        self.skills: dict[str, Skill] = {}; self.aliases: dict[str, str] = {}; self.cache: dict[str, str] = {}
        self.by_behav: dict[str, str] = {}; self.dag = TemporalDAG(); self.local_gates = 0; self.flat_equivalent_gates = 0

    def add(self, s: Skill, flat_gate_count: int | None = None) -> tuple[str, bool]:
        if s.behav_digest and s.behav_digest in self.by_behav:
            target = self.by_behav[s.behav_digest]; self.aliases[s.name] = target; return target, False
        dg = s.semantic_digest()
        for name, old in self.skills.items():
            if old.semantic_digest() == dg: self.aliases[s.name] = name; return name, False
        self.skills[s.name] = s
        if s.behav_digest: self.by_behav[s.behav_digest] = s.name
        self.local_gates += len(s.net["gates"]); self.flat_equivalent_gates += flat_gate_count if flat_gate_count is not None else len(s.net["gates"])
        self.dag.add(s); return s.name, True

    def resolve(self, name: str) -> str:
        while name in self.aliases: name = self.aliases[name]
        if name not in self.skills: raise KeyError(name)
        return name


# ---------------------------------------------------------------------------
# 5. Interactive Oracle, Lee-Yannakakis Wp-ADT & Active Stochastic Mealy L*
# ---------------------------------------------------------------------------

class NonDeterministicOracleError(RuntimeError): pass
class UnrecoverableNoiseError(RuntimeError): pass

class InteractiveMealyOracle:
    def __init__(self, ni: int, no: int, eval_fn: Callable[[tuple[int, ...]], tuple[tuple[int, ...], ...]],
                 max_pac_reps: int = 120, alpha_step: float = 0.001, min_pac_m: int = 10):
        self.ni, self.no, self._eval_fn = ni, no, eval_fn
        self.queries = self.resets = self.pac_samples = 0
        self.cache: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}; self.pac_mode = False
        self.trie: dict[tuple[int, ...], list[list[int]]] = {}; self.locked: dict[tuple[int, ...], tuple[int, ...]] = {}
        self.alpha_step = alpha_step; self.max_pac_reps = max_pac_reps; self.min_pac_m = min_pac_m
        self._thresh = _binom_thresh_table(alpha_step, min_pac_m, max_pac_reps * 2)

    @classmethod
    def from_rows(cls, ns: int, ni: int, no: int, rows: list[tuple[int, ...]], reset: int = 0) -> "InteractiveMealyOracle":
        def _fn(seq: tuple[int, ...]):
            s = reset; outs = []
            for u in seq: r = rows[s | (u << ns)]; s = sum(v << i for i, v in enumerate(r[:ns])); outs.append(tuple(r[ns:]))
            return tuple(outs)
        return cls(ni, no, _fn)

    def _raw_eval(self, seq: tuple[int, ...]):
        self.resets += 1; outs = self._eval_fn(seq); self.queries += len(seq); return outs

    def _update_trie(self, seq: tuple[int, ...], raw_outs: tuple[tuple[int, ...], ...]):
        thresh = self._thresh; tlen = len(thresh); min_m = self.min_pac_m
        for i in range(len(seq)):
            pref = seq[:i + 1]; cnts = self.trie.setdefault(pref, [[0, 0] for _ in range(self.no)])
            for b in range(self.no): cnts[b][raw_outs[i][b]] += 1
            if pref not in self.locked:
                ok = True
                for c0, c1 in cnts:
                    m_tot = c0 + c1
                    if m_tot < min_m: ok = False; break
                    maj = c0 if c0 >= c1 else c1
                    req = thresh[m_tot] if m_tot < tlen else (m_tot + 1 if binom_p_ge(m_tot, maj) > self.alpha_step else maj)
                    if maj < req: ok = False; break
                if ok: self.locked[pref] = tuple(1 if c1 > c0 else 0 for c0, c1 in cnts)

    def eval_seq(self, seq: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        if not self.pac_mode:
            if seq in self.cache: return self.cache[seq]
            outs = self._raw_eval(seq)
            for i in range(1, len(seq) + 1):
                pref = seq[:i]
                if pref in self.cache and self.cache[pref] != outs[:i]:
                    raise NonDeterministicOracleError(f"non-deterministic output on prefix {pref} at reset {self.resets}")
                self.cache[pref] = outs[:i]
            return outs
        if all(seq[:i + 1] in self.locked for i in range(len(seq))):
            return tuple(self.locked[seq[:i + 1]] for i in range(len(seq)))
        for _ in range(self.max_pac_reps):
            self._update_trie(seq, self._raw_eval(seq)); self.pac_samples += 1
            if all(seq[:i + 1] in self.locked for i in range(len(seq))):
                return tuple(self.locked[seq[:i + 1]] for i in range(len(seq)))
        raise UnrecoverableNoiseError(f"Exact Binomial separation failed after {self.max_pac_reps} reps on seq len {len(seq)}")

    def eval_match(self, seq: tuple[int, ...], expected: tuple[tuple[int, ...], ...]) -> tuple[bool, tuple[int, ...] | None]:
        if not self.pac_mode:
            got = self.eval_seq(seq)
            if got == expected: return True, None
            for i in range(len(seq)):
                if got[i] != expected[i]: return False, seq[:i + 1]
            return True, None
        for i in range(len(seq)):
            pref = seq[:i + 1]
            if pref in self.locked and self.locked[pref] != expected[i]: return False, pref
        if all(seq[:i + 1] in self.locked for i in range(len(seq))): return True, None
        for _ in range(self.max_pac_reps):
            self._update_trie(seq, self._raw_eval(seq)); self.pac_samples += 1
            for i in range(len(seq)):
                pref = seq[:i + 1]
                if pref in self.locked and self.locked[pref] != expected[i]: return False, pref
            if all(seq[:i + 1] in self.locked for i in range(len(seq))): return True, None
        raise UnrecoverableNoiseError(f"Exact Binomial separation failed after {self.max_pac_reps} reps on seq len {len(seq)}")

def eval_quotient_seq(T, seq: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    c = 0; out = []
    for u in seq: c, o = T[c][u]; out.append(o)
    return tuple(out)

def run_seq_state_T(T, seq: tuple[int, ...]) -> int:
    c = 0
    for u in seq: c, _ = T[c][u]
    return c

def build_PW_and_Wi(k: int, ni: int, T):
    P = {0: ()}; q = deque([0])
    while q:
        c = q.popleft()
        for u in range(1 << ni):
            nc, _ = T[c][u]
            if nc not in P: P[nc] = P[c] + (u,); q.append(nc)
    if k == 1: W = [(u,) for u in range(1 << ni)]; return [P[0]], W, {0: W}
    def ev_w(c: int, w: tuple[int, ...]):
        cur = c; out = []
        for u in w: cur, o = T[cur][u]; out.append(o)
        return tuple(out)
    W = []
    while True:
        blocks: dict = {}
        for c in range(k): blocks.setdefault(tuple(ev_w(c, w) for w in W), []).append(c)
        big = [b for b in blocks.values() if len(b) > 1]
        if not big: break
        p, q_st = big[0][0], big[0][1]; pq = deque([(p, q_st, ())]); seen = {(p, q_st)}; found = None
        while pq and found is None:
            a, b, pref = pq.popleft()
            for u in range(1 << ni):
                na, oa = T[a][u]; nb, ob = T[b][u]
                if oa != ob: found = pref + (u,); break
                if (na, nb) not in seen: seen.add((na, nb)); pq.append((na, nb, pref + (u,)))
        if found not in W: W.append(found)
    Wi = {}
    for i in range(k):
        rem = [j for j in range(k) if j != i]; chosen = []
        while rem:
            best_w = max(W, key=lambda w: (sum(1 for j in rem if ev_w(j, w) != ev_w(i, w)) / max(1, len(w)), -len(w)))
            sig_i = ev_w(i, best_w); rem = [j for j in rem if ev_w(j, best_w) == sig_i]; chosen.append(best_w)
        Wi[i] = chosen
    return [P[c] for c in range(k)], W, Wi

def build_hsi_adt(k: int, ni: int, T):
    """Lee-Yannakakis exact min-cost Adaptive Distinguishing Tree + Adaptive Forest HSI."""
    P_map = {0: ()}; q = deque([0])
    while q:
        c = q.popleft()
        for u in range(1 << ni):
            nc, _ = T[c][u]
            if nc not in P_map: P_map[nc] = P_map[c] + (u,); q.append(nc)
    P = [P_map[c] for c in range(k)]
    if k <= 1: W = [(u,) for u in range(1 << ni)]; return P, [{0: w} for w in W], {0: W}
    all_pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]

    if k <= 24 and (1 << ni) <= 4:
        memo = {}; visiting = set()
        def solve_exact(cur_set: frozenset[int], d_left: int):
            if len(cur_set) <= 1: return (0, 0, {c: () for c in cur_set})
            if d_left == 0 or cur_set in visiting: return None
            key = (cur_set, d_left)
            if key in memo: return memo[key]
            visiting.add(cur_set); best = None
            for u in range(1 << ni):
                by_out = {}; merged = False
                for c in cur_set:
                    nc, o = T[c][u]; grp = by_out.setdefault(o, set())
                    if nc in grp: merged = True; break
                    grp.add(nc)
                if merged or (len(by_out) == 1 and frozenset(next(iter(by_out.values()))) == cur_set): continue
                ok = True; wd = 0; total_sz = 0; sub_map = {}
                for o, grp in by_out.items():
                    res = solve_exact(frozenset(grp), d_left - 1)
                    if res is None: ok = False; break
                    d_sub, sz_sub, child_map = res; wd = max(wd, 1 + d_sub); total_sz += len(grp) + sz_sub; sub_map[o] = child_map
                if ok:
                    cand = (total_sz, wd, {c: (u,) + sub_map[T[c][u][1]][T[c][u][0]] for c in cur_set})
                    if best is None or (cand[0], cand[1]) < (best[0], best[1]): best = cand
            visiting.remove(cur_set); memo[key] = best; return best
        best_res = first_d = None
        for d_lim in range(1, min(k + 2, 10)):
            res = solve_exact(frozenset(range(k)), d_lim)
            if res is not None:
                if best_res is None or (res[0], res[1]) < (best_res[0], best_res[1]): best_res = res
                if first_d is None: first_d = d_lim
                if d_lim >= first_d + 1: break
        if best_res is not None: paths = best_res[2]; return P, [paths], {s: [paths[s]] for s in range(k)}

    def shortest_sep_word(ca: int, cb: int):
        pq = deque([(ca, cb, ())]); vis = {(ca, cb)}
        while pq:
            x, y, pref = pq.popleft()
            for u in range(1 << ni):
                nx, ox = T[x][u]; ny, oy = T[y][u]
                if ox != oy: return pref + (u,)
                if nx != ny and (nx, ny) not in vis: vis.add((nx, ny)); pq.append((nx, ny, pref + (u,)))
        raise RuntimeError(f"States {ca}, {cb} indistinguishable")

    def build_adaptive_tree(pref: tuple[int, ...], target_pairs: set[tuple[int, int]]):
        def split_node(state_map, rem_pairs, seen_configs, depth, forced_seq):
            if not rem_pairs or depth >= 2 * k: return {s: () for s in state_map}
            if forced_seq:
                u0 = forced_seq[0]; by_out = {}; sub_pairs = {}
                for orig_s, cur_s in state_map.items(): nc, o = T[cur_s][u0]; by_out.setdefault(o, {})[orig_s] = nc
                for (a, b) in rem_pairs:
                    na, oa = T[state_map[a]][u0]; nb, ob = T[state_map[b]][u0]
                    if oa == ob and na != nb: sub_pairs.setdefault(oa, set()).add((a, b))
                paths = {}
                for o, grp in by_out.items():
                    sp = sub_pairs.get(o, set())
                    if not sp:
                        for s in grp: paths[s] = (u0,)
                    else:
                        sub = split_node(grp, sp, seen_configs, depth + 1, forced_seq[1:])
                        for s in grp: paths[s] = (u0,) + sub[s]
                return paths
            cfg = frozenset((min(state_map[a], state_map[b]), max(state_map[a], state_map[b])) for a, b in rem_pairs)
            if cfg in seen_configs: return {s: () for s in state_map}
            seen_configs = seen_configs | {cfg}; best_score = best_u = best_groups = None
            for u in range(1 << ni):
                by_o = {}; sp = {}; sep = mrg = 0
                for (a, b) in rem_pairs:
                    na, oa = T[state_map[a]][u]; nb, ob = T[state_map[b]][u]
                    if oa != ob: sep += 1
                    elif na == nb: mrg += 1
                    else: sp.setdefault(oa, set()).add((a, b))
                for orig_s, cur_s in state_map.items(): nc, o = T[cur_s][u]; by_o.setdefault(o, {})[orig_s] = nc
                next_cfg = frozenset((min(by_o[o][a], by_o[o][b]), max(by_o[o][a], by_o[o][b])) for o, s_p in sp.items() for a, b in s_p)
                if sep == 0 and (not next_cfg or next_cfg in seen_configs): continue
                sc = (sep - mrg * 100, sep, -mrg, len(by_o))
                if best_score is None or sc > best_score: best_score, best_u, best_groups = sc, u, (by_o, sp)
            if best_u is None or (best_score[1] == 0 and best_score[0] < 0):
                best_w = min((shortest_sep_word(state_map[a], state_map[b]) for a, b in rem_pairs), key=len)
                u0 = best_w[0]; by_o = {}; sp = {}
                for orig_s, cur_s in state_map.items(): nc, o = T[cur_s][u0]; by_o.setdefault(o, {})[orig_s] = nc
                for (a, b) in rem_pairs:
                    na, oa = T[state_map[a]][u0]; nb, ob = T[state_map[b]][u0]
                    if oa == ob and na != nb: sp.setdefault(oa, set()).add((a, b))
                next_cfg = frozenset((min(by_o[o][a], by_o[o][b]), max(by_o[o][a], by_o[o][b])) for o, s_p in sp.items() for a, b in s_p)
                if next_cfg not in seen_configs: best_u, best_groups = u0, (by_o, sp)
                else: return {s: best_w for s in state_map}
            by_out, sub_pairs = best_groups; paths = {}
            for o, grp in by_out.items():
                sp = sub_pairs.get(o, set())
                if not sp:
                    for s in grp: paths[s] = (best_u,)
                else:
                    sub = split_node(grp, sp, seen_configs, depth + 1, ())
                    for s in grp: paths[s] = (best_u,) + sub[s]
            return paths
        active_orig = tuple(sorted({s for p in target_pairs for s in p}))
        tp = split_node({s: s for s in active_orig}, set(target_pairs), set(), 0, pref)
        return {s: tp.get(s, ()) for s in range(k)}

    cands = []; seen_cands = set()
    prefixes = [()] + ([p for L in range(1, 5 if ni == 1 else 3) for p in itertools.product(range(1 << ni), repeat=L)] if (1 << ni) <= 4 else [(u,) for u in range(1 << ni)])
    all_set = set(all_pairs)
    for pref in prefixes:
        tp = build_adaptive_tree(pref, all_set); sig = tuple(tp[s] for s in range(k))
        if sig not in seen_cands: seen_cands.add(sig); cands.append(tp)
        if len(pref) >= 2:
            tp_f = {s: pref for s in range(k)}; sig_f = tuple(tp_f[s] for s in range(k))
            if sig_f not in seen_cands: seen_cands.add(sig_f); cands.append(tp_f)
    _, W_leg, _ = build_PW_and_Wi(k, ni, T)
    for w in W_leg:
        tp_w = {s: w for s in range(k)}; sig_w = tuple(tp_w[s] for s in range(k))
        if sig_w not in seen_cands: seen_cands.add(sig_w); cands.append(tp_w)

    def tree_pair_depths(tp):
        d = {}
        for i, j in all_pairs:
            ci, cj = i, j
            for step, (u1, u2) in enumerate(zip(tp[i], tp[j]), 1):
                if u1 != u2: break
                ci, oi = T[ci][u1]; cj, oj = T[cj][u2]
                if oi != oj: d[(i, j)] = step; break
        return d

    cand_covs = [tree_pair_depths(tp) for tp in cands]
    uncov = all_set - (set().union(*[set(c.keys()) for c in cand_covs]) if cand_covs else set())
    while uncov:
        tp = build_adaptive_tree((), uncov); d_map = tree_pair_depths(tp)
        if not set(d_map.keys()) & uncov:
            a, b = next(iter(uncov)); w = shortest_sep_word(a, b); tp = {s: w for s in range(k)}; d_map = tree_pair_depths(tp)
        cands.append(tp); cand_covs.append(d_map); uncov -= set(d_map.keys())

    cand_singletons = [sum(1 for q in range(k) if all(((q, j) if q < j else (j, q)) in dm for j in range(k) if j != q)) for dm in cand_covs]
    chosen_idxs = []; rem_pairs = set(all_pairs)
    while rem_pairs:
        best_idx = max(range(len(cands)), key=lambda idx: (
            cand_singletons[idx] if any(all(((q, x) if q < x else (x, q)) in cand_covs[idx] for x in range(k) if x != q)
                                        and any(((q, j) if q < j else (j, q)) in rem_pairs for j in range(k) if j != q) for q in range(k)) else 0,
            len(rem_pairs & set(cand_covs[idx].keys())), len(cand_covs[idx]), -sum(len(cands[idx][s]) for s in range(k))))
        chosen_idxs.append(best_idx); rem_pairs -= set(cand_covs[best_idx].keys())

    chosen_trees = [cands[idx] for idx in chosen_idxs]; chosen_covs = [cand_covs[idx] for idx in chosen_idxs]; H_wp = {}
    for q in range(k):
        need = {j for j in range(k) if j != q}; chosen_q = {}
        while need:
            best_r = None; best_score = -1.0; best_L = 0; best_cov = set()
            for r in range(len(chosen_trees)):
                cov_j = {j for j in need if ((q, j) if q < j else (j, q)) in chosen_covs[r]}
                if not cov_j: continue
                L = max(chosen_covs[r][(q, j) if q < j else (j, q)] for j in cov_j); sc = len(cov_j) / (L + 2)
                if sc > best_score: best_score, best_r, best_L, best_cov = sc, r, L, cov_j
            chosen_q[best_r] = best_L; need -= best_cov
        H_wp[q] = [chosen_trees[r][q][:L] for r, L in sorted(chosen_q.items())]
    return P, chosen_trees, H_wp

def dedup_seqs(seqs: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    seen = set(); res = []
    for seq in sorted(set(seqs), key=len, reverse=True):
        if seq not in seen:
            res.append(seq)
            for i in range(1, len(seq) + 1): seen.add(seq[:i])
    return res

@functools.lru_cache(maxsize=512)
def build_wp_test_suite(k: int, ni: int, T, max_states: int = 8, use_adt: bool = True) -> tuple[tuple[int, ...], ...]:
    extra = max(0, max_states - k)
    if ((1 << ni) ** (extra + 1)) * max(1, k) <= 2048: extra_cap, extra_walks = extra, []
    else:
        extra_cap = min(extra, (2 if k >= 10 else 5) if ni == 1 else 1)
        rng = random.Random(0xBEEF + k * 131 + ni)
        extra_walks = [tuple(rng.randrange(1 << ni) for _ in range(max(24, max_states + 6))) for _ in range(48)]
    mids = [()] + [v for L in range(1, extra_cap + 2) for v in itertools.product(range(1 << ni), repeat=L)]
    P_l, W_l, Wi_l = build_PW_and_Wi(k, ni, T)
    leg_seqs = dedup_seqs([p + w for p in P_l for w in W_l] + [p + v + w for p in P_l for v in mids if v for w in Wi_l[run_seq_state_T(T, p + v)]] + extra_walks)
    if not use_adt: return tuple(leg_seqs)
    P, forest, H_q = build_hsi_adt(k, ni, T)
    adt_seqs = dedup_seqs([P[i] + forest[r][i] for i in range(k) for r in range(len(forest))] +
                          [p + v + w for p in P for v in mids if v for w in H_q[run_seq_state_T(T, p + v)]] + extra_walks)
    return tuple(leg_seqs if sum(len(s) for s in leg_seqs) < sum(len(s) for s in adt_seqs) else adt_seqs)

def wp_method_verify(k: int, ni: int, T, bb: InteractiveMealyOracle, max_states: int = 8, use_adt: bool = True):
    for seq in build_wp_test_suite(k, ni, T, max_states=max_states, use_adt=use_adt):
        ok_s, cex = bb.eval_match(seq, eval_quotient_seq(T, seq))
        if not ok_s: return False, cex
    return True, None

def lstar_learn_mealy(bb: InteractiveMealyOracle, max_states: int = 8, use_adt: bool = True):
    ni = bb.ni; alphabet = list(range(1 << ni)); S = [()]; E = [(u,) for u in alphabet]; row_cache: dict = {}
    def batch_eval(seqs):
        for seq in dedup_seqs(seqs): bb.eval_seq(seq)
    def row(s: tuple[int, ...]):
        key = (s, len(E))
        if key in row_cache: return row_cache[key]
        if bb.pac_mode: batch_eval([s + e for e in E])
        res = tuple(bb.eval_seq(s + e)[len(s):] for e in E); row_cache[key] = res; return res

    probe_len = max(12, max_states + 2)
    quick_probes = ([(1,) * probe_len, (0,) * probe_len, (0, 1) * (probe_len // 2), (1, 0) * (probe_len // 2),
                     (1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1)[:probe_len], (0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1)[:probe_len]] if ni == 1
                    else [tuple([u] * probe_len) for u in alphabet] + [tuple((i + j) % (1 << ni) for i in range(probe_len)) for j in alphabet])

    while True:
        while True:
            if bb.pac_mode: batch_eval([s + e for s in S for e in E] + [s + (u,) + e for s in S for u in alphabet for e in E])
            sr = {s: row(s) for s in S}; urows = {}
            for s in S:
                if sr[s] not in urows: urows[sr[s]] = s
            added = False
            for s in list(S):
                for u in alphabet:
                    if row(s + (u,)) not in urows: S.append(s + (u,)); added = True; break
                if added: break
            if added: continue
            by_row: dict = {}
            for s in S: by_row.setdefault(sr[s], []).append(s)
            inc = False
            for grp in by_row.values():
                if len(grp) < 2: continue
                s1 = grp[0]
                for s2 in grp[1:]:
                    for u in alphabet:
                        r1, r2 = row(s1 + (u,)), row(s2 + (u,))
                        if r1 != r2:
                            for idx, e in enumerate(E):
                                if r1[idx] != r2[idx] and ((u,) + e) not in E: E.append((u,) + e); inc = True; break
                        if inc: break
                    if inc: break
                if inc: break
            if not inc: break
        sr = {s: row(s) for s in S}; rep = []; r2id = {}
        for s in S:
            if sr[s] not in r2id: r2id[sr[s]] = len(rep); rep.append(s)
        k = len(rep); T = tuple(tuple((r2id[row(s + (u,))], bb.eval_seq(s + (u,))[-1]) for u in alphabet) for s in rep)

        cex = None
        if bb.pac_mode:
            for seq in sorted(bb.locked.keys(), key=len):
                if seq and bb.locked[seq] != eval_quotient_seq(T, seq)[-1]: cex = seq; break
        else:
            for seq in sorted(bb.cache.keys(), key=len):
                if not seq: continue
                exp_seq = eval_quotient_seq(T, seq)
                if bb.cache[seq] != exp_seq:
                    for i in range(len(seq)):
                        if bb.cache[seq][i] != exp_seq[i]: cex = seq[:i + 1]; break
                    break
        if cex is None and (k <= 2 or ni >= 2):
            for seq in quick_probes:
                ok_s, cex = bb.eval_match(seq, eval_quotient_seq(T, seq))
                if not ok_s: break
        if cex is None:
            ok, cex = wp_method_verify(k, ni, T, bb, max_states=max(max_states, k), use_adt=use_adt)
            if ok: return k, T

        def hyp_match(i: int) -> bool:
            q_i = run_seq_state_T(T, cex[:i]); r_s = rep[q_i]; suff = cex[i:]
            exp_full = eval_quotient_seq(T, r_s); exp_suff = []; c = q_i
            for u in suff: c, o = T[c][u]; exp_suff.append(o)
            return bb.eval_match(r_s + suff, exp_full + tuple(exp_suff))[0]

        if not hyp_match(len(cex) - 1):
            for j in range(1, len(cex) + 1):
                if cex[:j] not in S: S.append(cex[:j])
        else:
            lo, hi = 0, len(cex) - 1
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if not hyp_match(mid): lo = mid
                else: hi = mid
            e_new = cex[lo + 1:]
            if e_new not in E: E.append(e_new)
            else:
                s_new = rep[run_seq_state_T(T, cex[:lo])] + (cex[lo],)
                if s_new not in S: S.append(s_new)
                else:
                    for j in range(1, len(cex) + 1):
                        if cex[:j] not in S: S.append(cex[:j])


# ---------------------------------------------------------------------------
# 6. AutonomousTemporalAgent Facade & Checkpoint Persistence
# ---------------------------------------------------------------------------

class AutonomousTemporalAgent:
    def __init__(self, issuer: str = DEFAULT_ISSUER, min_warmup_trials: int = 3,
                 use_adt: bool = True, use_factored_synthesis: bool = True):
        self.issuer = issuer; self.use_adt = use_adt; self.use_factored_synthesis = use_factored_synthesis
        self.learner = ProofCarryingANFLearner({issuer}, max_full_support_inputs=14)
        self.lib = Library(); self.model = SelfModel(min_warmup_trials); self.trace: list[dict] = []

    def _compile_anf(self, n: int, anf: tuple[frozenset[int], ...]) -> dict:
        sop = compact_netlist(self.learner.compile(n, anf))
        if self.use_factored_synthesis:
            opt = self.learner.compile_optimal(n, anf); self.model.record_synthesis(len(sop["gates"]), len(opt["gates"])); return opt
        self.model.record_synthesis(len(sop["gates"]), len(sop["gates"])); return sop

    def build_skill_from_rows(self, name: str, ns: int, ni: int, no: int, rows: list[tuple[int, ...]],
                              route: str, op: str = "PRIMITIVE", parents: tuple[str, ...] = (),
                              queries: int = 0, prefer_mdl: bool = True, generation: int = 0) -> Skill:
        n = ns + ni; anf = mobius(rows, n); ns_min, k, bdig, qT, min_rows, _ = minimize_mealy_from_rows(ns, ni, no, rows, 0)
        cands = [("FLAT", ns, rows, anf, self._compile_anf(n, anf))]
        if prefer_mdl:
            if self.use_factored_synthesis:
                opt_ns, opt_rows, opt_anf, opt_net = optimal_encode_quotient(k, ni, no, qT)
                sop_m = compact_netlist(self.learner.compile(ns_min + ni, mobius(min_rows, ns_min + ni)))
                self.model.record_synthesis(len(sop_m["gates"]), len(opt_net["gates"]))
                cands.append(("OPT_QUOTIENT", opt_ns, opt_rows, opt_anf, opt_net))
            else:
                min_anf = mobius(min_rows, ns_min + ni)
                cands.append(("MIN_QUOTIENT", ns_min, min_rows, min_anf, compact_netlist(self.learner.compile(ns_min + ni, min_anf))))
        mode, b_ns, b_rows, b_anf, b_net = min(cands, key=lambda c: (len(c[4]["gates"]), c[4]["metrics"]["depth"], c[1], c[4]["metrics"]["program_bits"]))
        if not symbolic_verify_general(b_net, b_anf): raise RuntimeError(f"{name}: symbolic verify fail")
        return Skill(name, b_ns, ni, no, 0, b_rows, b_anf, b_net, route, op, parents, queries, mode, k, ns_min, bdig, qT, True, None, generation)

    def learn_primitive(self, name: str, ns: int, ni: int, no: int, oracle_fn: Callable[[int], tuple[int, ...]],
                        degree_bound: int | None = None, contract: DegreeContract | None = None,
                        allow_full_support: bool = True, generation: int = 0) -> Skill | None:
        n = ns + ni
        if contract is None and degree_bound is not None: contract = DegreeContract(n, degree_bound, self.issuer, 2)
        c, dec = self.learner.fit(n, ns + no, oracle_fn, contract=contract, allow_full_support=allow_full_support, audit_full_support=True)
        if c is None:
            self.model.log_alert("PRIMITIVE_LEARN_HOLD", f"{name}: {dec.get('reason')}", True)
            self.trace.append({"name": name, "op": "PRIMITIVE", "route": "HOLD", "queries": 0, "audit_rows": 0, "action": "HOLD", "generation": generation})
            return None
        if dec.get("degree_contract_rejected"):
            self.model.log_alert("FALSE_DEGREE_BOUND_ESCALATED", f"{name}: escalated to {c.query_count} full-support queries", True)
        rows = [tuple(int(v) & 1 for v in oracle_fn(x)) for x in range(1 << n)]
        s = self.build_skill_from_rows(name, ns, ni, no, rows, c.mode.upper(), "PRIMITIVE", (), c.query_count, True, generation)
        target, added = self.lib.add(s)
        self.trace.append({"name": name, "op": "PRIMITIVE", "route": s.route, "queries": c.query_count, "audit_rows": len(rows),
                           "action": "COMMIT" if added else "REUSE", "parents": [], "nvars": s.nvars, "expert": target,
                           "compile_mode": s.compile_mode, "k_states": s.k_states, "global_gates_after": len(self.lib.dag.gates), "generation": generation})
        return self.lib.skills[target]

    def compose(self, op: str, pa: Skill, pb: Skill | None, name: str, pc: Skill | None = None, lut4: int = 0,
                route_override: str | None = None, extra_queries: int = 0, generation: int = 0) -> Skill:
        pnames = (pa.name,) if pb is None else ((pa.name, pb.name) if pc is None else (pa.name, pb.name, pc.name))
        key = digest({"op": op, "lut4": lut4, "parents": [pa.behav_digest or pa.semantic_digest(),
                      None if pb is None else (pb.behav_digest or pb.semantic_digest()),
                      None if pc is None else (pc.behav_digest or pc.semantic_digest())]})
        if key in self.lib.cache:
            old = self.lib.skills[self.lib.resolve(self.lib.cache[key])]
            alias = Skill(name, old.ns, old.ni, old.no, old.reset, old.rows, old.anf, old.net, "MACRO_REUSE", op, pnames, 0,
                          old.compile_mode, old.k_states, old.ns_min, old.behav_digest, old.quotient_T, True, old.ag_cert, generation)
            resolved, _ = self.lib.add(alias)
            self.trace.append({"name": name, "op": op, "route": "MACRO_REUSE", "queries": 0, "audit_rows": 0, "action": "REUSE", "parents": list(pnames), "expert": resolved, "nvars": old.nvars, "generation": generation})
            return self.lib.skills[resolved]

        if op == "CASCADE":
            if pa.no != pb.ni: raise ValueError(f"cascade port mismatch {pa.no}!={pb.ni}")
            ns, ni, no, raw_hier = pa.ns + pb.ns, pa.ni, pb.no, compose_net_cascade(pa.net, pa.ns, pa.ni, pb.net, pb.ns, pb.ni)
        elif op == "PARALLEL_PAIR":
            if pa.ni != pb.ni: raise ValueError("parallel pair ni mismatch")
            ns, ni, no, raw_hier = pa.ns + pb.ns, pa.ni, pa.no + pb.no, compose_net_parallel_pair(pa.net, pa.ns, pb.net, pb.ns, pa.ni)
        elif op == "PARALLEL_LUT":
            if pa.ni != pb.ni or pa.no != 1 or pb.no != 1: raise ValueError("parallel lut port mismatch")
            ns, ni, no, raw_hier = pa.ns + pb.ns, pa.ni, 1, compose_net_parallel_lut(pa.net, pa.ns, pb.net, pb.ns, pa.ni, lut4)
        elif op == "MUX_2IN":
            if not (pa.ni == pb.ni == 1 and pa.no == pb.no == 1): raise ValueError("mux_2in port mismatch")
            ns, ni, no, raw_hier = pa.ns + pb.ns, 2, 1, compose_net_mux_2in(pa.net, pa.ns, pb.net, pb.ns)
        elif op == "MOE_MUX":
            if not (pa.ni == pb.ni == pc.ni and pa.no == pb.no == pc.no == 1): raise ValueError("moe port mismatch")
            ns, ni, no, raw_hier = pa.ns + pb.ns + pc.ns, pa.ni, 1, compose_net_moe_mux(pa.net, pa.ns, pb.net, pb.ns, pc.net, pc.ns, pa.ni)
        elif op == "DELAYED_FEEDBACK":
            if pa.ni != 2 or pa.no != 1: raise ValueError("feedback requires ni=2, no=1")
            ns, ni, no, raw_hier = pa.ns + 1, 1, 1, compose_net_delayed_feedback(pa.net, pa.ns)
        elif op == "COUPLED_FEEDBACK_2BIT":
            if not (pa.ni == pb.ni == 2 and pa.no == pb.no == 1): raise ValueError("coupled 2bit requires two ni=2,no=1")
            ns, ni, no, raw_hier = pa.ns + pb.ns + 2, 1, 1, compose_net_coupled_fb_2bit(pa.net, pa.ns, pb.net, pb.ns)
        else: raise ValueError(op)

        n = ns + ni
        if n > AUDIT_CAP:
            ok_ag, msg_ag = verify_ag_raw(op, raw_hier, pa, pb, pc, lut4)
            if not ok_ag: raise RuntimeError(f"{name}: AG fail: {msg_ag}")
            opt_net, comp_proof = compact_netlist_with_proof(raw_hier)
            ok_cp, msg_cp = verify_compaction_proof(raw_hier, opt_net, comp_proof)
            if not ok_cp: raise RuntimeError(f"{name}: compaction proof fail: {msg_cp}")
            ag_cert = {"kind": f"assume-guarantee-compacted-{op.lower()}", "parents": list(pnames),
                       "raw_gates": len(raw_hier["gates"]), "compacted_gates": len(opt_net["gates"]), "proof_steps": len(comp_proof["proof_steps"])}
            s = Skill(name, ns, ni, no, 0, None, None, opt_net, route_override or "BEYOND_AUDIT_MODULAR_AG", op, pnames, extra_queries,
                      "MODULAR_AG_COMPACT", 0, ns, "AG:" + digest({"op": op, "parents": list(pnames), "lut4": lut4}), None, False, ag_cert, generation)
            resolved, added = self.lib.add(s, flat_gate_count=len(raw_hier["gates"])); self.lib.cache[key] = resolved
            self.trace.append({"name": name, "op": op, "route": s.route, "queries": extra_queries, "audit_rows": 0,
                               "action": "COMMIT" if added else "REUSE", "parents": list(pnames), "nvars": n, "expert": resolved,
                               "compile_mode": "MODULAR_AG_COMPACT", "chosen_gates": len(opt_net["gates"]), "global_gates_after": len(self.lib.dag.gates), "generation": generation})
            return self.lib.skills[resolved]

        hier_net = compact_netlist(raw_hier); cand_anf, rows = _compose_anf_and_rows(op, pa, pb, pc, lut4)
        if not symbolic_verify_general(hier_net, cand_anf): raise RuntimeError(f"{name}: hierarchical symbolic fail")
        expected = mobius(rows, n); bound = max((deg(p) for p in expected), default=0)
        warm = (not self.model.enabled) and (op == "CASCADE") and (route_override is None); queries = extra_queries
        if warm:
            c, dec = self.learner.fit(n, ns + no, lambda x: rows[x], contract=DegreeContract(n, bound, self.issuer, 2), allow_full_support=True, audit_full_support=False)
            if c is None: raise RuntimeError(f"warmup synthesis failed {name}: {dec}")
            exact = tuple(c.outputs) == cand_anf; self.model.observe(exact, c.query_count); queries += c.query_count
            learned, route = (tuple(c.outputs), "CASCADE_WARMUP") if exact else (expected, "EXACT_FALLBACK")
        else:
            learned, route = cand_anf, route_override or f"CERTIFIED_{op}"
        if learned != expected: raise RuntimeError(f"{name}: candidate ANF mismatch")

        ns_min, k, bdig, qT, min_rows, _ = minimize_mealy_from_rows(ns, ni, no, rows, 0)
        if bdig in self.lib.by_behav:
            old = self.lib.skills[self.lib.by_behav[bdig]]
            alias = Skill(name, old.ns, old.ni, old.no, old.reset, old.rows, old.anf, old.net, "BEHAVIORAL_REUSE", op, pnames, queries,
                          old.compile_mode, old.k_states, old.ns_min, bdig, old.quotient_T, True, old.ag_cert, generation)
            resolved, _ = self.lib.add(alias); self.lib.cache[key] = resolved
            self.trace.append({"name": name, "op": op, "route": "BEHAVIORAL_REUSE", "queries": queries, "audit_rows": len(rows), "action": "REUSE", "parents": list(pnames), "nvars": n, "expert": resolved, "generation": generation})
            return self.lib.skills[resolved]

        flat_sop = compact_netlist(self.learner.compile(n, learned)); flat_net = self._compile_anf(n, learned)
        if self.use_factored_synthesis:
            opt_ns, opt_rows, opt_anf, opt_net = optimal_encode_quotient(k, ni, no, qT)
            min_sop = compact_netlist(self.learner.compile(ns_min + ni, mobius(min_rows, ns_min + ni)))
            self.model.record_synthesis(len(min_sop["gates"]), len(opt_net["gates"]))
            q_cand = ("OPT_QUOTIENT", opt_ns, opt_rows, opt_anf, opt_net)
        else:
            min_anf = mobius(min_rows, ns_min + ni)
            q_cand = ("MIN_QUOTIENT", ns_min, min_rows, min_anf, compact_netlist(self.learner.compile(ns_min + ni, min_anf)))
        mode, b_ns, b_rows, b_anf, b_net = min([("HIERARCHICAL", ns, rows, learned, hier_net), ("FLAT", ns, rows, learned, flat_net), q_cand],
                                               key=lambda c: (len(c[4]["gates"]), c[4]["metrics"]["depth"], c[1], c[4]["metrics"]["program_bits"]))
        if not symbolic_verify_general(b_net, b_anf): raise RuntimeError(f"{name}: chosen netlist {mode} symbolic fail")
        deg_a = max((deg(p) for p in pa.anf), default=0); deg_b = max((deg(p) for p in pb.anf), default=0) if pb is not None else 1
        self.model.record_mdl((deg_a * deg_b >= 3) or (len(hier_net["gates"]) < len(flat_sop["gates"])), mode, len(flat_sop["gates"]), len(hier_net["gates"]), len(q_cand[4]["gates"]))

        s = Skill(name, b_ns, ni, no, 0, b_rows, b_anf, b_net, route, op, pnames, queries, mode, k, ns_min, bdig, qT, True, None, generation)
        resolved, added = self.lib.add(s, flat_gate_count=len(flat_sop["gates"])); self.lib.cache[key] = resolved
        self.trace.append({"name": name, "op": op, "route": route, "queries": queries, "audit_rows": len(rows), "action": "COMMIT" if added else "REUSE",
                           "parents": list(pnames), "nvars": n, "compile_mode": mode, "chosen_gates": len(b_net["gates"]), "k_states": k,
                           "global_gates_after": len(self.lib.dag.gates), "generation": generation})
        return self.lib.skills[resolved]

    def solve_goal(self, name: str, oracle: InteractiveMealyOracle, max_states_hint: int = 12,
                   allow_pac_denoise: bool = True, generation: int = 1) -> tuple[Skill | None, dict]:
        for _ in (0, 1):
            try:
                return self._solve_goal_inner(name, oracle, max_states_hint, generation)
            except NonDeterministicOracleError as e:
                if not allow_pac_denoise or oracle.pac_mode:
                    self.model.log_alert("NON_DETERMINISTIC_ORACLE_HOLD", str(e), True)
                    self.trace.append({"name": name, "op": "GOAL_SOLVE", "route": "HOLD_NON_DETERMINISTIC", "queries": oracle.queries, "audit_rows": 0, "action": "HOLD", "reason": str(e), "generation": generation})
                    return None, {"solved_by": "HOLD_NON_DETERMINISTIC", "bb_ticks": oracle.queries}
                oracle.cache.clear(); oracle.pac_mode = True
            except UnrecoverableNoiseError as ue:
                self.model.log_alert("UNRECOVERABLE_NOISE_HOLD", str(ue), True)
                self.trace.append({"name": name, "op": "GOAL_SOLVE", "route": "HOLD_UNRECOVERABLE_NOISE", "queries": oracle.queries, "audit_rows": 0, "action": "HOLD", "reason": str(ue), "generation": generation})
                return None, {"solved_by": "HOLD_UNRECOVERABLE_NOISE", "bb_ticks": oracle.queries}
        return None, {"solved_by": "HOLD", "bb_ticks": oracle.queries}

    def _solve_goal_inner(self, name: str, bb: InteractiveMealyOracle, max_states_hint: int, generation: int) -> tuple[Skill | None, dict]:
        ni, no = bb.ni, bb.no
        probes = ([(0,)*8, (1,)*8, (0,1)*4, (1,0)*4, (1,0,1,1,0,1,0,0), (0,1,1,0,1,0,1,1), (1,1,1,0,0,0,1,0), (0,0,1,1,0,0,1,1,0,0,1,1)]
                  if ni == 1 else [tuple([u] * 8) for u in range(1 << ni)] + [tuple((i + j) % (1 << ni) for i in range(8)) for j in range(1 << ni)])
        probe_obs = [(seq, bb.eval_seq(seq)) for seq in sorted(probes, key=len, reverse=True)]
        route_tag = "GOAL_PAC_DENOISED_SOLVED" if bb.pac_mode else "GOAL_COMPOSITION_SOLVED"

        def verify_T(k_cand: int, T_cand) -> tuple[bool, tuple[int, ...] | None]:
            hint_m = max(max_states_hint, k_cand)
            if self.use_adt:
                adt_suite = build_wp_test_suite(k_cand, ni, T_cand, max_states=hint_m, use_adt=True)
                for seq in adt_suite:
                    ok_s, cex = bb.eval_match(seq, eval_quotient_seq(T_cand, seq))
                    if not ok_s: return False, cex
                leg_suite = build_wp_test_suite(k_cand, ni, T_cand, max_states=hint_m, use_adt=False)
                self.model.record_adt(sum(len(s) for s in leg_suite), sum(len(s) for s in adt_suite))
                return True, None
            return wp_method_verify(k_cand, ni, T_cand, bb, max_states=hint_m, use_adt=False)

        def build_probe_bitvec(obs, out_bit: int = 0):
            total_bits = sum(len(seq) for seq, _ in obs); g_vec = bit = 0
            for _, out in obs:
                for o in out:
                    if o[out_bit]: g_vec |= 1 << bit
                    bit += 1
            return g_vec, (1 << total_bits) - 1

        def skill_probe_vec(T, obs):
            v = bit = 0
            for seq, _ in obs:
                for o in eval_quotient_seq(T, seq):
                    if o[0]: v |= 1 << bit
                    bit += 1
            return v

        for s in list(self.lib.skills.values()):
            if s.ni == ni and s.no == no and s.quotient_T is not None and all(eval_quotient_seq(s.quotient_T, seq) == out for seq, out in probe_obs):
                ok, cex = verify_T(s.k_states, s.quotient_T)
                if ok:
                    self.model.record_family_hit("LIBRARY_REUSE")
                    alias = Skill(name, s.ns, s.ni, s.no, s.reset, s.rows, s.anf, s.net, "GOAL_LIBRARY_REUSE", "REUSE", (s.name,), bb.queries, s.compile_mode, s.k_states, s.ns_min, s.behav_digest, s.quotient_T, True, s.ag_cert, generation)
                    res, _ = self.lib.add(alias)
                    self.trace.append({"name": name, "op": "GOAL_SOLVE", "route": "GOAL_LIBRARY_REUSE", "queries": bb.queries, "audit_rows": 0, "action": "REUSE", "expert": res, "generation": generation})
                    return self.lib.skills[res], {"solved_by": "LIBRARY_REUSE" + ("+PAC" if bb.pac_mode else ""), "bb_ticks": bb.queries, "pac_samples": bb.pac_samples}
                probe_obs.append((cex, bb.eval_seq(cex)))

        skills_list = [s for s in self.lib.skills.values() if s.quotient_T is not None and s.ns <= 6]
        s11 = [s for s in skills_list if s.ni == ni and s.no == 1]; s21 = [s for s in skills_list if s.ni == 2 and s.no == 1]

        def try_commit(op_kind: str, parents_tuple: tuple[Skill, ...], lut4_val: int = 0, fam_tag: str | None = None):
            pa = parents_tuple[0]; pb = parents_tuple[1] if len(parents_tuple) > 1 else None; pc = parents_tuple[2] if len(parents_tuple) > 2 else None
            k_cand, _, T_cand = compose_quotient(op_kind, pa, pb, pc, lut4_val)
            ok, cex = verify_T(k_cand, T_cand)
            if not ok: probe_obs.append((cex, bb.eval_seq(cex))); return None
            self.model.record_family_hit(fam_tag or op_kind)
            solved = self.compose(op_kind, pa, pb, name, pc=pc, lut4=lut4_val, route_override=route_tag, extra_queries=bb.queries, generation=generation)
            label = op_kind if op_kind != "PARALLEL_LUT" else f"PARALLEL_{lut4_name(lut4_val)}"
            return solved, {"solved_by": f"COMPOSITION({label})" + ("+PAC" if bb.pac_mode else ""), "parents": [p.name for p in parents_tuple], "bb_ticks": bb.queries, "k_states": k_cand, "pac_samples": bb.pac_samples}

        def run_family(fam: str):
            if fam == "PARALLEL_PAIR" and ni == 1 and no == 2:
                g0, _ = build_probe_bitvec(probe_obs, 0); g1, _ = build_probe_bitvec(probe_obs, 1); sv = [skill_probe_vec(s.quotient_T, probe_obs) for s in s11]
                for a in [s for i, s in enumerate(s11) if sv[i] == g0]:
                    for b in [s for j, s in enumerate(s11) if sv[j] == g1]:
                        if a.ns + b.ns <= 8 and (res := try_commit("PARALLEL_PAIR", (a, b), fam_tag=fam)) is not None: return res
            elif fam == "DELAYED_FEEDBACK" and ni == 1 and no == 1:
                def ev_fb(Tm, seq):
                    sm = d = 0; out = []
                    for u0 in seq: sm, om = Tm[sm][u0 | (d << 1)]; d = om[0]; out.append(om)
                    return tuple(out)
                for m in s21:
                    if m.ns + 1 <= 10 and all(ev_fb(m.quotient_T, seq) == out for seq, out in probe_obs) and (res := try_commit("DELAYED_FEEDBACK", (m,), fam_tag=fam)) is not None: return res
            elif fam == "COUPLED_FEEDBACK_2BIT" and ni == 1 and no == 1:
                def ev_cfb(Ta, Tb, seq):
                    sa = sb = da = db = 0; out = []
                    for u0 in seq: sa, oa = Ta[sa][u0 | (db << 1)]; sb, ob = Tb[sb][u0 | (da << 1)]; da, db = oa[0], ob[0]; out.append(ob)
                    return tuple(out)
                for a in s21:
                    for b in s21:
                        if a.ns + b.ns + 2 <= 8 and all(ev_cfb(a.quotient_T, b.quotient_T, seq) == out for seq, out in probe_obs) and (res := try_commit("COUPLED_FEEDBACK_2BIT", (a, b), fam_tag=fam)) is not None: return res
            elif fam == "CASCADE" and no == 1:
                for a in [s for s in skills_list if s.ni == ni]:
                    for b in skills_list:
                        if a.no == b.ni and b.no == no and a.ns + b.ns <= 10:
                            def ev_c(seq, Ta=a.quotient_T, Tb=b.quotient_T):
                                sa = sb = 0; out = []
                                for u in seq: sa, o1 = Ta[sa][u]; sb, o2 = Tb[sb][sum(v << i for i, v in enumerate(o1))]; out.append(o2)
                                return tuple(out)
                            if all(ev_c(seq) == out for seq, out in probe_obs) and (res := try_commit("CASCADE", (a, b), fam_tag=fam)) is not None: return res
            elif fam == "PARALLEL_LUT" and ni == 1 and no == 1:
                g_vec, mask_all = build_probe_bitvec(probe_obs); sv = [skill_probe_vec(s.quotient_T, probe_obs) for s in s11]
                for i, a in enumerate(s11):
                    va, na = sv[i], sv[i] ^ mask_all
                    for j in range(i + 1, len(s11)):
                        b = s11[j]
                        if a.ns + b.ns > 9: continue
                        vb, nb = sv[j], sv[j] ^ mask_all; ok_l, fixed_l, free_b = True, 0, []
                        for bit, m in enumerate((na & nb, va & nb, na & vb, va & vb)):
                            if m == 0: free_b.append(bit)
                            elif (g_vec & m) == m: fixed_l |= 1 << bit
                            elif (g_vec & m) != 0: ok_l = False; break
                        if ok_l:
                            luts = [fixed_l]
                            for fb in free_b: luts = [l for x in luts for l in (x, x | (1 << fb))]
                            for lc in sorted(luts, key=lambda l: (bin(l).count("1"), l)):
                                if lc not in (0, 15, 3, 5, 10, 12):
                                    if (res := try_commit("PARALLEL_LUT", (a, b), lut4_val=lc, fam_tag=fam)) is not None: return res
                                    g_vec, mask_all = build_probe_bitvec(probe_obs)
            elif fam == "PAIR2_TO_2IN" and ni == 1 and no == 1:
                small11 = [s for s in s11 if s.ns <= 2]
                for m in s21:
                    for a in small11:
                        for b in small11:
                            if a is b or a.ns + b.ns + m.ns > 8: continue
                            def ev_p2c(seq, Ta=a.quotient_T, Tb=b.quotient_T, Tm=m.quotient_T):
                                sa = sb = sm = 0; out = []
                                for u in seq: sa, o1 = Ta[sa][u]; sb, o2 = Tb[sb][u]; sm, om = Tm[sm][o1[0] | (o2[0] << 1)]; out.append(om)
                                return tuple(out)
                            if all(ev_p2c(seq) == out for seq, out in probe_obs):
                                _, _, Tab = compose_quotient("PARALLEL_PAIR", a, b)
                                k_f, _, Tf = compose_quotient("CASCADE", Skill("__p", a.ns + b.ns, 1, 2, 0, None, None, {}, "", "", (), 0, "", 0, 0, "", Tab), m)
                                ok, cex = verify_T(k_f, Tf)
                                if ok:
                                    self.model.record_family_hit(fam)
                                    pair_sk = self.compose("PARALLEL_PAIR", a, b, f"{name}_vec2", route_override="GOAL_PAIR2_INTERMEDIATE", generation=generation)
                                    solved = self.compose("CASCADE", pair_sk, m, name, route_override="GOAL_PAIR2_CASCADE_SOLVED", extra_queries=bb.queries, generation=generation)
                                    return solved, {"solved_by": "COMPOSITION(PAIR2_TO_2IN)" + ("+PAC" if bb.pac_mode else ""), "parents": [a.name, b.name, m.name], "bb_ticks": bb.queries, "k_states": k_f, "pac_samples": bb.pac_samples}
                                probe_obs.append((cex, bb.eval_seq(cex)))
            elif fam == "RECURRENT_MUX_FB" and ni == 1 and no == 1:
                small11 = [s for s in s11 if s.ns <= 2]
                for a in small11:
                    for b in small11:
                        if a is b or a.ns + b.ns > 4: continue
                        def ev_mf(seq, Ta=a.quotient_T, Tb=b.quotient_T):
                            sa = sb = d = 0; out = []
                            for u0 in seq: sa, oa = Ta[sa][u0]; sb, ob = Tb[sb][u0]; y = ob[0] if d else oa[0]; d = y; out.append((y,))
                            return tuple(out)
                        if all(ev_mf(seq) == out for seq, out in probe_obs):
                            _, _, Tmux = compose_quotient("MUX_2IN", a, b)
                            k_fb, _, Tfb = compose_quotient("DELAYED_FEEDBACK", Skill("__m", a.ns + b.ns, 2, 1, 0, None, None, {}, "", "", (), 0, "", 0, 0, "", Tmux))
                            ok, cex = verify_T(k_fb, Tfb)
                            if ok:
                                self.model.record_family_hit(fam)
                                mux_sk = self.compose("MUX_2IN", a, b, f"{name}_mux2", route_override="GOAL_MUX2IN_INTERMEDIATE", generation=generation)
                                solved = self.compose("DELAYED_FEEDBACK", mux_sk, None, name, route_override="GOAL_RECURRENT_MUX_FB_SOLVED", extra_queries=bb.queries, generation=generation)
                                return solved, {"solved_by": "COMPOSITION(RECURRENT_MUX_FB)" + ("+PAC" if bb.pac_mode else ""), "parents": [a.name, b.name], "bb_ticks": bb.queries, "k_states": k_fb, "pac_samples": bb.pac_samples}
                            probe_obs.append((cex, bb.eval_seq(cex)))
            elif fam == "MOE_MUX" and ni == 1 and no == 1:
                g_vec, mask_all = build_probe_bitvec(probe_obs); sv = [skill_probe_vec(s.quotient_T, probe_obs) for s in s11]
                for i_s, sel in enumerate(s11):
                    if sel.ns > 3: continue
                    vs, nvs = sv[i_s], sv[i_s] ^ mask_all
                    va = [i for i in range(len(s11)) if i != i_s and s11[i].ns <= 3 and ((sv[i] ^ g_vec) & nvs) == 0]
                    vb = [j for j in range(len(s11)) if j != i_s and s11[j].ns <= 3 and ((sv[j] ^ g_vec) & vs) == 0] if va else []
                    for i_a in va:
                        for i_b in vb:
                            if i_a != i_b and sel.ns + s11[i_a].ns + s11[i_b].ns <= 8 and (res := try_commit("MOE_MUX", (sel, s11[i_a], s11[i_b]), fam_tag=fam)) is not None: return res
            elif fam == "DEPTH2_CASCADE" and ni == 1 and no == 1:
                small = [s for s in s11 if s.ns <= 2]
                for a in small:
                    for b in small:
                        if a is b or a.ns + b.ns > 4: continue
                        for c in small:
                            if a.ns + b.ns + c.ns > 6: continue
                            def ev_abc(seq, Ta=a.quotient_T, Tb=b.quotient_T, Tc=c.quotient_T):
                                sa = sb = sc = 0; out = []
                                for u in seq: sa, o1 = Ta[sa][u]; sb, o2 = Tb[sb][o1[0]]; sc, o3 = Tc[sc][o2[0]]; out.append(o3)
                                return tuple(out)
                            if all(ev_abc(seq) == out for seq, out in probe_obs):
                                _, _, Tab = compose_quotient("CASCADE", a, b)
                                k_abc, _, Tabc = compose_quotient("CASCADE", Skill("__t", a.ns + b.ns, 1, 1, 0, None, None, {}, "", "", (), 0, "", 0, 0, "", Tab), c)
                                ok, cex = verify_T(k_abc, Tabc)
                                if ok:
                                    self.model.record_family_hit(fam)
                                    mid = self.compose("CASCADE", a, b, f"{name}_mid", route_override="GOAL_DEPTH2_INTERMEDIATE", generation=generation)
                                    solved = self.compose("CASCADE", mid, c, name, route_override="GOAL_DEPTH2_COMPOSITION_SOLVED", extra_queries=bb.queries, generation=generation)
                                    return solved, {"solved_by": "COMPOSITION(DEPTH2_CASCADE)" + ("+PAC" if bb.pac_mode else ""), "parents": [a.name, b.name, c.name], "bb_ticks": bb.queries, "k_states": k_abc, "pac_samples": bb.pac_samples}
                                probe_obs.append((cex, bb.eval_seq(cex)))
            return None

        default_order = ["PARALLEL_PAIR", "DELAYED_FEEDBACK", "COUPLED_FEEDBACK_2BIT", "CASCADE", "PARALLEL_LUT", "PAIR2_TO_2IN", "RECURRENT_MUX_FB", "MOE_MUX", "DEPTH2_CASCADE"]
        for fam in self.model.ranked_families(default_order):
            if (res := run_family(fam)) is not None: return res

        k_l, T_l = lstar_learn_mealy(bb, max_states=max_states_hint, use_adt=self.use_adt)
        self.model.record_family_hit("LSTAR_FALLBACK")
        if bb.pac_mode: self.model.pac_lstar_solves += 1
        ns_l, rows_l = quotient_to_rows(k_l, ni, no, T_l); lstar_route = "GOAL_PAC_LSTAR_LEARNED" if bb.pac_mode else "GOAL_LSTAR_LEARNED"
        learned = self.build_skill_from_rows(name, ns_l, ni, no, rows_l, lstar_route, "LSTAR_BLACKBOX", (), bb.queries, True, generation)
        res, added = self.lib.add(learned)
        self.trace.append({"name": name, "op": "LSTAR_BLACKBOX", "route": lstar_route, "queries": bb.queries, "audit_rows": len(rows_l),
                           "action": "COMMIT" if added else "REUSE", "parents": [], "nvars": learned.nvars, "expert": res,
                           "k_states": k_l, "chosen_gates": len(learned.net["gates"]), "global_gates_after": len(self.lib.dag.gates), "generation": generation})
        return self.lib.skills[res], {"solved_by": "ACTIVE_STOCHASTIC_LSTAR_PAC" if bb.pac_mode else "ACTIVE_LSTAR_WP_FALLBACK",
                                      "bb_ticks": bb.queries, "k_states": k_l, "pac_samples": bb.pac_samples}

    def discover_novel(self, max_new: int = 8, seed: int = 0, generation: int = 2) -> list[Skill]:
        rng = random.Random(seed)
        eligible = [s for s in self.lib.skills.values() if s.ni == 1 and s.no == 1 and s.quotient_T is not None and s.ns <= 5]
        pool = []
        for a in eligible:
            for b in eligible:
                if a.ns + b.ns <= 8:
                    pool.append(("CASCADE", a, b, 0))
                    if a.name < b.name and a.ns + b.ns <= 7:
                        for l4 in (6, 8, 9): pool.append(("PARALLEL_LUT", a, b, l4))
        rng.shuffle(pool); committed: list[Skill] = []
        for op, a, b, l4 in pool:
            if len(committed) >= max_new: break
            k_q, bdig_q, _ = compose_quotient(op, a, b, lut4=l4)
            if 2 <= k_q <= 64 and bdig_q not in self.lib.by_behav:
                committed.append(self.compose(op, a, b, f"novel_{generation}_{len(committed):02d}", lut4=l4, generation=generation))
        return committed

    def repair_skill_netlist(self, skill_name: str, corrupted_net: dict) -> tuple[dict, list[int], int]:
        sk = self.lib.skills[self.lib.resolve(skill_name)]
        if sk.anf is None or sk.rows is None: raise ValueError("surgical cone repair requires auditable skill")
        return surgical_repair_netlist(corrupted_net, sk.anf, sk.rows, self.learner, self.issuer)

    def save_checkpoint(self, path: str | Path, meta: dict | None = None) -> dict:
        ok, errs = self.lib.dag.verify(self.lib.skills)
        if not ok: raise RuntimeError(f"global DAG invalid: {errs[:3]}")
        payload = {"version": "4.0", "issuer": self.issuer, "use_adt": self.use_adt, "use_factored_synthesis": self.use_factored_synthesis,
                   "meta": meta or {}, "skills": {k: v.export() for k, v in self.lib.skills.items()}, "aliases": self.lib.aliases,
                   "cache": self.lib.cache, "by_behav": self.lib.by_behav, "self_model": self.model.export(), "trace": self.trace, "dag": self.lib.dag.export()}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "AutonomousTemporalAgent":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(issuer=d.get("issuer", DEFAULT_ISSUER), min_warmup_trials=d["self_model"]["min_trials"],
                    use_adt=d.get("use_adt", True), use_factored_synthesis=d.get("use_factored_synthesis", True))
        for v in d["skills"].values():
            qT = tuple(tuple((x[0], tuple(x[1])) for x in row) for row in v["quotient_T"]) if v["quotient_T"] is not None else None
            rows = [tuple(x) for x in v["rows"]] if v["rows"] is not None else None
            anf = tuple(frozenset(x) for x in v["anf"]) if v["anf"] is not None else None
            agent.lib.add(Skill(v["name"], v["ns"], v["ni"], v["no"], v["reset"], rows, anf, v["net"], v["route"], v["op"],
                                tuple(v["parents"]), v["queries"], v["compile_mode"], v["k_states"], v["ns_min"],
                                v["behav_digest"], qT, v["bisim_verified"], v.get("ag_cert"), v.get("generation", 0)))
        agent.lib.aliases = dict(d["aliases"]); agent.lib.cache = dict(d["cache"]); agent.lib.by_behav = dict(d["by_behav"])
        sm = d["self_model"]; agent.model.trials, agent.model.exact, agent.model.enabled = sm["trials"], sm["exact"], sm["enabled"]
        agent.model.history, agent.model.epistemic_alerts = list(sm["history"]), list(sm["epistemic_alerts"])
        agent.model.mdl_predictions = list(sm.get("mdl_predictions", [])); agent.model.family_hits = dict(sm.get("family_hits", {}))
        agent.model.adt_verifications = sm.get("adt_verifications", 0); agent.model.adt_baseline_wp_ticks = sm.get("adt_baseline_wp_ticks", 0)
        agent.model.adt_actual_ticks = sm.get("adt_actual_ticks", 0); agent.model.pac_lstar_solves = sm.get("pac_lstar_solves", 0)
        agent.model.sop_baseline_gates = sm.get("sop_baseline_gates", 0); agent.model.opt_chosen_gates = sm.get("opt_chosen_gates", 0)
        agent.trace = list(d.get("trace", []))
        if agent.lib.dag.export()["gates"] != d["dag"]["gates"] or agent.lib.dag.export()["outputs"] != d["dag"]["outputs"]:
            raise RuntimeError("checkpoint reconstruction mismatch")
        return agent

    def status(self) -> dict:
        ok, errs = self.lib.dag.verify(self.lib.skills)
        return {"version": "4.0", "unique_behavioral_skills": len(self.lib.skills), "aliases": len(self.lib.aliases),
                "multi_output_skills_no2": sum(1 for s in self.lib.skills.values() if s.no >= 2),
                "ag_compacted_skills": sum(1 for s in self.lib.skills.values() if s.compile_mode == "MODULAR_AG_COMPACT"),
                "max_nvars": max((s.nvars for s in self.lib.skills.values()), default=0),
                "global_dag": self.lib.dag.metrics(), "global_dag_exact": ok, "dag_errors": errs, "self_model": self.model.export()}


# ---------------------------------------------------------------------------
# 7. Zero-Import Replay Verifier, Built-In Self-Test & CLI
# ---------------------------------------------------------------------------

def zero_import_replay_verify(archive_path: str | Path) -> dict:
    """Standalone replay verifier: validates Global DAG == Local Netlist == Quotient_T & AG tamper rejection."""
    d = json.loads(Path(archive_path).read_text(encoding="utf-8"))
    dag, skills = d["dag"], d["skills"]
    def ev_net(gates, outs, inp):
        w = list(inp)
        for g in gates:
            m = g["mask"]
            w.append(0 if m == 0 else (1 if m == 15 else (w[g["a"]] ^ w[g["b"]] if m == 6 else w[g["a"]] & w[g["b"]])))
        return tuple(w[o] for o in outs)
    rng = random.Random(0xCAFE); rows_checked = ticks_checked = ag_count = 0
    for nm, sk in skills.items():
        ns, ni, net, dag_outs = sk["ns"], sk["ni"], sk["net"], dag["outputs"][nm]
        if sk["anf"] is not None:
            anf = sk["anf"]
            for x in range(1 << (ns + ni)):
                inp_l = tuple((x >> i) & 1 for i in range(ns + ni))
                inp_d = [0] * GLOBAL_N
                for i in range(ns): inp_d[i] = (x >> i) & 1
                for j in range(ni): inp_d[22 + j] = (x >> (ns + j)) & 1
                assert ev_net(net["gates"], net["outs"], inp_l) == ev_net(dag["gates"], dag_outs, inp_d) == tuple(sum(1 for t in ts if (t & x) == t) & 1 for ts in anf)
                rows_checked += 1
            qT = sk["quotient_T"]; s_d = s_q = 0
            for _ in range(256):
                u = rng.randrange(1 << ni); inp_d = [0] * GLOBAL_N
                for i in range(ns): inp_d[i] = (s_d >> i) & 1
                for j in range(ni): inp_d[22 + j] = (u >> j) & 1
                yd = ev_net(dag["gates"], dag_outs, inp_d); s_d = sum(v << i for i, v in enumerate(yd[:ns])); s_q, o_q = qT[s_q][u]
                assert yd[ns:] == tuple(o_q); ticks_checked += 1
        else:
            ag_count += 1; s_l = s_d = 0
            for _ in range(256):
                u = rng.randrange(1 << ni); inp_d = [0] * GLOBAL_N
                for i in range(ns): inp_d[i] = (s_d >> i) & 1
                for j in range(ni): inp_d[22 + j] = (u >> j) & 1
                yl = ev_net(net["gates"], net["outs"], tuple([(s_l >> i) & 1 for i in range(ns)] + [(u >> j) & 1 for j in range(ni)]))
                yd = ev_net(dag["gates"], dag_outs, inp_d); assert yl == yd
                s_l = sum(v << i for i, v in enumerate(yl[:ns])); s_d = sum(v << i for i, v in enumerate(yd[:ns])); ticks_checked += 1
    tamper_rejected = False
    if dag["ag_skill_names"]:
        t_nm = dag["ag_skill_names"][0]; bad_outs = list(dag["outputs"][t_nm]); bad_outs[-1] = 22 if bad_outs[-1] != 22 else 0
        for _ in range(16):
            inp_d = tuple(rng.randrange(2) for _ in range(GLOBAL_N))
            if ev_net(dag["gates"], dag["outputs"][t_nm], inp_d) != ev_net(dag["gates"], bad_outs, inp_d): tamper_rejected = True; break
    return {"archive": str(archive_path), "skills_verified": len(skills), "ag_skills": ag_count, "dag_gates": len(dag["gates"]),
            "rows_checked": rows_checked, "ticks_checked": ticks_checked, "tamper_rejected": tamper_rejected, "pass": True}

def run_self_verify(seed: int = 20261127, wide_bits: int = 3, out_dir: str | Path | None = None) -> dict:
    t0 = time.perf_counter(); rng = random.Random(seed)
    o_par = lambda x: ((x & 1) ^ ((x >> 1) & 1),) * 2
    o_edg = lambda x: ((x >> 1) & 1, ((x >> 1) & 1) & (1 ^ (x & 1)))
    o_sh3 = lambda x: ((((x & 7) << 1) | ((x >> 3) & 1)) & 1, ((((x & 7) << 1) | ((x >> 3) & 1)) >> 1) & 1, ((((x & 7) << 1) | ((x >> 3) & 1)) >> 2) & 1, (x >> 2) & 1)
    o_wc = lambda x, m=(1 << wide_bits) - 1: tuple([(((x & m) + ((x >> wide_bits) & 1)) & m) >> i & 1 for i in range(wide_bits)] + [int((x & m) == m and ((x >> wide_bits) & 1) == 1)])
    o_101 = lambda x: (lambda s, u: (lambda ns, y: (ns & 1, (ns >> 1) & 1, y))(*((1 if u else 0, 0) if s == 0 else ((2 if u == 0 else 1, 0) if s == 1 else ((3 if u else 0, int(u == 1)) if s == 2 else (1 if u else 2, 0))))))(x & 3, (x >> 2) & 1)
    o_rc = lambda x: (lambda ns: (ns & 1, (ns >> 1) & 1, int(ns == 3)))(0 if (x >> 3) & 1 else min(3, (x & 3) + ((x >> 2) & 1)))
    o_gp = lambda x: ((x & 1) ^ (((x >> 1) & 1) & ((x >> 2) & 1)),) * 2
    def mk_mod(km, p=0.0, sd=0):
        r = random.Random(sd)
        return InteractiveMealyOracle(1, 1, lambda seq: _ev_mod(seq, km, p, r))
    def _ev_mod(seq, km, p, r):
        s = 0; out = []
        for u in seq: s = (s + u) % km; y = 1 if (s == 0 and u == 1) else 0; out.append((y ^ (1 if p > 0 and r.random() < p else 0),))
        return tuple(out)
    def mk_mod_ni2(km, p=0.0, sd=0):
        r = random.Random(sd)
        def fn(seq):
            s = 0; out = []
            for u in seq:
                if (u >> 1) & 1: s, y = 0, 0
                elif u & 1: y = 1 if s == km - 1 else 0; s = (s + 1) % km
                else: y = 0
                out.append((y ^ (1 if p > 0 and r.random() < p else 0),))
            return tuple(out)
        return InteractiveMealyOracle(2, 1, fn)
    def mk_noisy(sk, p, sd):
        r = random.Random(sd); rows, ns = sk.rows, sk.ns
        def fn(seq):
            s = 0; out = []
            for u in seq: row = rows[s | (u << ns)]; s = sum(v << i for i, v in enumerate(row[:ns])); out.append(tuple(v ^ (1 if r.random() < p else 0) for v in row[ns:]))
            return tuple(out)
        return InteractiveMealyOracle(sk.ni, sk.no, fn)

    ref = AutonomousTemporalAgent()
    pa = ref.learn_primitive("parity", 1, 1, 1, o_par, 1); ed = ref.learn_primitive("rising_edge", 1, 1, 1, o_edg, 2)
    sh = ref.learn_primitive("shift3", 3, 1, 1, o_sh3, 1); s101 = ref.learn_primitive("seq101", 2, 1, 1, o_101, 3)
    rc = ref.learn_primitive("reset_counter", 2, 2, 1, o_rc, 4); gp = ref.learn_primitive("gated_parity", 1, 2, 1, o_gp, 2)
    r_s101_pa = ref.compose("CASCADE", s101, pa, "r1"); r_re101 = ref.compose("CASCADE", ed, s101, "r2"); r_pa101 = ref.compose("CASCADE", pa, s101, "r3")
    r_xor = ref.compose("PARALLEL_LUT", ed, pa, "r4", lut4=6); r_pp = ref.compose("PARALLEL_PAIR", ed, pa, "r5")
    r_p2gp = ref.compose("CASCADE", r_pp, gp, "r6"); r_dfb = ref.compose("DELAYED_FEEDBACK", rc, None, "r7")
    r_cfb = ref.compose("COUPLED_FEEDBACK_2BIT", rc, gp, "r8"); r_rmux = ref.compose("DELAYED_FEEDBACK", ref.compose("MUX_2IN", ed, pa, "r9m"), None, "r9")
    r_moe = ref.compose("MOE_MUX", pa, ed, "r10", pc=sh); r_d2 = ref.compose("CASCADE", r_re101, pa, "r11")
    r_g3c = ref.compose("CASCADE", r_cfb, pa, "r12")
    m5, _ = ref.solve_goal("m5", mk_mod(5), 8); m6, _ = ref.solve_goal("m6", mk_mod(6), 8)
    m7, _ = ref.solve_goal("m7", mk_mod(7), 8); m5n2, _ = ref.solve_goal("m5n2", mk_mod_ni2(5), 8)
    r_g3m7 = ref.compose("CASCADE", m7, pa, "r13")

    ag = AutonomousTemporalAgent()
    pa_g = ag.learn_primitive("parity", 1, 1, 1, o_par, 1); ed_g = ag.learn_primitive("rising_edge", 1, 1, 1, o_edg, 2)
    sh_g = ag.learn_primitive("shift3", 3, 1, 1, o_sh3, 1); _ = ag.learn_primitive("wide_counter", wide_bits, 1, 1, o_wc, wide_bits + 1)
    _ = ag.learn_primitive("seq101", 2, 1, 1, o_101, 3); rc_g = ag.learn_primitive("reset_counter", 2, 2, 1, o_rc, 4)
    _ = ag.learn_primitive("gated_parity", 1, 2, 1, o_gp, 2); _ = ag.learn_primitive("seq101_false", 2, 1, 1, o_101, 1)
    assert ag.learn_primitive("untrusted", 1, 1, 1, o_par, contract=DegreeContract(2, 1, "BAD", 1), allow_full_support=False) is None
    w1 = ag.compose("CASCADE", ed_g, pa_g, "warm1"); _ = ag.compose("CASCADE", pa_g, ed_g, "warm2"); _ = ag.compose("CASCADE", sh_g, pa_g, "warm3")
    assert ag.model.enabled

    bad = {"n_inputs": rc_g.net["n_inputs"], "gates": [dict(g) for g in rc_g.net["gates"]], "outs": list(rc_g.net["outs"])}
    bad["gates"].append({"a": bad["outs"][2], "b": 0, "mask": 6}); bad["outs"][2] = bad["n_inputs"] + len(bad["gates"]) - 1
    rep, bw, _ = ag.repair_skill_netlist("reset_counter", bad); assert bw == [2] and symbolic_verify_general(rep, rc_g.anf)

    goals = [
        ("bb_goal_library_reuse", InteractiveMealyOracle.from_rows(w1.ns, w1.ni, w1.no, w1.rows), 8, w1.behav_digest),
        ("bb_goal_cascade", InteractiveMealyOracle.from_rows(r_s101_pa.ns, r_s101_pa.ni, r_s101_pa.no, r_s101_pa.rows), 12, r_s101_pa.behav_digest),
        ("bb_goal_parallel_xor", InteractiveMealyOracle.from_rows(r_xor.ns, r_xor.ni, r_xor.no, r_xor.rows), 8, r_xor.behav_digest),
        ("bb_goal_parallel_pair_no2", InteractiveMealyOracle.from_rows(r_pp.ns, r_pp.ni, r_pp.no, r_pp.rows), 8, r_pp.behav_digest),
        ("bb_goal_pair2_to_2in", InteractiveMealyOracle.from_rows(r_p2gp.ns, r_p2gp.ni, r_p2gp.no, r_p2gp.rows), 12, r_p2gp.behav_digest),
        ("bb_goal_delayed_feedback", InteractiveMealyOracle.from_rows(r_dfb.ns, r_dfb.ni, r_dfb.no, r_dfb.rows), 8, r_dfb.behav_digest),
        ("bb_goal_coupled_feedback_2bit", InteractiveMealyOracle.from_rows(r_cfb.ns, r_cfb.ni, r_cfb.no, r_cfb.rows), 12, r_cfb.behav_digest),
        ("bb_goal_recurrent_mux_fb", InteractiveMealyOracle.from_rows(r_rmux.ns, r_rmux.ni, r_rmux.no, r_rmux.rows), 12, r_rmux.behav_digest),
        ("bb_goal_temporal_moe_mux", InteractiveMealyOracle.from_rows(r_moe.ns, r_moe.ni, r_moe.no, r_moe.rows), 20, r_moe.behav_digest),
        ("bb_goal_depth2_cascade", InteractiveMealyOracle.from_rows(r_d2.ns, r_d2.ni, r_d2.no, r_d2.rows), 12, r_d2.behav_digest),
        ("bb_goal_noncomposable_mod5", mk_mod(5, 0.0, seed + 1), 8, m5.behav_digest),
        ("bb_goal_noisy_5pct_pac", mk_noisy(r_re101, 0.05, seed + 2), 8, r_re101.behav_digest),
        ("bb_goal_noisy_10pct_pac", mk_noisy(r_pa101, 0.10, seed + 3), 8, r_pa101.behav_digest),
        ("bb_goal_noisy_noncomp_mod7_10pct", mk_mod(7, 0.10, seed + 4), 12, m7.behav_digest),
        ("bb_goal_noisy_noncomp_mod6_15pct", mk_mod(6, 0.15, seed + 5), 10, m6.behav_digest),
        ("bb_goal_noisy_noncomp_mod5_ni2_10pct", mk_mod_ni2(5, 0.10, seed + 6), 6, m5n2.behav_digest),
    ]
    g_res = {}
    for nm, bb_g, hint, exp_bd in goals:
        sk_g, inf_g = ag.solve_goal(nm, bb_g, max_states_hint=hint, generation=1)
        assert sk_g is not None and sk_g.behav_digest == exp_bd, (nm, inf_g)
        g_res[nm] = inf_g
    sk_ch, inf_ch = ag.solve_goal("bb_chaos45", mk_noisy(r_re101, 0.45, seed + 7), 8, generation=1)
    assert sk_ch is None and inf_ch["solved_by"] == "HOLD_UNRECOVERABLE_NOISE"

    import tempfile
    tmp_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="hagi_v40_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cp1 = tmp_dir / f"cp1_{seed}.json"; ag.save_checkpoint(cp1); ag = AutonomousTemporalAgent.load_checkpoint(cp1)
    ag.discover_novel(max_new=18, seed=seed, generation=2)
    pa2, ed2, sh2 = ag.lib.skills["parity"], ag.lib.skills["rising_edge"], ag.lib.skills["shift3"]
    ag.compose("MUX_2IN", ed2, pa2, "gen2_mux2", generation=2)
    for l4 in (8, 9, 14): ag.compose("PARALLEL_LUT", ed2, sh2, f"gen2_plut_{l4}", lut4=l4, generation=2)
    c6 = ag.compose("CASCADE", sh2, sh2, "deep_c6", generation=2); c9 = ag.compose("CASCADE", c6, sh2, "deep_c9", generation=2)
    c12 = ag.compose("CASCADE", c9, sh2, "deep_c12", generation=2); c15 = ag.compose("CASCADE", c12, sh2, "deep_c15_ag", generation=2)
    c18 = ag.compose("CASCADE", c15, sh2, "deep_c18_ag", generation=2); c21 = ag.compose("CASCADE", c18, sh2, "deep_c21_ag", generation=2)
    ag.compose("MOE_MUX", pa2, c6, "ag_moe_wide", pc=c6, generation=2); assert c21.nvars == 22

    cp2 = tmp_dir / f"cp2_{seed}.json"; ag.save_checkpoint(cp2); ag = AutonomousTemporalAgent.load_checkpoint(cp2)
    sk_g3c, inf_g3c = ag.solve_goal("bb_goal_gen3_coupled_transfer", InteractiveMealyOracle.from_rows(r_g3c.ns, r_g3c.ni, r_g3c.no, r_g3c.rows), 16, generation=3)
    sk_g3m, inf_g3m = ag.solve_goal("bb_goal_gen3_mod7_transfer", InteractiveMealyOracle.from_rows(r_g3m7.ns, r_g3m7.ni, r_g3m7.no, r_g3m7.rows), 18, generation=3)
    assert sk_g3c.behav_digest == r_g3c.behav_digest and sk_g3m.behav_digest == r_g3m7.behav_digest
    g_res["bb_goal_gen3_coupled_transfer"] = inf_g3c; g_res["bb_goal_gen3_mod7_transfer"] = inf_g3m
    ag.compose("PARALLEL_LUT", c15, c6, "ag_par_xor_22", lut4=6, generation=3)

    cp3 = tmp_dir / f"cp3_{seed}.json"; ag.save_checkpoint(cp3); ag = AutonomousTemporalAgent.load_checkpoint(cp3)
    rep = zero_import_replay_verify(cp3); st = ag.status()
    return {"seed": seed, "elapsed_sec": round(time.perf_counter() - t0, 3), "status": st, "replay": rep, "goals": g_res}

def main() -> None:
    ap = argparse.ArgumentParser(description="HAGI Fusion v4.0 Single-File Universal Temporal Skill Engine")
    ap.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint JSON and print status")
    ap.add_argument("--replay", type=str, default=None, help="Run zero-import replay verification on checkpoint JSON")
    ap.add_argument("--verify", action="store_true", help="Run full self-verification suite (18 goals + AG22 + PAC-L* + 3 checkpoints)")
    ap.add_argument("--seed", type=int, default=20261127, help="Seed for --verify")
    args = ap.parse_args()
    if args.replay: print(json.dumps(zero_import_replay_verify(args.replay), indent=2)); return
    if args.verify: print(json.dumps(run_self_verify(seed=args.seed), indent=2)); return
    agent = AutonomousTemporalAgent.load_checkpoint(args.checkpoint) if args.checkpoint else AutonomousTemporalAgent()
    print(json.dumps(agent.status(), indent=2))

if __name__ == "__main__":
    main()
