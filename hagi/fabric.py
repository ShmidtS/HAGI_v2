#!/usr/bin/env python3
"""Venn fabric — the self-assembling processor substrate.

Genome: k (inputs per gate), W (bit lanes sharing masks), depth,
wiring + 4-bit Venn masks per gate. Masks ARE the program.
Emulator: bitwise ops on W-wide lanes (mask shared across lanes —
weight sharing).

Certified-first doctrine (proved necessary by the full-adder demo):
GROW phase — correctness only + neutral drift (cost pressure before
correctness collapses search into a 1-gate local optimum); OPTIMIZE
phase — correctness is a hard gate, cost monotonically down.

Theory anchors (verified in earlier cycles): BoolVenn lattice
(16 opcodes; implication oracle f⇒g ⟺ mask_f & ~mask_g == 0;
NPN 16→4 essential classes = Zhegalkin basis {1, x, ∧, ⊕}),
CertifiedArgmax strict gap, GatePipeline budget, MDL bits.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FABRIC_LOG = ROOT / "checkpoints" / "self" / "fabric.json"

# --- 16 Venn opcodes as 4-bit masks, order (++),(+-),(-+),(--) in bits
# 3,2,1,0 (bit 3 = value at (T,T), bit 0 = (F,F)) ---
MASKS = {
    "FALSE": 0b0000, "AND": 0b1000, "A_GT_B": 0b0100, "A": 0b1010,
    "B_GT_A": 0b0010, "B": 0b1100, "XOR": 0b0110, "OR": 0b1110,
    "NOR": 0b0001, "XNOR": 0b1001, "NOT_B": 0b0011, "B_LE_A": 0b1011,
    "NOT_A": 0b0101, "A_LE_B": 0b1101, "NAND": 0b0111, "TRUE": 0b1111,
}
ALL_MASKS = dict(MASKS)
NAME_OF = {v: k for k, v in MASKS.items()}


def eval_mask(mask: int, a: bool, b: bool) -> bool:
    idx = 0 if (a and b) else (1 if (a and not b) else
                               (2 if ((not a) and b) else 3))
    return bool((mask >> (3 - idx)) & 1)


def imply(mf: int, mg: int) -> bool:
    """O(1) lattice implication: f⇒g ⟺ mf & ~mg == 0."""
    return (mf & ~mg) == 0


def npn_canonical(mask: int) -> int:
    """Minimal mask over NPN group (swap A/B, input inversions,
    output inversion)."""
    best = mask
    for swap in (False, True):
        for ia in (False, True):
            for ib in (False, True):
                for inv in (False, True):
                    m = mask ^ (0b1111 if inv else 0)
                    new = 0
                    for idx in range(4):
                        a, b = [(True, True), (True, False),
                                (False, True), (False, False)][idx]
                        if ia:
                            a = not a
                        if ib:
                            b = not b
                        if swap:
                            a, b = b, a
                        if eval_mask(m, a, b):
                            new |= 1 << (3 - idx)
                    best = min(best, new)
    return best


# --- Fabric genome ---

def random_genome(rng: random.Random, W: int = 8, n_inputs: int = 2,
                  n_outputs: int = 1) -> dict:
    gates = []
    n_gates = rng.randint(4, 10)
    for i in range(n_gates):
        srcs = n_inputs + i
        gates.append({"a": rng.randrange(srcs), "b": rng.randrange(srcs),
                      "mask": rng.choice(list(ALL_MASKS.values()))})
    outs = [rng.randrange(n_inputs + n_gates) for _ in range(n_outputs)]
    return {"W": W, "n_inputs": n_inputs, "gates": gates, "outs": outs}


def program_bits(genome: dict) -> int:
    n_in = genome["n_inputs"]
    bits_per_gate = 4 + 2 * max(1, n_in + len(genome["gates"]) - 1).bit_length()
    return len(genome["gates"]) * bits_per_gate + \
        len(genome["outs"]) * max(1, n_in + len(genome["gates"])).bit_length()


def emulate(genome: dict, inputs: list[int]) -> list[int]:
    """Bitwise: inputs are W-bit buses; each gate applies its Venn mask
    per bit-lane (mask shared across lanes — weight sharing)."""
    W = genome["W"]
    wires = list(inputs[:genome["n_inputs"]])
    for g in genome["gates"]:
        a = wires[g["a"]] if g["a"] < len(wires) else 0
        b = wires[g["b"]] if g["b"] < len(wires) else 0
        m = g["mask"]
        res = 0
        for bit in range(W):
            ab = (a >> bit) & 1
            bb = (b >> bit) & 1
            idx = (ab << 1) | bb          # bit3=(T,T) ... bit0=(F,F)
            res |= ((m >> idx) & 1) << bit
        wires.append(res)
    n = len(wires)
    return [wires[o] if 0 <= o < n else wires[-1] for o in genome["outs"]]


def certify(genome: dict, truth: dict, inputs_from: callable):
    """Hard gate: exhaustive truth-table check."""
    for key, want in truth.items():
        got = emulate(genome, inputs_from(key))
        if got != want:
            return False, f"mismatch {key}"
    return True, "ok"


def correct_fraction(genome: dict, truth: dict, inputs_from: callable) -> float:
    """Soft search signal (pre-certification hill-climb only).
    The COMMIT gate remains exact certify()."""
    n = ok = 0
    for key, want in truth.items():
        n += 1
        if emulate(genome, inputs_from(key)) == want:
            ok += 1
    return ok / n if n else 0.0


def cost(genome: dict) -> float:
    """OPTIMIZE objective: MDL bits + 4×depth."""
    n_in = genome["n_inputs"]
    depth = {}
    for i, g in enumerate(genome["gates"]):
        gi = n_in + i
        da = depth.get(g["a"], 0) if g["a"] >= n_in else 0
        db = depth.get(g["b"], 0) if g["b"] >= n_in else 0
        depth[gi] = 1 + max(da, db)
    return program_bits(genome) + 4 * max(depth.values(), default=0)


# --- Certified moves ---

def mutate(genome: dict, rng: random.Random) -> dict:
    g = {"W": genome["W"], "n_inputs": genome["n_inputs"],
         "outs": list(genome["outs"]),
         "gates": [dict(x) for x in genome["gates"]]}
    n_in = g["n_inputs"]
    ng = len(g["gates"])
    move = rng.choice(["mask", "rewire", "insert", "delete", "out",
                       "xor_pair"])
    if move == "mask" and ng:
        g["gates"][rng.randrange(ng)]["mask"] = \
            rng.choice(list(ALL_MASKS.values()))
    elif move == "rewire" and ng:
        i = rng.randrange(ng)
        g["gates"][i][("a", "b")[rng.randrange(2)]] = \
            rng.randrange(n_in + ng)
    elif move == "insert" and ng < 24:
        srcs = n_in + ng
        g["gates"].insert(rng.randrange(ng + 1),
                          {"a": rng.randrange(srcs),
                           "b": rng.randrange(srcs),
                           "mask": rng.choice(list(ALL_MASKS.values()))})
    elif move == "delete" and ng > 2:
        i = rng.randrange(ng)
        victim = g["gates"].pop(i)
        repl = victim["a"]
        for other in g["gates"]:
            for s in ("a", "b"):
                if other[s] == n_in + i:
                    other[s] = repl
                elif other[s] > n_in + i:
                    other[s] -= 1
        g["outs"] = [o if o != n_in + i else repl for o in g["outs"]]
        g["outs"] = [o - 1 if o > n_in + i else o for o in g["outs"]]
    elif move == "out" and ng:
        g["outs"][rng.randrange(len(g["outs"]))] = rng.randrange(n_in + ng)
    elif move == "xor_pair" and ng < 22:
        # Zhegalkin lemma: append a⊕b gate pair on random wires
        srcs = n_in + ng
        g["gates"].append({"a": rng.randrange(srcs),
                           "b": rng.randrange(srcs),
                           "mask": MASKS["XOR"]})
    return g


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    cut = rng.randrange(1, len(a["gates"]) + 1)
    gates = [dict(x) for x in a["gates"][:cut]] + \
            [dict(x) for x in b["gates"][cut:]]
    n_in = a["n_inputs"]
    for i, gg in enumerate(gates):
        lim = n_in + i
        gg["a"] = min(gg["a"], lim)
        gg["b"] = min(gg["b"], lim)
    child = dict(a)
    child["gates"] = gates
    child["outs"] = [min(o, n_in + len(gates) - 1) for o in a["outs"]]
    return child


# --- Evolution: GROW then OPTIMIZE ---

def evolve(truth: dict, inputs_from: callable, n_inputs: int = 2,
           pop_size: int = 40, generations: int = 300, seed: int = 7,
           log: bool = True, profile: str = "adder",
           seed_cells: list[dict] | None = None) -> dict:
    rng = random.Random(seed)
    W = truth.get("_W", 8)
    table = {k: v for k, v in truth.items() if k != "_W"}
    n_out = len(next(iter(table.values())))
    pop = [random_genome(rng, W=W, n_inputs=n_inputs, n_outputs=n_out)
           for _ in range(pop_size)]
    # curriculum: seed certified cells into the initial population
    # (standard cell-library flow — cells carry their own exhaustive
    # certificates; the composition search starts from proven parts)
    if seed_cells:
        for sc in seed_cells[:max(1, pop_size // 4)]:
            g = {"W": W, "n_inputs": n_inputs,
                 "gates": [dict(x) for x in sc["gates"]],
                 "outs": list(sc.get("outs", []))}
            if g["outs"]:
                g["outs"] = [min(o, n_inputs + len(g["gates"]) - 1)
                             for o in g["outs"]]
            pop[0] = g
            break
    certified, best, grow_gen = None, None, None
    best_soft = [0.0]
    best_soft_gen = [0]
    history = []
    for gen in range(generations):
        scored = []
        soft = []
        for g in pop:
            ok, _ = certify(g, table, inputs_from)
            if ok:
                scored.append((cost(g), rng.random(), g))
            else:
                soft.append((correct_fraction(g, table, inputs_from),
                             g))
        cert_pop = sorted(scored, key=lambda t: (t[0], t[1]))
        if cert_pop and certified is None:
            certified = cert_pop[0][2]
            grow_gen = gen
            history.append({"phase": "GROW", "gen": gen,
                            "gates": len(certified["gates"])})
        if len(cert_pop) >= 2:
            elite = [g for _, _, g in cert_pop[:max(2, pop_size // 4)]]
            children = list(elite)
            while len(children) < pop_size:
                c = mutate(crossover(rng.choice(elite), rng.choice(elite),
                                     rng), rng)
                ok, _ = certify(c, table, inputs_from)
                if ok:
                    children.append(c)
            pop = children
            best_now = min(children, key=cost)
            if best is None or cost(best_now) < cost(best):
                best = best_now
                history.append({"phase": "OPTIMIZE", "gen": gen,
                                "gates": len(best["gates"]),
                                "bits": program_bits(best),
                                "cost": round(cost(best), 1)})
        else:
            # pre-certification: soft hill-climb toward correctness
            # (COMMIT gate stays exact certify())
            soft.sort(key=lambda t: -t[0])
            if soft and soft[0][0] > best_soft[0]:
                best_soft[0] = soft[0][0]
                best_soft_gen[0] = gen
            # plateau (200 gens w/o improvement) → diversity restart:
            # keep elite quarter, rest fresh immigrants
            plateau = gen - best_soft_gen[0] > 200
            elite = [g for _, g in soft[:max(2, pop_size // 4)]]
            children = list(elite)
            while len(children) < pop_size:
                if plateau and rng.random() < 0.5:
                    children.append(random_genome(
                        rng, W=W, n_inputs=n_inputs, n_outputs=n_out))
                else:
                    children.append(mutate(
                        crossover(rng.choice(elite), rng.choice(elite),
                                  rng), rng))
            pop = children
        if certified is not None and gen >= grow_gen + 80:
            break
    final = best or certified
    result = {"profile": profile, "seed": seed, "certified_gen": grow_gen,
              "best": final, "bits": program_bits(final) if final else None,
              "history": history,
              "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    if log:
        FABRIC_LOG.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(FABRIC_LOG.read_text(encoding="utf-8")) \
            if FABRIC_LOG.exists() else []
        prev.append({k: v for k, v in result.items()})
        FABRIC_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    return result


def render(genome: dict) -> str:
    lines = []
    n_in = genome["n_inputs"]
    for i, g in enumerate(genome["gates"]):
        gi = n_in + i
        def nm(x):
            return ("in" + "ABCDEF"[x]) if x < n_in else f"w{x}"
        lines.append(f"w{gi} = {NAME_OF[g['mask']]}({nm(g['a'])}, {nm(g['b'])})")
    for j, o in enumerate(genome["outs"]):
        lines.append(f"OUT{j} = w{o}")
    return "\n".join(lines)


def full_adder_profile(W: int = 8):
    """Full adder: A, B, Cin → SUM, COUT. Exhaustive 8 rows."""
    full = (1 << W) - 1
    truth = {}
    for a in (0, 1):
        for b in (0, 1):
            for cin in (0, 1):
                s = a ^ b ^ cin
                cout = (a & b) | (cin & (a ^ b))
                truth[(a, b, cin)] = [s * full, cout * full]

    def ins(key):
        return [key[0] * full, key[1] * full, key[2] * full]
    return truth, ins, 3


def main() -> None:
    import sys
    profile = sys.argv[1] if len(sys.argv) > 1 else "adder"
    if profile != "adder":
        raise SystemExit(f"unknown profile {profile!r}; only 'adder' now")
    truth, ins, n_in = full_adder_profile()
    res = evolve(truth, ins, n_inputs=n_in, profile=profile)
    print(json.dumps({k: v for k, v in res.items() if k != "best"},
                     indent=2, ensure_ascii=False))
    print(render(res["best"]))


if __name__ == "__main__":
    main()
