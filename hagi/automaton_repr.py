"""Bounded inference and factorization of two-coordinate Moore automata."""
from __future__ import annotations

import itertools
import json
from collections import deque
from dataclasses import dataclass
from typing import Callable, Sequence


def words_upto(alphabet: Sequence[int], depth: int):
    words = [()]
    for n in range(1, depth + 1):
        words.extend(itertools.product(alphabet, repeat=n))
    return words


def encode_trace(actions: Sequence[int], alphabet: Sequence[int]) -> int:
    alphabet = tuple(int(x) for x in alphabet)
    index = {a: i + 1 for i, a in enumerate(alphabet)}
    if len(index) != len(alphabet):
        raise ValueError("HOLD_DUPLICATE_ACTION")
    base, code, place = len(alphabet) + 1, 0, 1
    for action in actions:
        if int(action) not in index:
            raise ValueError("HOLD_UNKNOWN_ACTION")
        code += index[int(action)] * place
        place *= base
    return code


def decode_trace(code: int, alphabet: Sequence[int], max_length: int):
    if isinstance(code, bool) or not isinstance(code, int) or code < 0:
        raise ValueError("HOLD_INVALID_TRACE_CODE")
    alphabet = tuple(int(x) for x in alphabet)
    base = len(alphabet) + 1
    if max_length < 0 or code >= base ** max_length:
        raise ValueError("HOLD_TRACE_OUTSIDE_SUPPORT")
    out = []
    while code:
        digit, code = code % base, code // base
        if digit == 0:
            if code:
                raise ValueError("HOLD_NONCANONICAL_TRACE")
            break
        if digit > len(alphabet):
            raise ValueError("HOLD_UNKNOWN_TRACE_DIGIT")
        out.append(alphabet[digit - 1])
    return tuple(out)


@dataclass(frozen=True)
class LearnedMoore:
    alphabet: tuple[int, ...]
    representatives: tuple[tuple[int, ...], ...]
    outputs: tuple[int, ...]
    transitions: tuple[dict[int, int], ...]
    query_count: int


def learn_bounded_moore(membership: Callable, alphabet: Sequence[int],
                        max_states: int) -> LearnedMoore:
    """Finite-state observation-table learner under an explicit state cap."""
    alphabet = tuple(int(a) for a in alphabet)
    if not alphabet or len(set(alphabet)) != len(alphabet) or max_states < 1:
        raise ValueError("HOLD_INVALID_LEARNER_BOUNDS")
    suffixes = words_upto(alphabet, max_states - 1)
    labels = {}

    def query(word):
        key = tuple(word)
        if key not in labels:
            value = membership(key)
            if value is None:
                raise ValueError("HOLD_UNLABELED_ORACLE")
            labels[key] = int(value)
        return labels[key]

    def signature(prefix):
        return tuple(query(prefix + suffix) for suffix in suffixes)

    sig0 = signature(())
    sigs, reps, outs, trans = [sig0], [()], [sig0[0]], [{}]
    by_sig, queue = {sig0: 0}, deque([0])
    while queue:
        sid = queue.popleft()
        prefix = reps[sid]
        row = {}
        for action in alphabet:
            sig = signature(prefix + (action,))
            if sig not in by_sig:
                if len(sigs) >= max_states:
                    raise ValueError("HOLD_STATE_BOUND_EXCEEDED")
                nxt = len(sigs)
                by_sig[sig] = nxt
                sigs.append(sig)
                reps.append(prefix + (action,))
                outs.append(sig[0])
                trans.append({})
                queue.append(nxt)
            row[action] = by_sig[sig]
        trans[sid] = row
    return LearnedMoore(alphabet, tuple(reps), tuple(outs), tuple(trans), len(labels))


def _orbit_partition(model: LearnedMoore, action: int):
    n = len(model.outputs)
    targets = [model.transitions[s][action] for s in range(n)]
    if sorted(targets) != list(range(n)):
        raise ValueError("HOLD_NONPERMUTATION_TRANSITION")
    unseen, blocks = set(range(n)), []
    while unseen:
        seed, cur, cycle = min(unseen), min(unseen), []
        while cur not in cycle:
            cycle.append(cur)
            cur = model.transitions[cur][action]
        if cur != seed:
            raise ValueError("HOLD_MALFORMED_ORBIT")
        unseen.difference_update(cycle)
        blocks.append(tuple(sorted(cycle)))
    return tuple(sorted(blocks))


@dataclass(frozen=True)
class FactorIR:
    sizes: tuple[int, int]
    action_factor: dict[int, int]
    local_transitions: dict[int, dict[int, int]]

    def to_dict(self):
        return {
            "sizes": list(self.sizes),
            "action_factor": {str(k): v for k, v in self.action_factor.items()},
            "local_transitions": {str(a): {str(s): t for s, t in m.items()}
                                  for a, m in self.local_transitions.items()},
        }

    @staticmethod
    def from_dict(p):
        sizes = tuple(int(x) for x in p["sizes"])
        af = {int(k): int(v) for k, v in p["action_factor"].items()}
        lt = {int(a): {int(s): int(t) for s, t in m.items()}
              for a, m in p["local_transitions"].items()}
        if len(sizes) != 2 or any(x < 1 for x in sizes):
            raise ValueError("HOLD_INVALID_FACTOR_SIZES")
        if set(af) != set(lt) or any(v not in (0, 1) for v in af.values()):
            raise ValueError("HOLD_INVALID_FACTOR_MAP")
        for a, dim in af.items():
            if set(lt[a]) != set(range(sizes[dim])):
                raise ValueError("HOLD_INCOMPLETE_LOCAL_MAP")
            if any(t < 0 or t >= sizes[dim] for t in lt[a].values()):
                raise ValueError("HOLD_LOCAL_TARGET_OUT_OF_RANGE")
        return FactorIR(sizes, af, lt)


def induce_factor_ir(model: LearnedMoore):
    partitions = {}
    for action in model.alphabet:
        partitions.setdefault(_orbit_partition(model, action), []).append(action)
    if len(partitions) != 2:
        raise ValueError("HOLD_EXPECTED_TWO_FACTORS")
    parts = list(partitions)
    block_maps = [
        {state: i for i, block in enumerate(part) for state in block}
        for part in parts
    ]
    coords = {s: (block_maps[0][s], block_maps[1][s])
              for s in range(len(model.outputs))}
    sizes = (len(parts[0]), len(parts[1]))
    cartesian = set(itertools.product(range(sizes[0]), range(sizes[1])))
    if set(coords.values()) != cartesian:
        raise ValueError("HOLD_NONCARTESIAN_STATE_SPACE")

    action_factor, local = {}, {}
    for action in model.alphabet:
        changed_dims, mapping = set(), {}
        for state in range(len(model.outputs)):
            before = coords[state]
            after = coords[model.transitions[state][action]]
            changed = [i for i in (0, 1) if before[i] != after[i]]
            if len(changed) != 1:
                raise ValueError("HOLD_ACTION_NOT_LOCAL")
            dim = changed[0]
            changed_dims.add(dim)
            old, new = before[dim], after[dim]
            if old in mapping and mapping[old] != new:
                raise ValueError("HOLD_NONUNIFORM_LOCAL_TRANSITION")
            mapping[old] = new
        if len(changed_dims) != 1:
            raise ValueError("HOLD_ACTION_CHANGES_FACTOR")
        dim = next(iter(changed_dims))
        if set(mapping) != set(range(sizes[dim])):
            raise ValueError("HOLD_INCOMPLETE_LOCAL_TRANSITION")
        action_factor[action], local[action] = dim, mapping

    for state, before in coords.items():
        for action in model.alphabet:
            after = list(before)
            dim = action_factor[action]
            after[dim] = local[action][before[dim]]
            if tuple(after) != coords[model.transitions[state][action]]:
                raise ValueError("HOLD_FACTOR_COMPOSITION_MISMATCH")
    return FactorIR(sizes, action_factor, local), coords


@dataclass(frozen=True)
class AutomatonIR:
    factor: FactorIR
    alphabet: tuple[int, ...]
    initial: tuple[int, int]
    output_kind: str
    max_length: int
    state_outputs: tuple[int, ...] = ()
    output_dimension: int | None = None

    @property
    def arity(self):
        return 1

    def __post_init__(self):
        if set(self.alphabet) != set(self.factor.action_factor):
            raise ValueError("HOLD_ALPHABET_MISMATCH")
        if self.output_kind not in ("state_table", "factor_initial", "all_initial"):
            raise ValueError("HOLD_INVALID_OUTPUT_KIND")
        if self.output_kind == "state_table" and len(self.state_outputs) != self.factor.sizes[0] * self.factor.sizes[1]:
            raise ValueError("HOLD_INVALID_OUTPUT_TABLE")
        if self.output_kind == "factor_initial" and self.output_dimension not in (0, 1):
            raise ValueError("HOLD_INVALID_OUTPUT_DIMENSION")
        if len(self.initial) != 2 or any(self.initial[i] < 0 or self.initial[i] >= self.factor.sizes[i] for i in (0, 1)):
            raise ValueError("HOLD_INVALID_INITIAL_COORDINATES")
        if self.max_length < 0:
            raise ValueError("HOLD_INVALID_MAX_LENGTH")

    def supports(self, xs):
        if len(xs) != 1:
            return False
        raw = xs[0]
        if isinstance(raw, bool):
            return False
        try:
            code = int(raw)
            return code == raw and len(decode_trace(code, self.alphabet, self.max_length)) <= self.max_length
        except (TypeError, ValueError, OverflowError):
            return False

    def _coords(self, code):
        coords = list(self.initial)
        for action in decode_trace(code, self.alphabet, self.max_length):
            dim = self.factor.action_factor[action]
            coords[dim] = self.factor.local_transitions[action][coords[dim]]
        return tuple(coords)

    def eval(self, xs, env=None):
        if not self.supports(xs):
            raise ValueError("HOLD_OUTSIDE_CERTIFIED_SUPPORT")
        coords = self._coords(int(xs[0]))
        if self.output_kind == "factor_initial":
            d = self.output_dimension
            return int(coords[d] == self.initial[d])
        if self.output_kind == "all_initial":
            return int(coords == self.initial)
        return int(self.state_outputs[coords[0] * self.factor.sizes[1] + coords[1]])

    def bits(self, expert_index=None):
        return (16 + sum(s.bit_length() for s in self.factor.sizes)
                + sum(len(m) * 4 for m in self.factor.local_transitions.values())
                + len(self.alphabet) * 4 + self.max_length.bit_length()
                + (len(self.state_outputs) if self.output_kind == "state_table" else 2))

    def verify_contract(self, oracle_or_domain):
        if isinstance(oracle_or_domain, tuple) and len(oracle_or_domain) == 2:
            teacher, domain = oracle_or_domain
            rows = (((x,), teacher(x)) for x in domain)
        else:
            rows = iter(oracle_or_domain)
        checked = 0
        for xs, y in rows:
            if not self.supports(xs) or self.eval(xs) != int(y):
                return {"ok": False, "checked": checked, "counterexample": tuple(xs)}
            checked += 1
        return {"ok": True, "checked": checked, "counterexample": None}

    def to_dict(self):
        return {
            "schema_version": 1, "factor": self.factor.to_dict(),
            "alphabet": list(self.alphabet), "initial": list(self.initial),
            "output_kind": self.output_kind, "max_length": self.max_length,
            "state_outputs": list(self.state_outputs),
            "output_dimension": self.output_dimension,
        }

    @staticmethod
    def from_dict(p):
        if p.get("schema_version") != 1:
            raise ValueError("HOLD_SCHEMA_MISMATCH")
        return AutomatonIR(
            FactorIR.from_dict(p["factor"]), tuple(int(x) for x in p["alphabet"]),
            tuple(int(x) for x in p["initial"]), str(p["output_kind"]),
            int(p["max_length"]), tuple(int(x) for x in p.get("state_outputs", [])),
            None if p.get("output_dimension") is None else int(p["output_dimension"]),
        )

    def emit_python(self, fn_name, env=None):
        if not fn_name.isidentifier():
            raise ValueError("HOLD_INVALID_FUNCTION_NAME")
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return (
            "def " + fn_name + "(code):\n"
            + "    p = " + repr(json.loads(payload)) + "\n"
            + "    base = len(p['alphabet']) + 1\n"
            + "    if code < 0 or code >= base ** p['max_length']: raise ValueError('outside support')\n"
            + "    af = {int(k): int(v) for k, v in p['factor']['action_factor'].items()}\n"
            + "    lt = {int(a): {int(s): int(t) for s, t in m.items()} for a, m in p['factor']['local_transitions'].items()}\n"
            + "    c = list(p['initial'])\n"
            + "    while code:\n"
            + "        d, code = code % base, code // base\n"
            + "        if d == 0:\n"
            + "            if code: raise ValueError('noncanonical trace')\n"
            + "            break\n"
            + "        a = p['alphabet'][d - 1]; k = af[a]; c[k] = lt[a][c[k]]\n"
            + "    kind = p['output_kind']\n"
            + "    if kind == 'factor_initial': return int(c[p['output_dimension']] == p['initial'][p['output_dimension']])\n"
            + "    if kind == 'all_initial': return int(c == p['initial'])\n"
            + "    return int(p['state_outputs'][c[0] * p['factor']['sizes'][1] + c[1]])\n"
        )


def _restore(payload):
    return AutomatonIR.from_dict(payload)


from hagi.representations import register_artifact_factory
register_artifact_factory("factorized_moore", _restore)


def invent_factorized_automata(membership, alphabet, max_states, max_length):
    model = learn_bounded_moore(membership, alphabet, max_states)
    factor, coords = induce_factor_ir(model)
    initial = coords[0]
    outputs = [0] * (factor.sizes[0] * factor.sizes[1])
    for state, xy in coords.items():
        outputs[xy[0] * factor.sizes[1] + xy[1]] = model.outputs[state]
    arts = {
        "observed_output": AutomatonIR(factor, model.alphabet, initial, "state_table", max_length, tuple(outputs)),
        "factor_0_initial": AutomatonIR(factor, model.alphabet, initial, "factor_initial", max_length, output_dimension=0),
        "factor_1_initial": AutomatonIR(factor, model.alphabet, initial, "factor_initial", max_length, output_dimension=1),
        "all_initial": AutomatonIR(factor, model.alphabet, initial, "all_initial", max_length),
    }
    return {
        "model": model, "factor_ir": factor, "coordinates": coords, "artifacts": arts,
        "metrics": {
            "states": len(model.outputs), "queries": model.query_count,
            "monolithic_transition_edges": len(model.outputs) * len(model.alphabet),
            "factorized_transition_entries": sum(len(m) for m in factor.local_transitions.values()),
        },
    }


def register_factorized_candidates(artifacts, namespace="automaton"):
    from hagi.representations import register_artifact_candidate
    names = []
    for key, artifact in sorted(artifacts.items()):
        name = namespace + ":" + key
        register_artifact_candidate(name, "factorized_moore", artifact)
        names.append(name)
    return names
