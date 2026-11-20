"""End-to-end tests for invention -> FactorIR -> ExprV9 -> BatchedMDL."""
import itertools
import json

import pytest

from hagi.automaton_repr import (
    AutomatonIR, decode_trace, encode_trace, invent_factorized_automata,
    register_factorized_candidates, words_upto,
)
from hagi.core import ExprV9, GrowingHagiProcessorV9, TaskSpec
from hagi.representations import clear_artifact_candidates


class ProductOracle:
    """Hidden Z3 x Z2 automaton; only its joint acceptance bit is observable."""
    alphabet = (0, 1, 2, 3)

    @staticmethod
    def run_state(word):
        a, b = 0, 0
        for action in word:
            if action == 0:
                a = (a + 1) % 3
            elif action == 1:
                a = (a - 1) % 3
            elif action == 2:
                b = (b + 1) % 2
            elif action == 3:
                b = (b - 1) % 2
            else:
                raise ValueError(action)
        return a, b

    def membership(self, word):
        return int(self.run_state(word) == (0, 0))


def _build():
    oracle = ProductOracle()
    learned = invent_factorized_automata(
        oracle.membership, oracle.alphabet, max_states=6, max_length=4
    )
    return oracle, learned


def test_factorizer_recovers_hidden_coordinates_and_exact_output():
    oracle, learned = _build()
    arts = learned["artifacts"]
    for word in words_upto(oracle.alphabet, 4):
        code = encode_trace(word, oracle.alphabet)
        state = oracle.run_state(word)
        assert arts["observed_output"].eval((code,)) == oracle.membership(word)
        assert arts["factor_0_initial"].eval((code,)) == int(state[0] == 0)
        assert arts["factor_1_initial"].eval((code,)) == int(state[1] == 0)
        assert arts["all_initial"].eval((code,)) == oracle.membership(word)


def test_factorized_artifact_json_restore_and_emitted_code():
    oracle, learned = _build()
    artifact = learned["artifacts"]["factor_0_initial"]
    restored = AutomatonIR.from_dict(json.loads(json.dumps(artifact.to_dict())))
    for word in words_upto(oracle.alphabet, 4):
        code = encode_trace(word, oracle.alphabet)
        assert restored.eval((code,)) == artifact.eval((code,))
    ns = {}
    exec(artifact.emit_python("factor0"), ns)
    assert ns["factor0"](encode_trace((0, 2, 1), oracle.alphabet)) == 1
    assert not artifact.supports((1.5,))
    assert not artifact.supports((True,))


def test_trace_encoding_is_canonical_and_bounded():
    alphabet = (0, 1, 2, 3)
    word = (0, 2, 1)
    assert decode_trace(encode_trace(word, alphabet), alphabet, 4) == word
    with pytest.raises(ValueError):
        decode_trace(1 + 0 * 5 + 2 * 25, alphabet, 4)
    with pytest.raises(ValueError):
        decode_trace(5 ** 4, alphabet, 4)


def test_real_processor_composes_invented_coordinates_via_batched_mdl():
    clear_artifact_candidates()
    oracle, learned = _build()
    # Deliberately publish only the new coordinate projections, not the
    # joint target. A successful COMMIT must discover bit_and composition.
    register_factorized_candidates({
        "factor_0_initial": learned["artifacts"]["factor_0_initial"],
        "factor_1_initial": learned["artifacts"]["factor_1_initial"],
    }, namespace="test_product")
    domain = tuple(
        encode_trace(w, oracle.alphabet)
        for w in words_upto(oracle.alphabet, 4)
    )
    spec = TaskSpec(
        name="AUTOMATON_PRODUCT_FACTOR_COMPOSE",
        arity=1,
        domains=(domain,),
        teacher=lambda code: oracle.membership(
            decode_trace(code, oracle.alphabet, 4)
        ),
        family="factorized-automaton",
    )
    proc = GrowingHagiProcessorV9(seed=3, node_budget=8000)
    result = proc.learn(spec, seed=19, train_neural=False)
    assert result["action"] == "COMMIT", result
    expert = proc.experts[spec.name]
    assert expert.program.expr is not None
    assert expert.program.expr.op == "bit_and", expert.program.pretty()
    assert all(
        expert.program.eval((code,), proc.experts) == spec.teacher(code)
        for code in domain
    )
    # Persisted ExprV9 carries only the declarative artifact descriptor.
    payload = expert.program.expr.to_dict()
    restored = ExprV9.from_dict(payload)
    assert all(restored.eval((code,), proc.experts) == spec.teacher(code) for code in domain)
    clear_artifact_candidates()
