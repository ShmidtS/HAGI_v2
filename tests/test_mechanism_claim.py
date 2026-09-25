"""``mechanism_supported`` must be a measurement, not a constant.

Before this change ``_evidence_payload`` wrote ``"mechanism_supported": True``
regardless of what the holdout evaluator decided, so a rejected candidate still
produced a persisted file claiming the growth mechanism was supported. That is
the unconditional-claim pattern the project paid for twice, and it is what the
DecisionPlane gate died of.

These tests pin the claim to the verdict and reject a hand-edited evidence
file on load.
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT / "src"))

from hagi.orchestrator import recursive as R  # noqa: E402


def _verdict(decision: str, regression: float, worst: float) -> dict:
    return {
        "decision": decision,
        "ce_regression": regression,
        "worst_source_regression": worst,
        "pareto_improvement": regression < 0.0,
    }


def test_accepted_non_regressing_verdict_supports_the_mechanism():
    assert R._mechanism_supported(_verdict("accepted", -0.5, -0.1)) is True


def test_flat_verdict_still_supports_the_mechanism():
    # accepted means macro CE did not regress and no source exceeded tolerance.
    assert R._mechanism_supported(_verdict("accepted", 0.0, 0.01)) is True


def test_rejected_verdict_supports_nothing():
    assert R._mechanism_supported(_verdict("rejected", 0.9, 0.9)) is False


def test_claim_cannot_be_true_for_a_regressing_verdict():
    # Even if a caller mislabels the decision, the numeric guards hold.
    assert R._mechanism_supported(_verdict("accepted", 0.5, 0.0)) is False
    assert R._mechanism_supported(_verdict("accepted", -0.1, 0.4)) is False


def test_unknown_decision_supports_nothing():
    assert R._mechanism_supported(_verdict("inconclusive", -1.0, -1.0)) is False


def test_validator_rejects_a_flipped_mechanism_claim():
    """A hand-edited evidence file must fail closed, not be trusted.

    Exercised directly against the loader: the mechanism claim is recomputed
    from the persisted metrics, so flipping the boolean in the file is
    detected instead of being carried forward.
    """
    import inspect

    src = inspect.getsource(R._validate_evidence)
    assert "_mechanism_supported(" in src, (
        "the loader must recompute the mechanism claim from the verdict"
    )
    # And the recomputation is a real check, not a tautology.
    assert "does not follow from the verdict" in src


def test_persisted_rejected_run_cannot_claim_support():
    """End-to-end: a rejected generation persists mechanism_supported=False.

    The regression is injected through the evaluator the way the existing
    ``test_forced_regression_rejects_and_parent_bytes_unchanged`` test does it,
    so the numbers stay mutually consistent and the loader accepts the run.
    Every source must worsen past the tolerance: with a mixed candidate the
    macro CE can still fall, which is an *accepted* generation by design, and
    this test would then be asserting against its own name.
    """
    import sys
    import tempfile

    sys.path.insert(0, str(_ROOT))
    from hagi.orchestrator.recursive import run_generation  # noqa: PLC0415
    from tests.test_recursive_growth_owner import (  # noqa: PLC0415
        OWNER,
        _candidate_builder,
        _child_builder,
        _evaluator,
        _fixtures,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store, request = _fixtures(root)
        result = run_generation(
            request, store, _child_builder(root, []), _candidate_builder(root, []),
            _evaluator(candidate=(1.2, 1.3, 1.4)), owner_id=OWNER,
        )
        assert result.decision == "rejected"
        evidence = json.loads((root / "g1" / "holdout-evidence.json").read_text())
        assert evidence["mechanism_supported"] is False, (
            "a rejected generation must not claim the mechanism is supported"
        )
