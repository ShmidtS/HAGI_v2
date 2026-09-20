import unittest

from hagi.inference.adaptive import (
    AdaptiveInferenceController, BudgetPolicy, InferenceResult,
    ParameterMap, Route, RouteCandidate, RouteDecision,
)


class FakeRouter:
    def __init__(self, decision):
        self.decision = decision

    def predict(self, request):
        return self.decision


class FakeVerifier:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def verify(self, request, result):
        return next(self.outcomes)


class FakeLearner:
    def __init__(self):
        self.items = []

    def enqueue(self, request, result, decision):
        self.items.append((request, result, decision))
        return True


class AdaptiveInferenceTests(unittest.TestCase):
    def setUp(self):
        self.decision = RouteDecision(Route.SPECIALIST, "CODE", 0.91, risk=0.1)
        self.candidates = [
            RouteCandidate(Route.SPECIALIST, "CODE", 8.0, max_risk=0.2),
            RouteCandidate(Route.MAIN, "CODE", 30.0, max_risk=1.0),
        ]

    def test_policy_chooses_fast_feasible_route(self):
        selected = BudgetPolicy().select(self.candidates, self.decision, 10.0)
        self.assertEqual(selected.route, Route.SPECIALIST)

    def test_budget_fallback_chooses_candidate_even_if_over_budget(self):
        selected = BudgetPolicy().select(self.candidates, self.decision, 1.0)
        self.assertEqual(selected.route, Route.SPECIALIST)

    def test_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            RouteDecision(Route.MAIN, "EN", 1.1)

    def test_verifier_failure_uses_fallback(self):
        def execute(request, candidate):
            return InferenceResult("first", candidate.route)

        def fallback(request, previous):
            return InferenceResult("verified fallback", Route.FALLBACK)

        controller = AdaptiveInferenceController(
            FakeRouter(self.decision), ParameterMap(self.candidates), execute,
            FakeVerifier([False, True]), fallback=fallback,
        )
        result, trace = controller.generate("task", latency_budget_ms=10)
        self.assertEqual(result.output, "verified fallback")
        self.assertTrue(trace.fallback_used)
        self.assertTrue(trace.verifier_accepted)

    def test_learner_only_receives_verified_result(self):
        learner = FakeLearner()
        controller = AdaptiveInferenceController(
            FakeRouter(self.decision), ParameterMap(self.candidates),
            lambda request, candidate: InferenceResult("ok", candidate.route),
            FakeVerifier([True]), learner=learner,
        )
        _, trace = controller.generate("task", latency_budget_ms=10)
        self.assertTrue(trace.learner_enqueued)
        self.assertEqual(len(learner.items), 1)

    def test_shadow_prefers_main_route(self):
        seen = []
        controller = AdaptiveInferenceController(
            FakeRouter(self.decision), ParameterMap(self.candidates),
            lambda request, candidate: (seen.append(candidate.route) or InferenceResult("ok", candidate.route)),
            FakeVerifier([True]), shadow=True,
        )
        controller.generate("task", latency_budget_ms=10)
        self.assertEqual(seen, [Route.MAIN])


if __name__ == "__main__":
    unittest.main()
