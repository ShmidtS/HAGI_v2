"""Lightweight, dependency-free adaptive request router.

The first MVP intentionally uses a deterministic lexical scorer instead of
requiring a second neural model. It is designed as a replaceable interface:
a trained/quantized Jev-like classifier can implement the same Router protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from hagi.inference.adaptive import Route, RouteDecision

_WORD = re.compile(r"[\w+#./:-]+", re.UNICODE)


@dataclass(frozen=True)
class DomainProfile:
    keywords: tuple[str, ...]
    route: Route = Route.MAIN


PROFILES: dict[str, DomainProfile] = {
    "CODE": DomainProfile((
        "python", "javascript", "typescript", "rust", "c++", "java", "code",
        "код", "функция", "класс", "api", "bug", "stacktrace", "sql", "regex",
        "compile", "compiler", "implement", "реализуй",
    ), Route.SPECIALIST),
    "MATH": DomainProfile((
        "integral", "derivative", "equation", "matrix", "probability", "proof",
        "theorem", "math", "математика", "уравнение", "интеграл", "производная",
        "доказательство", "вероятность", "матрица", "формула",
    ), Route.SPECIALIST),
    "RU": DomainProfile((
        "русский", "русском", "российский", "падеж", "склонение", "кириллица",
    ), Route.SPECIALIST),
    "EN": DomainProfile((
        "english", "английский", "grammar", "essay", "переведи на английский",
    ), Route.SPECIALIST),
    "GENERAL": DomainProfile((), Route.MAIN),
}


class HeuristicRouter:
    """Tiny lexical router suitable for shadow-mode and calibration bootstrap.

    It is deliberately conservative: a specialist route requires both a
    meaningful domain score and a configurable confidence margin.
    """

    def __init__(self, *, temperature: float = 1.0, specialist_threshold: float = 0.72):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0.0 <= specialist_threshold <= 1.0:
            raise ValueError("specialist_threshold must be in [0, 1]")
        self.temperature = float(temperature)
        self.specialist_threshold = float(specialist_threshold)

    @staticmethod
    def _text(request: Any) -> str:
        if isinstance(request, str):
            return request
        if isinstance(request, dict):
            for key in ("prompt", "text", "input"):
                value = request.get(key)
                if value is not None:
                    return str(value)
        return str(request)

    def predict(self, request: Any) -> RouteDecision:
        text = self._text(request).lower()
        tokens = set(_WORD.findall(text))
        scores: dict[str, float] = {}
        for domain, profile in PROFILES.items():
            if not profile.keywords:
                scores[domain] = 0.5
                continue
            hits = sum(1 for kw in profile.keywords if kw.lower() in tokens or kw.lower() in text)
            scores[domain] = 2.5 * hits

        # Length/complexity acts as a weak prior toward the main model.
        complexity = min(1.0, len(tokens) / 160.0)
        scores["GENERAL"] += 1.0 + 0.25 * complexity

        domains = tuple(scores)
        logits = [scores[d] / self.temperature for d in domains]
        mx = max(logits)
        exps = [math.exp(v - mx) for v in logits]
        z = sum(exps)
        probs = {d: e / z for d, e in zip(domains, exps)}
        domain = max(probs, key=probs.get)
        confidence = float(probs[domain])
        profile = PROFILES[domain]

        route = profile.route
        reason = f"domain:{domain.lower()}"
        if route != Route.MAIN and confidence < self.specialist_threshold:
            route = Route.MAIN
            reason = f"ambiguous:{domain.lower()}"
        if domain == "GENERAL":
            route = Route.MAIN

        # Router risk is the uncertainty of the route decision, not model answer
        # error probability.
        risk = 1.0 - confidence
        return RouteDecision(
            route=route,
            domain=domain,
            confidence=confidence,
            risk=risk,
            domain_probs=probs,
            labels=(domain.lower(), reason, "high_confidence" if confidence >= self.specialist_threshold else "uncertain"),
            reason=reason,
        )
