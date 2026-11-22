"""Integrated retrieval-and-feedback loop for the local HAGI experiments.

The compact ternary ByteLM is used as an evidence reranker, not misrepresented
as a general-purpose answer generator. Answers in this prototype are extracted
from selected records; feedback must come from an external verifier.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import torch

from .streaming import DiskBackedContext, MemoryHit

_IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z_-]{1,}[0-9][A-Z0-9_-]*\b")
_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_VALUE_RE = re.compile(r"\bvalue\s*[=:]\s*([A-Za-z0-9._/-]+)", flags=re.IGNORECASE)
_FINAL_RE = re.compile(
    r"\b(final\s+decision|approved|adopted|selected|confirmed|current|authoritative)\b",
    flags=re.IGNORECASE,
)
_OLD_RE = re.compile(
    r"\b(earlier|preliminary|rejected|superseded|obsolete|draft|withdrawn|trial)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RankedEvidence:
    hit: MemoryHit
    neural_score: float
    utility_score: float
    cached: bool
    combined_score: float


@dataclass(frozen=True)
class MemoryAnswer:
    query: str
    status: str
    answer: str | None
    evidence: MemoryHit | None
    score: float | None
    margin: float | None
    candidates: tuple[RankedEvidence, ...]


def _first_identifier(text: str) -> str:
    match = _IDENTIFIER_RE.search(str(text))
    return match.group(0).casefold() if match else "none"


def _status_tag(text: str) -> str:
    final = bool(_FINAL_RE.search(text))
    old = bool(_OLD_RE.search(text))
    if old:
        return "old"
    if final:
        return "final"
    return "neutral"


def _compact_words(text: str, max_chars: int = 45) -> str:
    words = [w.casefold() for w in _TOKEN_RE.findall(str(text))]
    return " ".join(words)[:max_chars]


def pair_prompt(query: str, evidence: str, context: int = 160) -> bytes:
    """Create a fixed-size, inspectable pair representation for the ByteLM.

    Explicit identifier equality and a coarse evidence-status tag are exposed
    as features. The neural model learns how to combine these with query and
    evidence words; this is not an opaque embedding or a truth detector.
    """
    if context < 32:
        raise ValueError("context must be >= 32")
    query_id = _first_identifier(query)
    evidence_id = _first_identifier(evidence)
    id_match = int(query_id != "none" and query_id == evidence_id)
    status = _status_tag(evidence)
    fixed = f"|IDM={id_match}|ST={status}|R="
    available = max(0, context - len(fixed.encode("utf-8")) - len("Q=|E="))
    half = available // 2
    q_words = _compact_words(query, max_chars=half)
    e_words = _compact_words(evidence, max_chars=available - len(q_words))
    body = f"Q={q_words}|E={e_words}{fixed}".encode("utf-8")
    if len(body) > context:
        body = body[-context:]
    return b" " * (context - len(body)) + body


def pair_training_record(query: str, evidence: str, relevant: bool,
                         context: int = 160) -> bytes:
    """One supervised next-byte example: 1 means relevant, 0 means not."""
    return pair_prompt(query, evidence, context=context) + (b"1" if relevant else b"0")


class TernaryEvidenceReranker:
    """Scores retrieved evidence using a deployed TernaryAwareByteLM."""

    def __init__(self, model, context: int | None = None):
        self.model = model
        self.context = int(context if context is not None else getattr(model, "context", 160))
        if int(getattr(model, "context", self.context)) != self.context:
            raise ValueError("reranker context does not match model architecture")
        self.model.eval()

    @torch.no_grad()
    def score(self, query: str, evidence: str) -> float:
        encoded = pair_prompt(query, evidence, context=self.context)
        x = torch.tensor([list(encoded)], dtype=torch.long)
        logits = self.model(x)[0]
        # Logit margin is more stable for ranking than a saturated probability.
        return float((logits[ord("1")] - logits[ord("0")]).item())


class MemoryAugmentedAgent:
    """Hybrid FTS + semantic retrieval, ternary reranking and verified feedback.

    The persistent archive is never pruned. Utility feedback only affects the
    bounded working-set preference; the archive remains the fallback source.
    """

    def __init__(
        self,
        memory: DiskBackedContext,
        semantic_index,
        reranker: TernaryEvidenceReranker,
        utility_index=None,
        utility_weight: float = 0.10,
        cache_bonus: float = 0.0,
        max_candidates: int = 24,
        min_score_margin: float = 0.0,
    ):
        if max_candidates < 1 or utility_weight < 0 or cache_bonus < 0 or min_score_margin < 0:
            raise ValueError("invalid agent configuration")
        self.memory = memory
        self.semantic_index = semantic_index
        self.reranker = reranker
        self.utility_index = utility_index
        self.utility_weight = float(utility_weight)
        self.cache_bonus = float(cache_bonus)
        self.max_candidates = int(max_candidates)
        self.min_score_margin = float(min_score_margin)

    def retrieve(self, query: str) -> tuple[RankedEvidence, ...]:
        hits: list[MemoryHit] = []
        seen: set[tuple[int, int, int]] = set()

        def add(found: Iterable[MemoryHit]) -> None:
            for hit in found:
                key = (hit.page_id, hit.char_start, hit.char_end)
                if key not in seen:
                    seen.add(key)
                    hits.append(hit)

        # Exact identifier lookup anchors entity identity even when the user's
        # wording is a paraphrase; semantic/FTS retrieval adds related context.
        identifier = _first_identifier(query)
        if identifier != "none":
            add(self.memory.search(identifier, limit=self.max_candidates))
        if self.semantic_index is not None:
            add(self.semantic_index.search(query, limit=self.max_candidates))
        add(self.memory.search(query, limit=self.max_candidates))

        if not hits:
            return ()

        cached_ids: set[int] = set()
        learner = None
        if self.utility_index is not None:
            cached_ids = self.utility_index.working_set.cached_page_ids
            learner = self.utility_index.working_set.learner

        ranked: list[RankedEvidence] = []
        for hit in hits[: self.max_candidates * 3]:
            neural = self.reranker.score(query, hit.text)
            utility = learner.score(hit.text) if learner is not None else 0.0
            cached = hit.page_id in cached_ids
            combined = neural + self.utility_weight * utility + (self.cache_bonus if cached else 0.0)
            ranked.append(RankedEvidence(hit, neural, utility, cached, combined))
        ranked.sort(key=lambda item: (item.combined_score, -item.hit.page_id), reverse=True)
        return tuple(ranked[: self.max_candidates])

    def answer(self, query: str) -> MemoryAnswer:
        candidates = self.retrieve(query)
        if not candidates:
            return MemoryAnswer(query, "unknown", None, None, None, None, ())
        top = candidates[0]
        second_score = candidates[1].combined_score if len(candidates) > 1 else None
        margin = None if second_score is None else top.combined_score - second_score
        value_match = _VALUE_RE.search(top.hit.text)
        if value_match is None:
            return MemoryAnswer(query, "no_extractable_value", None, top.hit, top.combined_score,
                                margin, candidates)
        if margin is not None and margin < self.min_score_margin:
            return MemoryAnswer(query, "uncertain", None, top.hit, top.combined_score,
                                margin, candidates)
        return MemoryAnswer(query, "answered", value_match.group(1), top.hit,
                            top.combined_score, margin, candidates)

    def record_verified_feedback(self, result: MemoryAnswer, verified_correct: bool,
                                 verifier: str, refresh: bool = False):
        """Update utility only after an external correctness judgement.

        There is deliberately no self-reward or confidence-only update path.
        """
        if not isinstance(verified_correct, bool):
            raise TypeError("verified_correct must be an explicit bool")
        source = str(verifier).strip()
        if not source:
            raise ValueError("a verifier/source label is required")
        if result.evidence is None:
            raise ValueError("cannot provide page feedback without selected evidence")
        if self.utility_index is None:
            raise RuntimeError("no utility index is attached to this agent")
        reward = 1.0 if verified_correct else -1.0
        feedback = self.utility_index.record_page_feedback(result.evidence.page_id, reward=reward)
        if refresh:
            self.utility_index.refresh()
        return feedback
