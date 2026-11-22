"""Outcome-trained utility model for a bounded working set over long-term memory.

The raw archive remains the source of truth.  This module learns which content
features are repeatedly useful from successful/unsuccessful retrieval feedback,
then selects a small fast working set from an iterable archive.  The learner is
intentionally simple, local, deterministic and inspectable; it is not a semantic
language model or a truth detector.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Iterable, Iterator

import numpy as np

_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "was", "were", "are", "is",
    "has", "have", "had", "into", "onto", "about", "after", "before", "under", "over",
    "record", "note", "entry", "fact", "value", "item", "asset", "uses", "using",
    "used", "shows", "states", "says", "saying", "according", "reference", "details",
    "what", "which", "when", "where", "who", "how", "why", "did", "does", "can",
    "could", "would", "should", "please", "tell", "me", "find", "look", "up",
}


def _hash_feature(token: str, bins: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % bins


@dataclass(frozen=True)
class UtilityRecord:
    record_id: str
    text: str


@dataclass(frozen=True)
class UtilityRefreshStats:
    scanned: int
    retained: int
    seconds: float
    threshold_score: float


@dataclass(frozen=True)
class UtilityFeedback:
    record_id: str
    was_cached: bool
    reward: float
    update_count: int


class MemoryUseLearner:
    """Online content utility learner updated from retrieval outcome feedback.

    A positive reward increases weights of features in a useful record; a
    negative reward decreases them.  All weights decay slightly per feedback
    event so the working set can follow changing interests. Feature hashing
    keeps model size fixed regardless of archive size.
    """

    VERSION = 1

    def __init__(self, feature_bins: int = 4096, learning_rate: float = 1.0,
                 decay: float = 0.97, min_token_length: int = 3):
        if feature_bins < 128:
            raise ValueError("feature_bins must be >=128")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be >0")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0,1]")
        if min_token_length < 2:
            raise ValueError("min_token_length must be >=2")
        self.feature_bins = int(feature_bins)
        self.learning_rate = float(learning_rate)
        self.decay = float(decay)
        self.min_token_length = int(min_token_length)
        self.weights = np.zeros(self.feature_bins, dtype=np.float32)
        self.update_count = 0

    def feature_indices(self, text: str) -> np.ndarray:
        tokens = {
            t.casefold() for t in _TOKEN_RE.findall(str(text))
            if len(t) >= self.min_token_length and not any(ch.isdigit() for ch in t)
            and t.casefold() not in _STOP
        }
        if not tokens:
            return np.empty(0, dtype=np.int64)
        return np.asarray(sorted({_hash_feature(t, self.feature_bins) for t in tokens}), dtype=np.int64)

    def score(self, text: str) -> float:
        idx = self.feature_indices(text)
        if idx.size == 0:
            return 0.0
        return float(np.sum(self.weights[idx], dtype=np.float64) / np.sqrt(float(idx.size)))

    def observe(self, text: str, reward: float = 1.0) -> None:
        reward = float(reward)
        if not -1.0 <= reward <= 1.0:
            raise ValueError("reward must be in [-1,1]")
        self.weights *= self.decay
        idx = self.feature_indices(text)
        if idx.size:
            self.weights[idx] += np.float32(self.learning_rate * reward / np.sqrt(float(idx.size)))
        self.update_count += 1

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            version=np.asarray([self.VERSION], dtype=np.int32),
            feature_bins=np.asarray([self.feature_bins], dtype=np.int32),
            learning_rate=np.asarray([self.learning_rate], dtype=np.float32),
            decay=np.asarray([self.decay], dtype=np.float32),
            min_token_length=np.asarray([self.min_token_length], dtype=np.int32),
            update_count=np.asarray([self.update_count], dtype=np.int64),
            weights=self.weights,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MemoryUseLearner":
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["version"][0]) != cls.VERSION:
                raise ValueError("unsupported utility-model checkpoint")
            obj = cls(
                feature_bins=int(payload["feature_bins"][0]),
                learning_rate=float(payload["learning_rate"][0]),
                decay=float(payload["decay"][0]),
                min_token_length=int(payload["min_token_length"][0]),
            )
            weights = np.asarray(payload["weights"], dtype=np.float32)
            if weights.shape != (obj.feature_bins,):
                raise ValueError("invalid utility-model weight shape")
            obj.weights[:] = weights
            obj.update_count = int(payload["update_count"][0])
        return obj


class UtilityWorkingSet:
    """A bounded, rebuildable cache selected from an external archive iterator.

    ``record_feedback`` must be called only after an external retrieval outcome
    is known.  The archive itself is never deleted or modified by this cache.
    Rebuild uses a top-k heap, so selection memory is O(capacity), not O(archive).
    """

    def __init__(self, capacity: int, learner: MemoryUseLearner | None = None):
        if capacity < 1:
            raise ValueError("capacity must be >=1")
        self.capacity = int(capacity)
        self.learner = learner or MemoryUseLearner()
        self.cached_ids: set[str] = set()
        self._feedback_since_refresh = 0
        self.last_refresh: UtilityRefreshStats | None = None

    def contains(self, record_id: str) -> bool:
        return str(record_id) in self.cached_ids

    def record_feedback(self, record: UtilityRecord, reward: float = 1.0) -> UtilityFeedback:
        record_id = str(record.record_id)
        was_cached = record_id in self.cached_ids
        self.learner.observe(record.text, reward=reward)
        self._feedback_since_refresh += 1
        return UtilityFeedback(record_id, was_cached, float(reward), self.learner.update_count)

    def refresh(self, records: Iterable[UtilityRecord]) -> UtilityRefreshStats:
        """Select highest-utility records with bounded additional memory."""
        import heapq

        started = time.perf_counter()
        heap: list[tuple[float, str]] = []
        scanned = 0
        for record in records:
            scanned += 1
            rid = str(record.record_id)
            score = self.learner.score(record.text)
            # Stable lexicographic tie-break. Keep only capacity best IDs.
            item = (score, rid)
            if len(heap) < self.capacity:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        self.cached_ids = {rid for _, rid in heap}
        self._feedback_since_refresh = 0
        threshold = min((score for score, _ in heap), default=0.0)
        stats = UtilityRefreshStats(scanned, len(self.cached_ids), time.perf_counter() - started, float(threshold))
        self.last_refresh = stats
        return stats

    def feedback_and_refresh(self, record: UtilityRecord,
                             records: Iterable[UtilityRecord], reward: float = 1.0) -> tuple[UtilityFeedback, UtilityRefreshStats]:
        feedback = self.record_feedback(record, reward=reward)
        return feedback, self.refresh(records)

    def record_page_feedback(self, memory, page_id: int, reward: float = 1.0) -> UtilityFeedback:
        """Learn from a retrieved SQLite archive page (e.g. a rated MemoryHit)."""
        row = memory.db.execute("SELECT content FROM pages WHERE rowid=?", (int(page_id),)).fetchone()
        if row is None:
            raise KeyError(f"archive page {page_id} does not exist")
        return self.record_feedback(UtilityRecord(str(int(page_id)), str(row[0])), reward=reward)

    def refresh_from_memory(self, memory) -> UtilityRefreshStats:
        """Rebuild the bounded page working set by streaming rows from SQLite."""
        cursor = memory.db.execute("SELECT rowid,content FROM pages ORDER BY rowid")
        def records() -> Iterator[UtilityRecord]:
            for page_id, content in cursor:
                yield UtilityRecord(str(int(page_id)), str(content))
        return self.refresh(records())

    @property
    def cached_page_ids(self) -> set[int]:
        """Cached identifiers as integers for direct use with archive page IDs."""
        result: set[int] = set()
        for value in self.cached_ids:
            try:
                result.add(int(value))
            except ValueError:
                continue
        return result

    def state(self) -> dict:
        return {
            "capacity": self.capacity,
            "cached_ids": sorted(self.cached_ids),
            "feedback_since_refresh": self._feedback_since_refresh,
            "learner_updates": self.learner.update_count,
        }


class SQLiteUtilityIndex:
    """Persistent inverted-feature index for fast utility-cache refreshes.

    The archive stays append-only in ``DiskBackedContext``. This class stores
    page->feature postings on disk once, then a refreshed working set scores
    only pages sharing currently rewarded features instead of re-tokenizing the
    entire archive on every update. Additional RAM stays O(feature_bins + K).
    """

    def __init__(self, memory, working_set: UtilityWorkingSet):
        self.memory = memory
        self.working_set = working_set
        db = memory.db
        db.execute("""CREATE TABLE IF NOT EXISTS utility_page_stats (
            page_id INTEGER PRIMARY KEY, feature_count INTEGER NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS utility_page_features (
            page_id INTEGER NOT NULL, feature_idx INTEGER NOT NULL,
            PRIMARY KEY(page_id, feature_idx)) WITHOUT ROWID""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_utility_feature_page ON utility_page_features(feature_idx,page_id)")
        db.execute("""CREATE TEMP TABLE IF NOT EXISTS utility_active_weights (
            feature_idx INTEGER PRIMARY KEY, weight REAL NOT NULL) WITHOUT ROWID""")
        db.commit()
        self.indexed_page_id = int(db.execute(
            "SELECT COALESCE(MAX(page_id),0) FROM utility_page_stats"
        ).fetchone()[0])

    def index_new_pages(self, batch_pages: int = 512) -> int:
        """Tokenize/index only newly appended archive pages; return page count."""
        if batch_pages < 1:
            raise ValueError("batch_pages must be >=1")
        db = self.memory.db
        cur = db.execute(
            "SELECT rowid,content FROM pages WHERE rowid>? ORDER BY rowid",
            (self.indexed_page_id,),
        )
        indexed = 0
        while True:
            rows = cur.fetchmany(batch_pages)
            if not rows:
                break
            stat_rows: list[tuple[int, int]] = []
            feature_rows: list[tuple[int, int]] = []
            for page_id, content in rows:
                pid = int(page_id)
                indices = self.working_set.learner.feature_indices(str(content))
                stat_rows.append((pid, int(indices.size)))
                feature_rows.extend((pid, int(idx)) for idx in indices.tolist())
                self.indexed_page_id = pid
            db.executemany(
                "INSERT OR REPLACE INTO utility_page_stats(page_id,feature_count) VALUES (?,?)",
                stat_rows,
            )
            if feature_rows:
                db.executemany(
                    "INSERT OR REPLACE INTO utility_page_features(page_id,feature_idx) VALUES (?,?)",
                    feature_rows,
                )
            db.commit()
            indexed += len(rows)
        return indexed

    def record_page_feedback(self, page_id: int, reward: float = 1.0) -> UtilityFeedback:
        """Record utility feedback for one page, indexing it first if necessary."""
        self.index_new_pages()
        pid = int(page_id)
        row = self.memory.db.execute(
            "SELECT content FROM pages WHERE rowid=?", (pid,)
        ).fetchone()
        if row is None:
            raise KeyError(f"archive page {pid} does not exist")
        self.memory.db.execute(
            "INSERT OR IGNORE INTO utility_page_stats(page_id,feature_count) VALUES (?,0)", (pid,)
        )
        indices = self.working_set.learner.feature_indices(str(row[0]))
        self.memory.db.execute("DELETE FROM utility_page_features WHERE page_id=?", (pid,))
        if indices.size:
            self.memory.db.executemany(
                "INSERT INTO utility_page_features(page_id,feature_idx) VALUES (?,?)",
                [(pid, int(idx)) for idx in indices.tolist()],
            )
        self.memory.db.execute(
            "UPDATE utility_page_stats SET feature_count=? WHERE page_id=?", (int(indices.size), pid)
        )
        self.memory.db.commit()
        return self.working_set.record_feedback(
            UtilityRecord(str(pid), str(row[0])), reward=reward
        )

    def refresh(self) -> UtilityRefreshStats:
        """Score matching pages through the inverted index and keep only top-K."""
        import heapq

        started = time.perf_counter()
        self.index_new_pages()
        db = self.memory.db
        active = np.flatnonzero(np.abs(self.working_set.learner.weights) > 1e-9)
        db.execute("DELETE FROM utility_active_weights")
        if active.size:
            db.executemany(
                "INSERT INTO utility_active_weights(feature_idx,weight) VALUES (?,?)",
                [(int(i), float(self.working_set.learner.weights[i])) for i in active.tolist()],
            )
            cur = db.execute(
                """SELECT f.page_id,SUM(w.weight),s.feature_count
                   FROM utility_page_features AS f
                   JOIN utility_active_weights AS w ON w.feature_idx=f.feature_idx
                   JOIN utility_page_stats AS s ON s.page_id=f.page_id
                   GROUP BY f.page_id"""
            )
        else:
            cur = iter(())

        heap: list[tuple[float, int]] = []
        candidate_pages = 0
        for page_id, weight_sum, feature_count in cur:
            candidate_pages += 1
            if int(feature_count) <= 0:
                continue
            score = float(weight_sum) / np.sqrt(float(feature_count))
            if score <= 0.0:
                continue
            item = (score, int(page_id))
            if len(heap) < self.working_set.capacity:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

        selected = {str(pid) for _, pid in heap}
        # Fill unused slots with most recent archive pages (bounded cursor).
        if len(selected) < self.working_set.capacity:
            cur_recent = db.execute("SELECT rowid FROM pages ORDER BY rowid DESC")
            for (page_id,) in cur_recent:
                selected.add(str(int(page_id)))
                if len(selected) >= self.working_set.capacity:
                    break
        self.working_set.cached_ids = selected
        self.working_set._feedback_since_refresh = 0
        threshold = min((score for score, _ in heap), default=0.0)
        stats = UtilityRefreshStats(
            scanned=candidate_pages,
            retained=len(selected),
            seconds=time.perf_counter() - started,
            threshold_score=float(threshold),
        )
        self.working_set.last_refresh = stats
        return stats
