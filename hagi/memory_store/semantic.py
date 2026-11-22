"""Local, corpus-trained semantic retrieval for disk-backed HAGI memory.

This module does not download pretrained weights.  A small Word2Vec model is
trained from the indexed archive itself, and sentence/record vectors are
indexed with a bounded-table random-hyperplane LSH.  It is a practical
associative-retrieval layer, not a claim of human-level semantic understanding.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import sqlite3
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .streaming import DiskBackedContext, MemoryHit

_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_DECISION_RE = re.compile(
    r"\b(decision|selected|chosen|choose|chose|decided|agreed|approved|adopted|"
    r"settled on|final choice|final decision|configured|set to|opted for|we use|we chose)\b",
    flags=re.IGNORECASE,
)
_STOP = {
    "a", "an", "the", "and", "or", "but", "if", "then", "we", "i", "you",
    "he", "she", "it", "they", "our", "my", "your", "what", "which", "who",
    "where", "when", "why", "how", "did", "do", "does", "was", "were", "is",
    "are", "be", "been", "being", "to", "of", "in", "on", "at", "by", "for",
    "from", "with", "as", "that", "this", "these", "those", "have", "has",
    "had", "about", "into", "over", "after", "before", "than", "so", "not",
}


def tokenize(text: str) -> list[str]:
    return [t.casefold() for t in _TOKEN_RE.findall(text) if len(t) > 1]


def _segments(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield non-empty newline-delimited records with offsets in the page."""
    offset = 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        start, end = offset, offset + len(body)
        if body.strip():
            yield start, end, body.strip()
        offset += len(line)
    if not text.endswith(("\n", "\r")) and offset < len(text):
        body = text[offset:]
        if body.strip():
            yield offset, len(text), body.strip()


@dataclass(frozen=True)
class SemanticBuildStats:
    segments: int
    vocabulary: int
    vector_size: int
    tables: int
    bits_per_table: int
    build_seconds: float


@dataclass(frozen=True)
class SemanticSearchStats:
    candidates: int
    query_oov_tokens: int
    latency_ms: float


class _CorpusView:
    """Re-iterable SQLite corpus; each pass opens a fresh cursor."""
    def __init__(self, memory: DiskBackedContext):
        self.memory = memory

    def __iter__(self):
        if self.memory.path == ":memory:":
            raise ValueError("local semantic training currently requires a file-backed SQLite archive")
        # Gensim's producer thread must not share the caller's SQLite handle.
        uri = f"file:{Path(self.memory.path).resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            cur = conn.execute("SELECT content FROM pages ORDER BY rowid")
            for (content,) in cur:
                for _, _, body in _segments(str(content)):
                    words = [w for w in tokenize(body) if w not in _STOP]
                    if words:
                        yield words
        finally:
            conn.close()


class LocalSemanticIndex:
    """Corpus-trained dense vectors + disk-backed LSH candidate index.

    Memory growth is deliberately capped for the model vocabulary and bounded
    for each search. Persistent segment/bucket storage grows with the archive.
    The LSH is approximate: recall depends on corpus, vector quality and table
    configuration, so applications should retain lexical fallback.
    """

    def __init__(
        self,
        memory: DiskBackedContext,
        model_path: str | Path | None = None,
        vector_size: int = 48,
        window: int = 6,
        min_count: int = 2,
        max_vocab: int = 30_000,
        epochs: int = 5,
        tables: int = 10,
        bits_per_table: int = 14,
        seed: int = 17,
        max_candidates: int = 3000,
        bucket_limit: int = 512,
    ) -> None:
        if vector_size < 8 or window < 1 or min_count < 1 or max_vocab < 100:
            raise ValueError("invalid embedding configuration")
        if epochs < 1 or tables < 1 or not 4 <= bits_per_table <= 20:
            raise ValueError("invalid LSH/training configuration")
        if max_candidates < 1 or bucket_limit < 1:
            raise ValueError("candidate limits must be positive")
        self.memory = memory
        self.vector_size = int(vector_size)
        self.window = int(window)
        self.min_count = int(min_count)
        self.max_vocab = int(max_vocab)
        self.epochs = int(epochs)
        self.tables = int(tables)
        self.bits_per_table = int(bits_per_table)
        self.seed = int(seed)
        self.max_candidates = int(max_candidates)
        self.bucket_limit = int(bucket_limit)
        if model_path is None:
            model_path = None if memory.path == ":memory:" else memory.path + ".w2v"
        self.model_path = None if model_path is None else str(model_path)
        self.model = None
        self.planes: np.ndarray | None = None
        self.last_search_stats = SemanticSearchStats(0, 0, 0.0)
        self._create_schema()

    def _create_schema(self) -> None:
        db = self.memory.db
        db.execute("""CREATE TABLE IF NOT EXISTS semantic_meta (
            meta_key TEXT PRIMARY KEY, meta_value BLOB NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS semantic_segments (
            segment_id INTEGER PRIMARY KEY,
            page_id INTEGER NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            salience REAL NOT NULL DEFAULT 0.0)""")
        seg_cols = {str(row[1]) for row in db.execute("PRAGMA table_info(semantic_segments)")}
        if "salience" not in seg_cols:
            db.execute("ALTER TABLE semantic_segments ADD COLUMN salience REAL NOT NULL DEFAULT 0.0")
        db.execute("""CREATE TABLE IF NOT EXISTS semantic_lsh (
            table_id INTEGER NOT NULL,
            signature INTEGER NOT NULL,
            segment_id INTEGER NOT NULL,
            PRIMARY KEY(table_id, signature, segment_id)) WITHOUT ROWID""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_semantic_segment_page ON semantic_segments(page_id)")
        db.commit()

    def _load_gensim(self):
        try:
            from gensim.models import Word2Vec
        except Exception as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError(
                "Local semantic retrieval requires the optional 'gensim>=4.4' dependency"
            ) from exc
        return Word2Vec

    def fit(self, force: bool = False) -> SemanticBuildStats:
        """Train embeddings from existing memory and construct the LSH index."""
        import time

        started = time.perf_counter()
        Word2Vec = self._load_gensim()
        if self.memory.path == ":memory:":
            raise ValueError("local semantic training currently requires a file-backed SQLite archive")
        self.memory.flush()
        self.memory.db.execute("SELECT COUNT(*) FROM pages").fetchone()
        current_pages = int(self.memory.db.execute("SELECT COUNT(*) FROM pages").fetchone()[0])
        source_fingerprint = {
            "pages": current_pages,
            "chars": self.memory._indexed_chars,
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "max_vocab": self.max_vocab,
            "epochs": self.epochs,
            "tables": self.tables,
            "bits_per_table": self.bits_per_table,
            "seed": self.seed,
            "index_version": 2,
        }
        if not force and self.model_path and Path(self.model_path).exists():
            row = self.memory.db.execute(
                "SELECT meta_value FROM semantic_meta WHERE meta_key='source_fingerprint'"
            ).fetchone()
            complete_row = self.memory.db.execute(
                "SELECT meta_value FROM semantic_meta WHERE meta_key='index_complete'"
            ).fetchone()
            if (row is not None and complete_row is not None and complete_row[0] == b"1"
                    and row[0].decode("utf-8") == json.dumps(source_fingerprint, sort_keys=True)):
                self.model = Word2Vec.load(self.model_path)
                plane_row = self.memory.db.execute(
                    "SELECT meta_value FROM semantic_meta WHERE meta_key='planes'"
                ).fetchone()
                if plane_row is not None:
                    arr = np.frombuffer(plane_row[0], dtype=np.float32)
                    expected = self.tables * self.bits_per_table * self.vector_size
                    if arr.size == expected:
                        self.planes = arr.reshape(self.tables, self.bits_per_table, self.vector_size).copy()
                        existing = int(self.memory.db.execute(
                            "SELECT COUNT(*) FROM semantic_segments"
                        ).fetchone()[0])
                        if existing:
                            return SemanticBuildStats(existing, len(self.model.wv), self.vector_size,
                                                      self.tables, self.bits_per_table,
                                                      time.perf_counter() - started)

        model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1,
            sg=1,
            negative=5,
            sample=1e-3,
            seed=self.seed,
            max_final_vocab=self.max_vocab,
            sorted_vocab=1,
        )
        corpus = _CorpusView(self.memory)
        model.build_vocab(corpus_iterable=corpus)
        if not model.wv:
            raise ValueError("archive does not contain enough repeated tokens to train embeddings")
        model.train(corpus_iterable=corpus, total_examples=model.corpus_count,
                    epochs=self.epochs)
        self.model = model
        if self.model_path:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(self.model_path)

        rng = np.random.default_rng(self.seed)
        planes = rng.normal(size=(self.tables, self.bits_per_table, self.vector_size)).astype(np.float32)
        norms = np.linalg.norm(planes, axis=2, keepdims=True)
        planes /= np.maximum(norms, 1e-12)
        self.planes = planes

        db = self.memory.db
        db.execute("DELETE FROM semantic_lsh")
        db.execute("DELETE FROM semantic_segments")
        db.execute("DELETE FROM semantic_meta")
        db.execute("INSERT INTO semantic_meta(meta_key,meta_value) VALUES ('planes',?)",
                   (planes.tobytes(),))
        db.execute("INSERT INTO semantic_meta(meta_key,meta_value) VALUES ('config',?)",
                   (f"{self.vector_size},{self.tables},{self.bits_per_table},{self.seed}".encode(),))
        db.execute("INSERT INTO semantic_meta(meta_key,meta_value) VALUES ('source_fingerprint',?)",
                   (json.dumps(source_fingerprint, sort_keys=True).encode(),))
        db.execute("INSERT INTO semantic_meta(meta_key,meta_value) VALUES ('index_complete',?)", (b"0",))
        db.commit()

        insert_segment = """INSERT INTO semantic_segments
            (page_id,char_start,char_end,content,embedding,salience) VALUES (?,?,?,?,?,?)"""
        insert_lsh = "INSERT INTO semantic_lsh(table_id,signature,segment_id) VALUES (?,?,?)"
        batch_segments: list[tuple] = []
        batch_lsh: list[tuple[int, int, int]] = []
        seg_count = 0
        page_cur = db.execute(
            "SELECT p.rowid,m.char_start,p.content FROM pages p "
            "JOIN page_meta m ON m.page_id=p.rowid ORDER BY p.rowid"
        )
        for page_id, page_start, content in page_cur:
            for local_start, local_end, body in _segments(str(content)):
                tokens = [w for w in tokenize(body) if w not in _STOP and w in model.wv]
                if not tokens:
                    continue
                vec = np.mean(np.stack([model.wv[w] for w in tokens]), axis=0).astype(np.float32)
                norm = float(np.linalg.norm(vec))
                if norm < 1e-12:
                    continue
                vec /= norm
                sigs = self._signatures(vec, planes)
                seg_id = seg_count + 1
                salience = 0.025 if _DECISION_RE.search(body) else 0.0
                batch_segments.append((int(page_id), int(page_start) + local_start,
                                       int(page_start) + local_end, body,
                                       vec.astype(np.float16).tobytes(), salience))
                # SQLite segment ids are contiguous from 1 after DELETE.
                batch_lsh.extend((table_id, signature, seg_id)
                                 for table_id, signature in enumerate(sigs))
                seg_count += 1
                if len(batch_segments) >= 2000:
                    db.executemany(insert_segment, batch_segments)
                    db.executemany(insert_lsh, batch_lsh)
                    db.commit()
                    batch_segments.clear()
                    batch_lsh.clear()
        if batch_segments:
            db.executemany(insert_segment, batch_segments)
            db.executemany(insert_lsh, batch_lsh)
        db.execute("UPDATE semantic_meta SET meta_value=? WHERE meta_key='index_complete'", (b"1",))
        db.commit()
        return SemanticBuildStats(seg_count, len(model.wv), self.vector_size,
                                  self.tables, self.bits_per_table,
                                  time.perf_counter() - started)

    @staticmethod
    def _signatures(vec: np.ndarray, planes: np.ndarray) -> list[int]:
        projections = np.einsum("tbd,d->tb", planes, vec)
        signs = projections >= 0
        values = []
        for row in signs:
            signature = 0
            for bit, enabled in enumerate(row.tolist()):
                if enabled:
                    signature |= 1 << bit
            values.append(signature)
        return values

    def _query_vector(self, query: str) -> tuple[np.ndarray | None, int]:
        if self.model is None or self.planes is None:
            raise RuntimeError("call fit() before semantic search")
        words = [w for w in tokenize(query) if w not in _STOP]
        known = [w for w in words if w in self.model.wv]
        oov = len(words) - len(known)
        if not known:
            return None, oov
        vec = np.mean(np.stack([self.model.wv[w] for w in known]), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return None, oov
        return vec / norm, oov

    def search_exact(self, query: str, limit: int = 4, use_salience: bool = True) -> list[MemoryHit]:
        """Exact cosine top-k diagnostic that scans every indexed segment.

        This is O(number of segments) and is intended for benchmark validation,
        not the production retrieval path. It keeps only a top-k heap in RAM.
        """
        import heapq
        import time

        if limit < 1:
            raise ValueError("limit must be >= 1")
        started = time.perf_counter()
        qvec, oov = self._query_vector(query)
        if qvec is None:
            self.last_search_stats = SemanticSearchStats(0, oov, (time.perf_counter() - started) * 1000)
            return []
        heap: list[tuple[float, int, tuple]] = []
        scanned = 0
        cur = self.memory.db.execute(
            "SELECT segment_id,page_id,char_start,char_end,content,embedding,salience "
            "FROM semantic_segments ORDER BY segment_id"
        )
        for row in cur:
            scanned += 1
            emb = np.frombuffer(row[5], dtype=np.float16).astype(np.float32)
            score = float(np.dot(qvec, emb))
            if use_salience:
                score += float(row[6])
            item = (score, -int(row[0]), row)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
        best = sorted(heap, key=lambda item: item[:2], reverse=True)
        hits = [MemoryHit(int(row[1]), int(row[2]), int(row[3]), float(score),
                          str(row[4]), str(row[4])) for score, _, row in best]
        self.last_search_stats = SemanticSearchStats(
            scanned, oov, (time.perf_counter() - started) * 1000
        )
        return hits

    def search(self, query: str, limit: int = 4, use_salience: bool = True) -> list[MemoryHit]:
        """Approximate nearest-neighbour retrieval with bounded candidates.

        ``use_salience`` adds a small score bonus to explicitly marked decisions
        and choices.  It is a transparent heuristic, not a learned truth test.
        """
        import time

        if limit < 1:
            raise ValueError("limit must be >= 1")
        started = time.perf_counter()
        qvec, oov = self._query_vector(query)
        if qvec is None:
            self.last_search_stats = SemanticSearchStats(0, oov, (time.perf_counter() - started) * 1000)
            return []
        assert self.planes is not None
        signatures = self._signatures(qvec, self.planes)
        db = self.memory.db
        candidates: dict[int, tuple] = {}
        mask_bits = (1 << self.bits_per_table) - 1
        query_sql = """SELECT s.segment_id,s.page_id,s.char_start,s.char_end,s.content,s.embedding,s.salience
            FROM semantic_lsh b JOIN semantic_segments s ON s.segment_id=b.segment_id
            WHERE b.table_id=? AND b.signature IN ({})
            ORDER BY ((s.segment_id * 1103515245 + b.table_id * 12345) & 2147483647)
            LIMIT ?"""
        for table_id, signature in enumerate(signatures):
            probes = [signature]
            probes.extend(signature ^ (1 << bit) for bit in range(self.bits_per_table))
            probes = [p & mask_bits for p in probes]
            placeholders = ",".join("?" for _ in probes)
            rows = db.execute(query_sql.format(placeholders),
                              (table_id, *probes, self.bucket_limit)).fetchall()
            for row in rows:
                sid = int(row[0])
                if sid not in candidates:
                    candidates[sid] = row

        candidate_rows = list(candidates.values())
        if len(candidate_rows) > self.max_candidates:
            # Deterministic pseudo-random subsampling keeps all LSH tables in play
            # instead of letting the first table consume the whole candidate budget.
            candidate_rows.sort(key=lambda r: ((int(r[0]) * 1103515245 + self.seed) & 0x7FFFFFFF))
            candidate_rows = candidate_rows[: self.max_candidates]

        scored: list[tuple[float, tuple]] = []
        for row in candidate_rows:
            emb = np.frombuffer(row[5], dtype=np.float16).astype(np.float32)
            score = float(np.dot(qvec, emb))
            if use_salience:
                score += float(row[6])
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [MemoryHit(int(r[1]), int(r[2]), int(r[3]), float(score), str(r[4]), str(r[4]))
                for score, r in scored[:limit]]
        self.last_search_stats = SemanticSearchStats(
            len(candidate_rows), oov, (time.perf_counter() - started) * 1000
        )
        return hits


class HybridStreamingContext:
    """Semantic retrieval with a lexical fallback and bounded prompt assembly."""

    def __init__(self, memory: DiskBackedContext, semantic: LocalSemanticIndex,
                 max_context_chars: int = 8192):
        if max_context_chars < 1:
            raise ValueError("max_context_chars must be >= 1")
        self.memory = memory
        self.semantic = semantic
        self.max_context_chars = int(max_context_chars)

    def build_context(self, query: str, top_k: int = 4) -> str:
        semantic_hits = self.semantic.search(query, limit=top_k)
        lexical_hits = self.memory.search(query, limit=top_k)
        merged: list[MemoryHit] = []
        seen = set()
        for hit in semantic_hits + lexical_hits:
            key = (hit.page_id, hit.char_start, hit.char_end)
            if key not in seen:
                seen.add(key)
                merged.append(hit)
        query_tail = query[-self.max_context_chars :]
        budget = max(0, self.max_context_chars - len(query_tail) - 1)
        pieces: list[str] = []
        for hit in merged:
            if budget <= 0:
                break
            provenance = f"[memory page={hit.page_id} chars={hit.char_start}:{hit.char_end}]\n"
            evidence = hit.snippet or hit.text
            piece = (provenance + evidence)[:budget]
            if piece:
                pieces.append(piece)
                budget -= len(piece) + 1
        context = "\n".join(pieces)
        return (context + "\n" if context else "") + query_tail
