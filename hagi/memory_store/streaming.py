"""Disk-backed, bounded-RAM context for arbitrarily long input streams.

This is a context substrate, not a claim that a finite neural state can
losslessly remember infinitely many independent facts.  Text is chunked and
indexed incrementally in SQLite FTS5; queries fetch only a bounded number of
matching pages.  The working set is bounded by the page size and query budget,
while persistent storage grows with the amount of ingested text.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import re
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class MemoryHit:
    page_id: int
    char_start: int
    char_end: int
    score: float
    text: str
    snippet: str


@dataclass(frozen=True)
class FactRecord:
    key: str
    revision: int
    value: str
    source: str


@dataclass(frozen=True)
class MemoryStats:
    pages: int
    indexed_chars: int
    pending_chars: int
    max_pending_chars: int
    page_chars: int
    db_bytes: int


class DiskBackedContext:
    """Incrementally index a long text stream without retaining it in RAM.

    Args:
        path: SQLite database path. ``:memory:`` is accepted for unit tests.
        page_chars: Upper bound for a stored text page, except for a single
            unbroken token longer than this value.
        max_query_pages: Maximum number of pages returned per query.

    FTS5 uses SQLite's unicode61 tokenizer. Search is lexical, not semantic;
    an embedding index can be added behind this interface later.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        page_chars: int = 4096,
        max_query_pages: int = 4,
    ) -> None:
        if page_chars < 128:
            raise ValueError("page_chars must be >= 128")
        if max_query_pages < 1:
            raise ValueError("max_query_pages must be >= 1")
        self.path = str(path)
        self.page_chars = int(page_chars)
        self.max_query_pages = int(max_query_pages)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS pages
                          USING fts5(content, tokenize='unicode61')""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS page_meta (
                          page_id INTEGER PRIMARY KEY,
                          char_start INTEGER NOT NULL,
                          char_end INTEGER NOT NULL,
                          char_count INTEGER NOT NULL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS fact_history (
                          fact_key TEXT NOT NULL,
                          revision INTEGER NOT NULL,
                          value TEXT NOT NULL,
                          source TEXT NOT NULL,
                          PRIMARY KEY(fact_key, revision))""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS fact_current (
                          fact_key TEXT PRIMARY KEY,
                          revision INTEGER NOT NULL,
                          value TEXT NOT NULL,
                          source TEXT NOT NULL)""")
        self.db.commit()
        self._pending = ""
        self._indexed_chars = int(self.db.execute(
            "SELECT COALESCE(MAX(char_end), 0) FROM page_meta"
        ).fetchone()[0])
        self._max_pending_chars = 0
        self._pages_since_commit = 0
        self._commit_every_pages = 64
        self._closed = False

    def _insert_page(self, text: str) -> None:
        if not text:
            return
        start = self._indexed_chars
        end = start + len(text)
        cur = self.db.execute("INSERT INTO pages(content) VALUES (?)", (text,))
        page_id = int(cur.lastrowid)
        self.db.execute(
            "INSERT INTO page_meta(page_id,char_start,char_end,char_count) VALUES (?,?,?,?)",
            (page_id, start, end, len(text)),
        )
        self._indexed_chars = end
        self._pages_since_commit += 1
        if self._pages_since_commit >= self._commit_every_pages:
            self.db.commit()
            self._pages_since_commit = 0

    def append(self, text: str) -> None:
        """Append one stream fragment; fragment size does not set page size."""
        self._ensure_open()
        if not isinstance(text, str):
            raise TypeError("append expects str")
        if not text:
            return
        # Consume large caller fragments incrementally. The store's own buffer
        # never grows beyond one page, even when ``text`` is much larger.
        offset = 0
        while offset < len(text):
            room = self.page_chars - len(self._pending)
            take = min(room, len(text) - offset)
            self._pending += text[offset:offset + take]
            offset += take
            self._max_pending_chars = max(self._max_pending_chars, len(self._pending))
            if len(self._pending) == self.page_chars:
                # Prefer a newline or whitespace boundary so tokens are not split.
                cut = self._pending.rfind("\n")
                if cut < self.page_chars // 2:
                    cut = self._pending.rfind(" ")
                if cut < self.page_chars // 2:
                    cut = self.page_chars
                else:
                    cut += 1
                page = self._pending[:cut]
                self._pending = self._pending[cut:]
                self._insert_page(page)
        # Pages are committed in small batches by _insert_page. This avoids a
        # synchronous commit for every page while keeping crash recovery bounded.

    def ingest(self, fragments: Iterable[str]) -> None:
        """Consume an iterable of fragments without joining it in memory."""
        for fragment in fragments:
            self.append(fragment)

    def flush(self) -> None:
        self._ensure_open()
        if self._pending:
            self._insert_page(self._pending)
            self._pending = ""
        self.db.commit()
        self._pages_since_commit = 0

    @staticmethod
    def _fts_query(query: str) -> str:
        # Whitelist lexical terms and quote each one. Never pass raw user input
        # as FTS syntax; SQL itself is parameterized as well.
        terms = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
        if not terms:
            return ""
        return " AND ".join('"' + t.replace('"', '""') + '"' for t in terms)

    def search(self, query: str, limit: int | None = None,
               page_ids: Iterable[int] | None = None) -> list[MemoryHit]:
        """Return bounded FTS matches; optionally restrict to a working-set page set."""
        self._ensure_open()
        self.flush()
        fts = self._fts_query(query)
        if not fts:
            return []
        k = self.max_query_pages if limit is None else int(limit)
        if k < 1:
            raise ValueError("limit must be >= 1")
        k = min(k, self.max_query_pages)
        if page_ids is None:
            rows = self.db.execute(
                """SELECT p.rowid, m.char_start, m.char_end, bm25(pages), p.content,
                          snippet(pages, 0, '[match]', '[/match]', ' … ', 40)
                   FROM pages AS p JOIN page_meta AS m ON m.page_id=p.rowid
                   WHERE pages MATCH ? ORDER BY bm25(pages) LIMIT ?""",
                (fts, k),
            ).fetchall()
        else:
            allowed = sorted({int(pid) for pid in page_ids})
            if not allowed:
                return []
            self.db.execute("CREATE TEMP TABLE IF NOT EXISTS active_page_filter (page_id INTEGER PRIMARY KEY)")
            self.db.execute("DELETE FROM active_page_filter")
            self.db.executemany("INSERT OR IGNORE INTO active_page_filter(page_id) VALUES (?)",
                                ((pid,) for pid in allowed))
            rows = self.db.execute(
                """SELECT p.rowid, m.char_start, m.char_end, bm25(pages), p.content,
                          snippet(pages, 0, '[match]', '[/match]', ' … ', 40)
                   FROM pages AS p JOIN page_meta AS m ON m.page_id=p.rowid
                   JOIN active_page_filter AS f ON f.page_id=p.rowid
                   WHERE pages MATCH ? ORDER BY bm25(pages) LIMIT ?""",
                (fts, k),
            ).fetchall()
        return [MemoryHit(int(r[0]), int(r[1]), int(r[2]), float(r[3]), str(r[4]), str(r[5]))
                for r in rows]

    @staticmethod
    def _normalize_fact_key(key: str) -> str:
        normalized = " ".join(str(key).casefold().split())
        if not normalized:
            raise ValueError("fact key must be non-empty")
        return normalized

    def store_fact(self, key: str, value: str, source: str = "") -> FactRecord:
        """Persist a structured fact, keeping both latest value and revision history.

        This is an explicit API: a language model or parser must decide which
        text is a fact and provide its key/value. Raw text is still retained in
        the FTS archive for later inspection and retrieval.
        """
        self._ensure_open()
        fact_key = self._normalize_fact_key(key)
        value = str(value)
        source = str(source)
        row = self.db.execute(
            "SELECT COALESCE(MAX(revision), 0) FROM fact_history WHERE fact_key=?",
            (fact_key,),
        ).fetchone()
        revision = int(row[0]) + 1
        self.db.execute(
            "INSERT INTO fact_history(fact_key,revision,value,source) VALUES (?,?,?,?)",
            (fact_key, revision, value, source),
        )
        self.db.execute(
            """INSERT INTO fact_current(fact_key,revision,value,source) VALUES (?,?,?,?)
               ON CONFLICT(fact_key) DO UPDATE SET
               revision=excluded.revision, value=excluded.value, source=excluded.source""",
            (fact_key, revision, value, source),
        )
        self.db.commit()
        self._pages_since_commit = 0
        return FactRecord(fact_key, revision, value, source)

    def get_fact(self, key: str) -> FactRecord | None:
        self._ensure_open()
        fact_key = self._normalize_fact_key(key)
        row = self.db.execute(
            "SELECT fact_key,revision,value,source FROM fact_current WHERE fact_key=?",
            (fact_key,),
        ).fetchone()
        return None if row is None else FactRecord(str(row[0]), int(row[1]), str(row[2]), str(row[3]))

    def fact_history(self, key: str) -> list[FactRecord]:
        self._ensure_open()
        fact_key = self._normalize_fact_key(key)
        rows = self.db.execute(
            "SELECT fact_key,revision,value,source FROM fact_history WHERE fact_key=? ORDER BY revision",
            (fact_key,),
        ).fetchall()
        return [FactRecord(str(r[0]), int(r[1]), str(r[2]), str(r[3])) for r in rows]

    def get_stats(self) -> MemoryStats:
        self._ensure_open()
        pages = int(self.db.execute("SELECT COUNT(*) FROM page_meta").fetchone()[0])
        self.db.commit()
        self._pages_since_commit = 0
        if self.path == ":memory:":
            db_bytes = 0
        else:
            db_bytes = sum(
                os.path.getsize(p) for p in (self.path, self.path + "-wal", self.path + "-shm")
                if os.path.exists(p)
            )
        return MemoryStats(
            pages=pages,
            indexed_chars=self._indexed_chars,
            pending_chars=len(self._pending),
            max_pending_chars=self._max_pending_chars,
            page_chars=self.page_chars,
            db_bytes=db_bytes,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("context store is closed")

    def close(self) -> None:
        if not self._closed:
            self.flush()
            self.db.close()
            self._closed = True

    def __enter__(self) -> "DiskBackedContext":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class StreamingContext:
    """Small prompt assembler over the disk-backed context index.

    It never concatenates the full history. ``build_context`` returns at most
    ``max_context_chars`` retrieved text plus the user's query.
    """

    def __init__(self, memory: DiskBackedContext, max_context_chars: int = 8192,
                 utility_index=None):
        if max_context_chars < 1:
            raise ValueError("max_context_chars must be >= 1")
        self.memory = memory
        self.max_context_chars = int(max_context_chars)
        self.utility_index = utility_index

    def search(self, query: str, top_k: int = 4) -> list[MemoryHit]:
        """Search the learned working set first, then fall back to the full archive."""
        if self.utility_index is None:
            return self.memory.search(query, limit=top_k)
        cached_ids = self.utility_index.working_set.cached_page_ids
        cached_hits = self.memory.search(query, limit=top_k, page_ids=cached_ids) if cached_ids else []
        if len(cached_hits) >= top_k:
            return cached_hits[:top_k]
        archive_hits = self.memory.search(query, limit=top_k)
        merged: list[MemoryHit] = []
        seen: set[tuple[int, int, int]] = set()
        for hit in cached_hits + archive_hits:
            key = (hit.page_id, hit.char_start, hit.char_end)
            if key not in seen:
                seen.add(key)
                merged.append(hit)
            if len(merged) >= top_k:
                break
        return merged

    def feedback(self, page_id: int, reward: float = 1.0):
        """Feed usefulness feedback from an accepted/rejected retrieved page."""
        if self.utility_index is None:
            raise RuntimeError("no utility index is attached to this StreamingContext")
        return self.utility_index.record_page_feedback(page_id, reward=reward)

    def refresh_utility_memory(self):
        """Refresh the bounded working set after a batch of feedback events."""
        if self.utility_index is None:
            raise RuntimeError("no utility index is attached to this StreamingContext")
        return self.utility_index.refresh()

    def ingest(self, fragments: Iterable[str]) -> None:
        self.memory.ingest(fragments)

    def build_context(self, query: str, top_k: int = 4) -> str:
        hits = self.search(query, top_k=top_k)
        query_tail = query[-self.max_context_chars:]
        budget = max(0, self.max_context_chars - len(query_tail) - 1)
        pieces: list[str] = []
        for hit in hits:
            if budget <= 0:
                break
            provenance = f"[memory page={hit.page_id} chars={hit.char_start}:{hit.char_end}]\n"
            # Use the FTS snippet around the matched terms rather than the
            # page prefix: the relevant evidence may be at the page's end.
            evidence = hit.snippet if hit.snippet else hit.text
            piece = (provenance + evidence)[:budget]
            if piece:
                pieces.append(piece)
                budget -= len(piece) + 1
        retrieved = "\n".join(pieces)
        return (retrieved + "\n" if retrieved else "") + query_tail
