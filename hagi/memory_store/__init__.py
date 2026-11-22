"""WAVE 219: external memory store (hagi_lab integration) —
SQLite/FTS5 episodic archive + memory agent. The full archive
persists regardless of working-set size; the ternary LM reranks
evidence (hagi_lab Wave7 architecture). Semantic vectors come
from embeddinggemma-2 (GemmaEmbedder), not Word2Vec."""
from hagi.memory_store.streaming import (  # noqa: F401
    DiskBackedContext, StreamingContext, MemoryHit)



class SemanticMemory:
    """FTS5 archive + gemma-embedding cosine rerank (replaces the
    labs Word2Vec/LSH layer with a stronger teacher-encoded index)."""

    def __init__(self, db_path: str, page_chars: int = 2048):
        self.store = DiskBackedContext(db_path, page_chars=page_chars)
        self._vectors = {}  # page_id -> np.ndarray (lazy gemma)

    def add(self, text: str) -> None:
        self.store.append(text)
        self.store.flush()

    def _ensure_vectors(self):
        # WAVE 230 FIX: incremental — pages added after the first
        # query were never embedded (stale semantic index). Embed
        # only the missing rowids.
        from hagi.embeddings import GemmaEmbedder
        gemma = GemmaEmbedder.get()
        self.store._ensure_open()
        rows = self.store.db.execute(
            "SELECT rowid, content FROM pages").fetchall()
        texts = {rid: content for rid, content in rows}
        missing = [pid for pid in texts if pid not in self._vectors]
        if missing:
            import numpy as np
            vecs = gemma.embed([texts[pid][:512] for pid in missing])
            for pid, v in zip(missing, vecs):
                self._vectors[pid] = (v.detach().cpu().numpy()
                                      if hasattr(v, "cpu")
                                      else np.asarray(v))
        self._pages = texts

    def search(self, q: str, limit: int = 4):
        """memory_agent-compatible alias: hits with .text attr."""
        res = self.query(q, top_k=limit)
        from hagi.memory_store.streaming import MemoryHit

        class _Hit:
            def __init__(self, pid, text):
                self.page_id = pid
                self.text = text
        return [_Hit(pid, text) for _cos, pid, text in res]

    def query(self, q: str, top_k: int = 4):
        """WAVE 230 FIX: genuinely hybrid now — FTS5 (bm25) picks the
        lexical candidate set, gemma-cosine reranks it. Falls back to
        full cosine sweep when FTS has no matches (rare-token queries)."""
        import numpy as np
        from hagi.embeddings import GemmaEmbedder
        gemma = GemmaEmbedder.get()
        self._ensure_vectors()
        qv = gemma.embed([q[:512]])[0]
        if hasattr(qv, "cpu"):
            qv = qv.detach().cpu().numpy()

        def _cos(pid):
            v = self._vectors[pid]
            return float(np.dot(qv, v))

        # lexical phase: FTS5 match on the query terms
        cand = None
        try:
            terms = [t for t in
                     __import__("re").findall(r"[A-Za-z0-9]{3,}", q)][:8]
            if terms:
                esc = " ".join('"' + t.replace('"', '""') + '"'
                               for t in terms)
                rows = self.store.db.execute(
                    "SELECT rowid FROM pages WHERE pages MATCH ? "
                    "LIMIT 64", (esc,)).fetchall()
                if rows:
                    cand = {r[0] for r in rows}
        except Exception:
            cand = None  # FTS malformed query -> full sweep
        ids = sorted(cand) if cand else sorted(self._vectors)
        scored = [(_cos(pid), pid, self._pages.get(pid, ""))
                  for pid in ids if pid in self._vectors]
        scored.sort(reverse=True)
        return scored[:top_k]
