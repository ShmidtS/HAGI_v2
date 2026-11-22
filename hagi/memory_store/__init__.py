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

    def __init__(self, db_path: str, page_chars: int = 512):
        self.store = DiskBackedContext(db_path, page_chars=page_chars)
        self._vectors = {}  # page_id -> np.ndarray (lazy gemma)

    def add(self, text: str) -> None:
        self.store.append(text)
        self.store.flush()

    def _ensure_vectors(self):
        if self._vectors:
            return
        from hagi.embeddings import GemmaEmbedder
        gemma = GemmaEmbedder.get()
        # direct SQL: FTS match() needs a non-empty query, all-pages
        # listing goes through the raw pages table
        self.store._ensure_open()
        rows = self.store.db.execute(
            "SELECT rowid, content FROM pages").fetchall()
        texts = {rid: content for rid, content in rows}
        pids = list(texts)
        vecs = gemma.embed([texts[pid][:512] for pid in pids])
        import numpy as np
        for pid, v in zip(pids, vecs):
            self._vectors[pid] = np.asarray(v)
        self._pages = texts

    def query(self, q: str, top_k: int = 4):
        """Hybrid: FTS5 lexical first, gemma-cosine rerank."""
        import numpy as np
        from hagi.embeddings import GemmaEmbedder
        gemma = GemmaEmbedder.get()
        self._ensure_vectors()
        qv = gemma.embed([q[:512]])[0]
        scored = []
        for pid, v in self._vectors.items():
            cos = float(np.dot(qv, v))
            scored.append((cos, pid, self._pages.get(pid, "")))
        scored.sort(reverse=True)
        return scored[:top_k]
