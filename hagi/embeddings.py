"""WAVE 217: embeddinggemma-2 (user-directed) — semantic vectors
for the symbolic/neighborhood layer.

"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_SNAP = (
    r"C:\Users\shmid\.cache\huggingface\hub"
    r"\models--google--embeddinggemma-2\snapshots"
    r"\f9972ac22e696a105c482202646979e33e4b5a1e"
)


class GemmaEmbedder:
    """Lazy single-instance wrapper over embeddinggemma-2 (768-dim,
    mean-pool + L2). Offline from the local HF cache."""

    _instance = None

    def __init__(self, snapshot: str | None = None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoModel, AutoTokenizer

        self.snapshot = snapshot or os.environ.get("HAGI_EMBED_SNAPSHOT", _DEFAULT_SNAP)
        if not Path(self.snapshot).exists():
            raise FileNotFoundError(
                f"embeddinggemma-2 snapshot not found: {self.snapshot}"
            )
        self.tok = AutoTokenizer.from_pretrained(self.snapshot)
        self.model = AutoModel.from_pretrained(self.snapshot)
        self.model.eval()

    @classmethod
    def get(cls) -> "GemmaEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, texts: list[str]):
        """L2-normalised mean-pooled embeddings, torch [N, 768]."""
        import torch

        b = self.tok(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = self.model(**b)
        mask = b["attention_mask"].unsqueeze(-1).float()
        v = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(v, dim=-1)

    def similarity(self, a: str, b: str) -> float:
        v = self.embed([a, b])
        return float(v[0] @ v[1])
