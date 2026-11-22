"""Locally trained write-policy controller for long-term memory.

The controller classifies individual records as transient, factual, or decision-like.
It is intentionally small and CPU-friendly. It is a write filter, not an autonomous
truth detector: labels and training distribution bound what it can learn.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .streaming import DiskBackedContext, MemoryHit

_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_CLASSES = ("transient", "fact", "decision")


def _stable_hash(token: str) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little", signed=False)
    return value, (-1.0 if (value >> 63) else 1.0)


@dataclass(frozen=True)
class PolicyMetrics:
    accuracy: float
    macro_f1: float
    keep_precision: float
    keep_recall: float
    keep_f1: float
    decision_recall: float
    retention_rate: float
    n: int


@dataclass(frozen=True)
class PolicyFitReport:
    epochs: int
    best_validation_loss: float
    train_seconds: float
    train_metrics: PolicyMetrics
    validation_metrics: PolicyMetrics


class _PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MemoryPolicyController:
    """Tiny trainable classifier that decides what should enter durable memory.

    Feature hashing makes input dimensionality fixed regardless of archive size.
    No pretrained model or network access is used.
    """

    def __init__(self, feature_bins: int = 256, hidden: int = 48, seed: int = 31):
        if feature_bins < 64 or hidden < 8:
            raise ValueError("feature_bins must be >=64 and hidden >=8")
        self.feature_bins = int(feature_bins)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.input_dim = self.feature_bins + 13
        torch.manual_seed(self.seed)
        self.net = _PolicyNet(self.input_dim, self.hidden)
        self.trained = False
        self.fit_report: PolicyFitReport | None = None

    def featurize_one(self, text: str) -> np.ndarray:
        x = np.zeros(self.input_dim, dtype=np.float32)
        tokens = [t.casefold() for t in _TOKEN_RE.findall(str(text))]
        # Signed feature hashing of unigrams and adjacent bigrams.
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for token in features:
            hv, sign = _stable_hash(token)
            x[hv % self.feature_bins] += sign
        if features:
            x[:self.feature_bins] /= np.sqrt(float(len(features)))
        low = str(text).casefold()
        tail = x[self.feature_bins:]
        cue_groups = (
            ("decision", "decided", "agreed", "approved", "selected", "final choice", "settled on", "commitment", "binding choice"),
            ("remember", "fact", "reference", "documented", "verified", "record indicates", "for future lookup", "reference detail"),
            ("temporary", "today", "briefly", "scratchpad", "passing thought", "just for now", "speculative option"),
            ("maybe", "might", "perhaps", "could", "unconfirmed", "not decided", "not final", "possible idea"),
            ("ignore", "discard", "not retain", "do not keep", "do not save", "not to retain"),
            ("must", "required", "constraint", "always", "never", "binding"),
        )
        for i, group in enumerate(cue_groups):
            tail[i] = float(any(cue in low for cue in group))
        # Explicit negation/ephemerality should override positive-looking nouns
        # such as "fact" in the phrase "not a retained fact".
        tail[6] = float(any(cue in low for cue in (
            "not a retained fact", "not a verified fact", "not something to keep",
            "do not save", "not to retain", "no lasting information", "ignore this",
        )))
        tail[7] = float(any(cue in low for cue in (
            "not a decision", "not decided", "not final", "not approved", "not selected",
        )))
        tail[8] = float(any(cue in low for cue in (
            "final choice", "binding decision", "we agreed", "settled on", "commitment:",
        )))
        tail[9] = min(len(tokens), 100) / 100.0
        tail[10] = float(any(ch.isdigit() for ch in low))
        tail[11] = float("?" in low)
        tail[12] = min(sum(ch in "!?" for ch in low), 5) / 5.0
        return x

    def featurize(self, texts: Iterable[str]) -> np.ndarray:
        return np.stack([self.featurize_one(t) for t in texts]).astype(np.float32)

    @staticmethod
    def _metrics(y_true: np.ndarray, probs: np.ndarray) -> PolicyMetrics:
        pred = probs.argmax(axis=1)
        labels = (0, 1, 2)
        f1s = []
        for label in labels:
            tp = int(np.sum((pred == label) & (y_true == label)))
            fp = int(np.sum((pred == label) & (y_true != label)))
            fn = int(np.sum((pred != label) & (y_true == label)))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
        keep_pred = pred != 0
        keep_true = y_true != 0
        tp = int(np.sum(keep_pred & keep_true))
        fp = int(np.sum(keep_pred & ~keep_true))
        fn = int(np.sum(~keep_pred & keep_true))
        kp = tp / max(tp + fp, 1)
        kr = tp / max(tp + fn, 1)
        kf = 2 * kp * kr / max(kp + kr, 1e-12)
        decision_total = int(np.sum(y_true == 2))
        decision_recall = float(np.sum((pred == 2) & (y_true == 2)) / max(decision_total, 1))
        return PolicyMetrics(
            accuracy=float(np.mean(pred == y_true)), macro_f1=float(np.mean(f1s)),
            keep_precision=kp, keep_recall=kr, keep_f1=kf,
            decision_recall=decision_recall, retention_rate=float(np.mean(keep_pred)),
            n=int(len(y_true)),
        )

    def fit(self, texts: list[str], labels: list[int] | np.ndarray,
            validation_fraction: float = 0.2, epochs: int = 120,
            lr: float = 0.01, patience: int = 15) -> PolicyFitReport:
        if len(texts) != len(labels) or len(texts) < 30:
            raise ValueError("texts and labels must have equal length >=30")
        if not 0.05 <= validation_fraction <= 0.4 or epochs < 1 or lr <= 0 or patience < 1:
            raise ValueError("invalid training configuration")
        y = np.asarray(labels, dtype=np.int64)
        if set(np.unique(y).tolist()) != {0, 1, 2}:
            raise ValueError("labels must contain all classes 0, 1, 2")
        x = self.featurize(texts)
        rng = np.random.default_rng(self.seed)
        train_idx: list[int] = []
        valid_idx: list[int] = []
        for cls in (0, 1, 2):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            n_valid = max(1, int(round(len(idx) * validation_fraction)))
            valid_idx.extend(idx[:n_valid].tolist())
            train_idx.extend(idx[n_valid:].tolist())
        rng.shuffle(train_idx)
        rng.shuffle(valid_idx)
        xt = torch.from_numpy(x[train_idx])
        yt = torch.from_numpy(y[train_idx])
        xv = torch.from_numpy(x[valid_idx])
        yv = torch.from_numpy(y[valid_idx])
        torch.manual_seed(self.seed)
        self.net = _PolicyNet(self.input_dim, self.hidden)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        best_loss = float("inf")
        bad_epochs = 0
        start = time.perf_counter()
        best_epoch = 0
        for ep in range(1, epochs + 1):
            self.net.train()
            order = torch.randperm(len(xt))
            for batch in order.split(128):
                logits = self.net(xt[batch])
                loss = F.cross_entropy(logits, yt[batch])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = float(F.cross_entropy(self.net(xv), yv))
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_epoch = ep
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        elapsed = time.perf_counter() - start
        with torch.no_grad():
            train_probs = torch.softmax(self.net(xt), dim=1).numpy()
            valid_probs = torch.softmax(self.net(xv), dim=1).numpy()
        self.trained = True
        train_report = self._metrics(y[train_idx], train_probs)
        valid_report = self._metrics(y[valid_idx], valid_probs)
        self.fit_report = PolicyFitReport(best_epoch, best_loss, elapsed, train_report, valid_report)
        return self.fit_report

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("fit or load the memory policy before prediction")
        if not texts:
            return np.zeros((0, 3), dtype=np.float32)
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(torch.from_numpy(self.featurize(texts))), dim=1).numpy()

    def predict(self, text: str) -> tuple[str, float, np.ndarray]:
        probs = self.predict_proba([text])[0]
        cls = int(np.argmax(probs))
        return _CLASSES[cls], float(probs[cls]), probs

    def save(self, path: str | Path) -> None:
        if not self.trained:
            raise RuntimeError("cannot save an untrained policy")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"feature_bins": self.feature_bins, "hidden": self.hidden,
                    "seed": self.seed, "state_dict": self.net.state_dict(),
                    "classes": list(_CLASSES), "version": 1}, target)

    @classmethod
    def load(cls, path: str | Path) -> "MemoryPolicyController":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("version") != 1 or payload.get("classes") != list(_CLASSES):
            raise ValueError("unsupported memory-policy checkpoint")
        obj = cls(payload["feature_bins"], payload["hidden"], payload["seed"])
        obj.net.load_state_dict(payload["state_dict"])
        obj.net.eval()
        obj.trained = True
        return obj

    def attach(self, memory: DiskBackedContext) -> "PolicyMemoryStore":
        if not self.trained:
            raise RuntimeError("fit or load the memory policy before attaching it")
        return PolicyMemoryStore(memory, self)


class PolicyMemoryStore:
    """A gated durable-memory table attached to the raw disk-backed archive."""

    def __init__(self, memory: DiskBackedContext, policy: MemoryPolicyController):
        self.memory = memory
        self.policy = policy
        db = memory.db
        db.execute("""CREATE TABLE IF NOT EXISTS policy_memory (
            memory_id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            content_hash TEXT NOT NULL UNIQUE)""")
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS policy_memory_fts USING fts5(content, content='policy_memory', content_rowid='memory_id')")
        db.execute("""CREATE TRIGGER IF NOT EXISTS policy_memory_ai AFTER INSERT ON policy_memory BEGIN
            INSERT INTO policy_memory_fts(rowid,content) VALUES (new.memory_id,new.content);
        END""")
        db.execute("""CREATE TRIGGER IF NOT EXISTS policy_memory_ad AFTER DELETE ON policy_memory BEGIN
            INSERT INTO policy_memory_fts(policy_memory_fts,rowid,content) VALUES('delete',old.memory_id,old.content);
        END""")
        db.commit()

    def ingest(self, records: Iterable[str], source: str = "", keep_threshold: float = 0.5,
               keep_decisions: bool = True) -> dict:
        if not 0.0 <= keep_threshold <= 1.0:
            raise ValueError("keep_threshold must be in [0,1]")
        seen = kept = duplicates = 0
        for record in records:
            text = str(record).strip()
            if not text:
                continue
            seen += 1
            start = self.memory._indexed_chars + len(self.memory._pending)
            category, confidence, probs = self.policy.predict(text)
            retain_prob = float(probs[1] + probs[2])
            decision_prob = float(probs[2])
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            self.memory.append(text + "\n")
            if retain_prob >= keep_threshold and (keep_decisions or category != "decision"):
                cur = self.memory.db.execute(
                    """INSERT OR IGNORE INTO policy_memory
                    (content,category,confidence,source,char_start,char_end,content_hash)
                    VALUES (?,?,?,?,?,?,?)""",
                    (text, category, retain_prob, str(source), start, start + len(text), digest),
                )
                if cur.rowcount:
                    kept += 1
                else:
                    duplicates += 1
        self.memory.flush()
        self.memory.db.commit()
        total = int(self.memory.db.execute("SELECT COUNT(*) FROM policy_memory").fetchone()[0])
        return {"seen": seen, "kept": kept, "duplicates": duplicates,
                "total_retained": total, "retention_rate": kept / max(seen, 1)}

    def search(self, query: str, limit: int = 5) -> list[MemoryHit]:
        if limit < 1:
            raise ValueError("limit must be >=1")
        fts = self.memory._fts_query(query)
        if not fts:
            return []
        rows = self.memory.db.execute(
            """SELECT p.memory_id,p.char_start,p.char_end,bm25(policy_memory_fts),p.content,
                      snippet(policy_memory_fts,0,'[match]','[/match]',' … ',32)
               FROM policy_memory_fts f JOIN policy_memory p ON p.memory_id=f.rowid
               WHERE policy_memory_fts MATCH ? ORDER BY bm25(policy_memory_fts) LIMIT ?""",
            (fts, int(limit)),
        ).fetchall()
        return [MemoryHit(int(r[0]), int(r[1]), int(r[2]), float(r[3]), str(r[4]), str(r[5]))
                for r in rows]

    def stats(self) -> dict:
        total_raw = int(self.memory.db.execute("SELECT COUNT(*) FROM page_meta").fetchone()[0])
        retained = int(self.memory.db.execute("SELECT COUNT(*) FROM policy_memory").fetchone()[0])
        return {"raw_pages": total_raw, "retained_records": retained}
