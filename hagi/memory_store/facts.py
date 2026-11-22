"""Conflict-aware structured facts for persistent HAGI memory.

This layer prevents an unverified contradictory observation from silently
replacing an accepted value. It preserves every proposal and requires either
sufficient evidence or an explicit correction to update the active fact.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

from .streaming import DiskBackedContext


@dataclass(frozen=True)
class FactSnapshot:
    key: str
    revision: int
    value: str
    source: str
    confidence: float
    authority: float
    history_id: int
    updated_ns: int


@dataclass(frozen=True)
class FactAnswer:
    key: str
    status: str
    value: str | None
    current: FactSnapshot | None
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactProposal:
    key: str
    proposed_value: str
    status: str
    accepted: bool
    current: FactSnapshot
    confidence: float
    authority: float
    history_id: int


class ConflictAwareFactStore:
    """Append-only evidence ledger with guarded current-value updates.

    ``confidence`` estimates the quality of this observation; ``authority``
    estimates the source's standing. Neither is a truth probability supplied
    by the system automatically: callers must provide or derive them.
    """

    def __init__(self, memory: DiskBackedContext, min_confidence_gain: float = 0.15,
                 authority_override: float = 0.20, authority_override_min_confidence: float = 0.60):
        if not 0.0 <= min_confidence_gain <= 1.0:
            raise ValueError("min_confidence_gain must be in [0,1]")
        if not 0.0 <= authority_override <= 1.0:
            raise ValueError("authority_override must be in [0,1]")
        if not 0.0 <= authority_override_min_confidence <= 1.0:
            raise ValueError("authority_override_min_confidence must be in [0,1]")
        self.memory = memory
        self.db = memory.db
        self.min_confidence_gain = float(min_confidence_gain)
        self.authority_override = float(authority_override)
        self.authority_override_min_confidence = float(authority_override_min_confidence)
        self._create_schema()

    def _create_schema(self) -> None:
        self.db.execute("""CREATE TABLE IF NOT EXISTS guarded_fact_history (
            history_id INTEGER PRIMARY KEY,
            fact_key TEXT NOT NULL,
            revision INTEGER NOT NULL,
            value TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence REAL NOT NULL,
            authority REAL NOT NULL,
            status TEXT NOT NULL,
            explicit_correction INTEGER NOT NULL DEFAULT 0,
            created_ns INTEGER NOT NULL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_guarded_history_key ON guarded_fact_history(fact_key, history_id)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS guarded_fact_current (
            fact_key TEXT PRIMARY KEY,
            revision INTEGER NOT NULL,
            value TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence REAL NOT NULL,
            authority REAL NOT NULL,
            history_id INTEGER NOT NULL,
            updated_ns INTEGER NOT NULL)""")
        self.db.commit()

    @staticmethod
    def normalize_key(key: str) -> str:
        result = " ".join(str(key).casefold().split())
        if not result:
            raise ValueError("fact key must be non-empty")
        return result

    @staticmethod
    def normalize_value(value: str) -> str:
        return " ".join(str(value).casefold().split())

    @staticmethod
    def _validate_score(name: str, value: float) -> float:
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
        return score

    @staticmethod
    def _snapshot(row) -> FactSnapshot:
        return FactSnapshot(str(row[0]), int(row[1]), str(row[2]), str(row[3]),
                            float(row[4]), float(row[5]), int(row[6]), int(row[7]))

    def get(self, key: str) -> FactSnapshot | None:
        fact_key = self.normalize_key(key)
        row = self.db.execute(
            """SELECT fact_key,revision,value,source,confidence,authority,history_id,updated_ns
               FROM guarded_fact_current WHERE fact_key=?""", (fact_key,)
        ).fetchone()
        return None if row is None else self._snapshot(row)

    def answer(self, key: str) -> FactAnswer:
        """Return an answer only when no unresolved contradictory proposal exists.

        The current accepted version is exposed as provenance, but ``value`` is
        deliberately None while conflicts are pending so a caller can abstain.
        """
        fact_key = self.normalize_key(key)
        current = self.get(fact_key)
        if current is None:
            return FactAnswer(fact_key, "unknown", None, None, ())
        conflicts = self.pending_conflicts(fact_key)
        if conflicts:
            alternatives = tuple(dict.fromkeys(str(row["value"]) for row in conflicts))
            return FactAnswer(fact_key, "conflict", None, current, alternatives)
        return FactAnswer(fact_key, "found", current.value, current, ())

    def propose(self, key: str, value: str, source: str = "", confidence: float = 0.5,
                authority: float = 0.5, explicit_correction: bool = False) -> FactProposal:
        fact_key = self.normalize_key(key)
        proposed_value = str(value).strip()
        if not proposed_value:
            raise ValueError("fact value must be non-empty")
        conf = self._validate_score("confidence", confidence)
        auth = self._validate_score("authority", authority)
        src = str(source)
        now = time.time_ns()
        current = self.get(fact_key)
        if current is None:
            revision = 1
            status = "accepted"
        elif self.normalize_value(proposed_value) == self.normalize_value(current.value):
            revision = current.revision
            status = "reaffirmed"
        else:
            confidence_override = (
                auth >= current.authority
                and conf >= current.confidence + self.min_confidence_gain
            )
            authority_override = (
                auth >= current.authority + self.authority_override
                and conf >= self.authority_override_min_confidence
            )
            accepted = bool(explicit_correction or confidence_override or authority_override)
            revision = current.revision + 1
            status = "accepted" if accepted else "pending_conflict"
        cur = self.db.execute(
            """INSERT INTO guarded_fact_history
               (fact_key,revision,value,source,confidence,authority,status,explicit_correction,created_ns)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (fact_key, revision, proposed_value, src, conf, auth, status,
             int(bool(explicit_correction)), now),
        )
        history_id = int(cur.lastrowid)
        if current is None:
            self.db.execute(
                """INSERT INTO guarded_fact_current
                   (fact_key,revision,value,source,confidence,authority,history_id,updated_ns)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (fact_key, 1, proposed_value, src, conf, auth, history_id, now),
            )
        elif status == "reaffirmed":
            # Retain the stronger evidence summary while keeping the active value.
            new_conf = max(current.confidence, conf)
            new_auth = max(current.authority, auth)
            chosen_source = src if conf >= current.confidence else current.source
            self.db.execute(
                """UPDATE guarded_fact_current SET confidence=?,authority=?,source=?,updated_ns=?
                   WHERE fact_key=?""",
                (new_conf, new_auth, chosen_source, now, fact_key),
            )
        elif status == "accepted":
            self.db.execute(
                "UPDATE guarded_fact_history SET status='superseded' WHERE history_id=?",
                (current.history_id,),
            )
            self.db.execute(
                """UPDATE guarded_fact_current SET revision=?,value=?,source=?,confidence=?,authority=?,
                   history_id=?,updated_ns=? WHERE fact_key=?""",
                (revision, proposed_value, src, conf, auth, history_id, now, fact_key),
            )
        self.db.commit()
        new_current = self.get(fact_key)
        assert new_current is not None
        return FactProposal(fact_key, proposed_value, status,
                            status in ("accepted", "reaffirmed"), new_current,
                            conf, auth, history_id)

    def resolve(self, key: str, chosen_value: str, reviewer: str = "manual-review") -> FactProposal:
        """Explicitly resolve pending conflicts; the reviewer choice is authoritative."""
        result = self.propose(key, chosen_value, source=reviewer, confidence=1.0,
                              authority=1.0, explicit_correction=True)
        fact_key = self.normalize_key(key)
        # Once a human/reviewer resolves the key, competing pending proposals are closed.
        self.db.execute(
            """UPDATE guarded_fact_history SET status='reaffirmed'
               WHERE fact_key=? AND status='pending_conflict' AND lower(trim(value))=lower(trim(?))""",
            (fact_key, str(chosen_value).strip()),
        )
        self.db.execute(
            """UPDATE guarded_fact_history SET status='rejected'
               WHERE fact_key=? AND status='pending_conflict' AND lower(trim(value))<>lower(trim(?))""",
            (fact_key, str(chosen_value).strip()),
        )
        self.db.commit()
        return result

    def history(self, key: str, include_pending: bool = True) -> list[dict]:
        fact_key = self.normalize_key(key)
        sql = """SELECT history_id,revision,value,source,confidence,authority,status,
                        explicit_correction,created_ns
                 FROM guarded_fact_history WHERE fact_key=?"""
        params: list = [fact_key]
        if not include_pending:
            sql += " AND status<>'pending_conflict'"
        sql += " ORDER BY history_id"
        rows = self.db.execute(sql, params).fetchall()
        keys = ("history_id", "revision", "value", "source", "confidence", "authority",
                "status", "explicit_correction", "created_ns")
        return [dict(zip(keys, row)) for row in rows]

    def pending_conflicts(self, key: str | None = None) -> list[dict]:
        sql = """SELECT history_id,fact_key,revision,value,source,confidence,authority,created_ns
                 FROM guarded_fact_history WHERE status='pending_conflict'"""
        params: tuple = ()
        if key is not None:
            sql += " AND fact_key=?"
            params = (self.normalize_key(key),)
        sql += " ORDER BY history_id"
        rows = self.db.execute(sql, params).fetchall()
        keys = ("history_id", "fact_key", "revision", "value", "source", "confidence",
                "authority", "created_ns")
        return [dict(zip(keys, row)) for row in rows]
