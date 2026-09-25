"""Transactional state primitives for recursive ternary growth."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class RunState(StrEnum):
    RESERVED = "reserved"
    RUNNING = "running"
    PREPARED = "prepared"
    TERMINAL_ACCEPTED = "terminal_accepted"
    TERMINAL_REJECTED = "terminal_rejected"
    FAILED = "failed"


_TERMINAL = {RunState.TERMINAL_ACCEPTED, RunState.TERMINAL_REJECTED, RunState.FAILED}
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_-]{1,128}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_bytes(data: bytes) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    value = json.loads(data, object_pairs_hook=no_duplicates)
    if canonical_json_bytes(value) != data:
        raise ValueError("non-canonical JSON bytes")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_key_digest(state: dict[str, Any]) -> str:
    import torch

    entries = []
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state value must be a tensor: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"state tensor is non-finite: {key}")
        array = value.detach().cpu().contiguous()
        raw = array.view(torch.uint8).reshape(-1).numpy().tobytes()
        entries.append(
            {
                "bytes_sha256": hashlib.sha256(raw).hexdigest(),
                "dtype": str(value.dtype),
                "key": key,
                "shape": list(value.shape),
            }
        )
    return sha256_bytes(canonical_json_bytes(entries))


def _valid_owner(owner_id: str) -> str:
    if not isinstance(owner_id, str) or _SAFE_COMPONENT.fullmatch(owner_id) is None:
        raise ValueError("owner_id must be a safe path component")
    return owner_id


_DEFAULT_LEASE_SECONDS = 300
_MAX_LEASE_SECONDS = 86_400


def _lease_payload(generation_id: str, owner_id: str, ttl_seconds: int) -> dict[str, Any]:
    if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= _MAX_LEASE_SECONDS:
        raise ValueError("owner lease must be between 1 and 86400 seconds")
    return {
        "generation_id": _valid_owner(generation_id),
        "owner_id": _valid_owner(owner_id),
        "expires_at": time.time() + ttl_seconds,
    }


def _read_lease(run: Path) -> dict[str, Any]:
    path = run / ".owner.lease"
    _reject_link_chain(path, allow_missing=False)
    if _is_link(path) or not path.is_file():
        raise ValueError("owner lease must be regular non-symlink")
    try:
        value = _strict_json_bytes(_regular_bytes(path))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("malformed owner lease") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"generation_id", "owner_id", "expires_at"}
        or not isinstance(value.get("generation_id"), str)
        or not isinstance(value.get("owner_id"), str)
        or isinstance(value.get("expires_at"), bool)
        or not isinstance(value.get("expires_at"), (int, float))
        or not math.isfinite(float(value.get("expires_at")))
        or _SAFE_COMPONENT.fullmatch(value["generation_id"]) is None
        or _SAFE_COMPONENT.fullmatch(value["owner_id"]) is None
    ):
        raise ValueError("invalid owner lease contract")
    return value


def _is_link(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    # Windows junctions/reparse points are not always reported as symlinks.
    attrs = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attrs & reparse)


def _reject_link_chain(path: Path, *, allow_missing: bool = True) -> None:
    """Reject links in every existing component without resolving through them."""
    current = Path(os.path.abspath(path))
    missing = False
    while True:
        if _is_link(current):
            raise ValueError(f"symlink or junction is forbidden: {current}")
        if not current.exists():
            if not allow_missing:
                raise ValueError(f"path does not exist: {path}")
            missing = True
        if current.parent == current:
            break
        current = current.parent
        if missing and not allow_missing:
            break


def _regular_bytes(path: Path, expected: str | None = None) -> bytes:
    _reject_link_chain(path, allow_missing=False)
    if _is_link(path) or not path.is_file():
        raise ValueError(f"bound evidence must be a regular non-symlink file: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read bound evidence: {path}") from exc
    if expected is not None and sha256_bytes(data) != expected:
        raise ValueError("bound evidence digest mismatch")
    return data


@dataclass(frozen=True)
class PreparedDecision:
    generation_id: str
    manifest_sha256: str
    candidate_checkpoint_sha256: str
    decision: str
    holdout_evidence_sha256: str
    manifest_path: str | None = None
    candidate_checkpoint_path: str | None = None
    holdout_evidence_path: str | None = None
    parent_generation_id: str | None = None
    parent_checkpoint_sha256: str | None = None
    parent_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        _valid_owner(self.generation_id)
        if self.decision not in {"accepted", "rejected"}:
            raise ValueError("decision must be accepted or rejected")
        for value in (
            self.manifest_sha256,
            self.candidate_checkpoint_sha256,
            self.holdout_evidence_sha256,
        ):
            if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
                raise ValueError("prepared decision digest must be 64 lowercase hex characters")
        parent_values = (
            self.parent_generation_id,
            self.parent_checkpoint_sha256,
            self.parent_manifest_sha256,
        )
        if any(value is not None for value in parent_values):
            if any(value is None for value in parent_values):
                raise ValueError("prepared parent binding must be complete")
            _valid_owner(self.parent_generation_id)
            _valid_digest(self.parent_checkpoint_sha256, "parent_checkpoint_sha256")
            _valid_digest(self.parent_manifest_sha256, "parent_manifest_sha256")

    def validate_files(self) -> dict[str, bytes]:
        if (
            self.manifest_path is None
            or self.candidate_checkpoint_path is None
            or self.holdout_evidence_path is None
        ):
            raise ValueError("prepared decision requires bound evidence paths")
        return {
            "manifest": _regular_bytes(
                Path(self.manifest_path), self.manifest_sha256
            ),
            "candidate": _regular_bytes(
                Path(self.candidate_checkpoint_path), self.candidate_checkpoint_sha256
            ),
            "holdout": _regular_bytes(
                Path(self.holdout_evidence_path), self.holdout_evidence_sha256
            ),
        }

    def with_snapshot_paths(self, paths: dict[str, str]) -> PreparedDecision:
        if set(paths) != {"manifest", "candidate", "holdout"}:
            raise ValueError("snapshot paths must bind manifest, candidate, and holdout")
        return PreparedDecision(
            generation_id=self.generation_id,
            manifest_sha256=self.manifest_sha256,
            candidate_checkpoint_sha256=self.candidate_checkpoint_sha256,
            decision=self.decision,
            holdout_evidence_sha256=self.holdout_evidence_sha256,
            manifest_path=paths["manifest"],
            candidate_checkpoint_path=paths["candidate"],
            holdout_evidence_path=paths["holdout"],
            parent_generation_id=self.parent_generation_id,
            parent_checkpoint_sha256=self.parent_checkpoint_sha256,
            parent_manifest_sha256=self.parent_manifest_sha256,
        )


def _valid_digest(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hex characters")
    return value


def _decision_equal(left: PreparedDecision, right: PreparedDecision) -> bool:
    """Compare the complete source binding, including evidence paths."""
    return (
        left.generation_id == right.generation_id
        and left.manifest_sha256 == right.manifest_sha256
        and left.candidate_checkpoint_sha256 == right.candidate_checkpoint_sha256
        and left.decision == right.decision
        and left.holdout_evidence_sha256 == right.holdout_evidence_sha256
        and str(left.manifest_path) == str(right.manifest_path)
        and str(left.candidate_checkpoint_path) == str(right.candidate_checkpoint_path)
        and str(left.holdout_evidence_path) == str(right.holdout_evidence_path)
        and left.parent_generation_id == right.parent_generation_id
        and left.parent_checkpoint_sha256 == right.parent_checkpoint_sha256
        and left.parent_manifest_sha256 == right.parent_manifest_sha256
    )


def _prepared_payload(
    source: PreparedDecision, snapshot: PreparedDecision
) -> dict[str, Any]:
    parent_bound = source.parent_generation_id is not None
    payload = {
        "candidate_checkpoint_path": snapshot.candidate_checkpoint_path,
        "candidate_checkpoint_sha256": snapshot.candidate_checkpoint_sha256,
        "decision": snapshot.decision,
        "generation_id": snapshot.generation_id,
        "holdout_evidence_path": snapshot.holdout_evidence_path,
        "holdout_evidence_sha256": snapshot.holdout_evidence_sha256,
        "manifest_path": snapshot.manifest_path,
        "manifest_sha256": snapshot.manifest_sha256,
        "schema_version": 2 if parent_bound else 1,
        "source_candidate_checkpoint_path": str(source.candidate_checkpoint_path),
        "source_holdout_evidence_path": str(source.holdout_evidence_path),
        "source_manifest_path": str(source.manifest_path),
        "state": RunState.PREPARED.value,
    }
    if parent_bound:
        payload.update(
            {
                "parent_generation_id": source.parent_generation_id,
                "parent_checkpoint_sha256": source.parent_checkpoint_sha256,
                "parent_manifest_sha256": source.parent_manifest_sha256,
            }
        )
    return payload


def _prepared_contract(
    payload: Any, generation_id: str
) -> tuple[PreparedDecision, PreparedDecision]:
    required = {
        "schema_version", "state", "generation_id", "decision",
        "manifest_sha256", "candidate_checkpoint_sha256", "holdout_evidence_sha256",
        "manifest_path", "candidate_checkpoint_path", "holdout_evidence_path",
        "source_manifest_path", "source_candidate_checkpoint_path",
        "source_holdout_evidence_path",
    }
    parent_fields = {
        "parent_generation_id", "parent_checkpoint_sha256", "parent_manifest_sha256"
    }
    if not isinstance(payload, dict):
        raise ValueError("invalid prepared report contract")
    keys = set(payload)
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError("invalid prepared report state")
    if schema_version == 1 and keys != required:
        raise ValueError("invalid prepared report contract")
    if schema_version == 2 and keys != required | parent_fields:
        raise ValueError("invalid prepared report contract")
    if payload["state"] != RunState.PREPARED.value:
        raise ValueError("invalid prepared report state")
    if payload["generation_id"] != generation_id:
        raise ValueError("prepared report generation mismatch")
    if schema_version == 1 and payload.get("decision") == "accepted":
        raise ValueError("accepted schema v1 unsupported")
    for key in keys - {"schema_version", "state", "generation_id", "decision"}:
        if not isinstance(payload[key], str) or not payload[key]:
            raise ValueError("invalid prepared report field")
    parent_binding = {key: payload.get(key) for key in parent_fields}
    snapshot = PreparedDecision(
        generation_id=payload["generation_id"],
        manifest_sha256=payload["manifest_sha256"],
        candidate_checkpoint_sha256=payload["candidate_checkpoint_sha256"],
        decision=payload["decision"],
        holdout_evidence_sha256=payload["holdout_evidence_sha256"],
        manifest_path=payload["manifest_path"],
        candidate_checkpoint_path=payload["candidate_checkpoint_path"],
        holdout_evidence_path=payload["holdout_evidence_path"],
        **parent_binding,
    )
    source = PreparedDecision(
        generation_id=payload["generation_id"],
        manifest_sha256=payload["manifest_sha256"],
        candidate_checkpoint_sha256=payload["candidate_checkpoint_sha256"],
        decision=payload["decision"],
        holdout_evidence_sha256=payload["holdout_evidence_sha256"],
        manifest_path=payload["source_manifest_path"],
        candidate_checkpoint_path=payload["source_candidate_checkpoint_path"],
        holdout_evidence_path=payload["source_holdout_evidence_path"],
        **parent_binding,
    )
    return snapshot, source


def _read_prepared_report(
    run: Path, generation_id: str
) -> tuple[Path, dict[str, Any], PreparedDecision, PreparedDecision, str]:
    path = run / "prepared-report.json"
    data = _regular_bytes(path)
    try:
        payload = _strict_json_bytes(data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("malformed prepared report") from exc
    snapshot, source = _prepared_contract(payload, generation_id)
    expected_paths = {
        role: str(run / "evidence" / f"{role}-{snapshot_digest}.bin")
        for role, snapshot_digest in {
            "manifest": snapshot.manifest_sha256,
            "candidate": snapshot.candidate_checkpoint_sha256,
            "holdout": snapshot.holdout_evidence_sha256,
        }.items()
    }
    if (
        snapshot.manifest_path != expected_paths["manifest"]
        or snapshot.candidate_checkpoint_path != expected_paths["candidate"]
        or snapshot.holdout_evidence_path != expected_paths["holdout"]
    ):
        raise ValueError("prepared snapshot path mismatch")
    snapshot.validate_files()
    return path, payload, snapshot, source, sha256_bytes(data)


def _terminal_payload(
    prepared: dict[str, Any], prepared_sha256: str, snapshot: PreparedDecision
) -> dict[str, Any]:
    parent_bound = "parent_generation_id" in prepared
    payload = {
        "candidate_checkpoint_path": snapshot.candidate_checkpoint_path,
        "candidate_checkpoint_sha256": snapshot.candidate_checkpoint_sha256,
        "decision": snapshot.decision,
        "generation_id": snapshot.generation_id,
        "holdout_evidence_path": snapshot.holdout_evidence_path,
        "holdout_evidence_sha256": snapshot.holdout_evidence_sha256,
        "manifest_path": snapshot.manifest_path,
        "manifest_sha256": snapshot.manifest_sha256,
        "prepared_report_sha256": prepared_sha256,
        "schema_version": 2 if parent_bound else 1,
        "source_candidate_checkpoint_path": prepared["source_candidate_checkpoint_path"],
        "source_holdout_evidence_path": prepared["source_holdout_evidence_path"],
        "source_manifest_path": prepared["source_manifest_path"],
        "state": "terminal",
    }
    if parent_bound:
        payload.update(
            {
                "parent_generation_id": prepared["parent_generation_id"],
                "parent_checkpoint_sha256": prepared["parent_checkpoint_sha256"],
                "parent_manifest_sha256": prepared["parent_manifest_sha256"],
            }
        )
    return payload


def _terminal_contract(
    payload: Any, generation_id: str
) -> tuple[PreparedDecision, PreparedDecision]:
    required = {
        "schema_version", "state", "generation_id", "decision",
        "prepared_report_sha256", "manifest_sha256", "candidate_checkpoint_sha256",
        "holdout_evidence_sha256", "manifest_path", "candidate_checkpoint_path",
        "holdout_evidence_path", "source_manifest_path",
        "source_candidate_checkpoint_path", "source_holdout_evidence_path",
    }
    parent_fields = {
        "parent_generation_id", "parent_checkpoint_sha256", "parent_manifest_sha256"
    }
    if not isinstance(payload, dict):
        raise ValueError("invalid terminal report contract")
    keys = set(payload)
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError("invalid terminal report state")
    if schema_version == 1 and keys != required:
        raise ValueError("invalid terminal report contract")
    if schema_version == 2 and keys != required | parent_fields:
        raise ValueError("invalid terminal report contract")
    if payload["state"] != "terminal":
        raise ValueError("invalid terminal report state")
    if payload["generation_id"] != generation_id:
        raise ValueError("terminal report generation mismatch")
    if schema_version == 1 and payload.get("decision") == "accepted":
        raise ValueError("accepted schema v1 unsupported")
    _valid_digest(payload["prepared_report_sha256"], "prepared_report_sha256")
    parent_binding = {key: payload.get(key) for key in parent_fields}
    snapshot = PreparedDecision(
        generation_id=payload["generation_id"],
        manifest_sha256=payload["manifest_sha256"],
        candidate_checkpoint_sha256=payload["candidate_checkpoint_sha256"],
        decision=payload["decision"],
        holdout_evidence_sha256=payload["holdout_evidence_sha256"],
        manifest_path=payload["manifest_path"],
        candidate_checkpoint_path=payload["candidate_checkpoint_path"],
        holdout_evidence_path=payload["holdout_evidence_path"],
        **parent_binding,
    )
    source = PreparedDecision(
        generation_id=payload["generation_id"],
        manifest_sha256=payload["manifest_sha256"],
        candidate_checkpoint_sha256=payload["candidate_checkpoint_sha256"],
        decision=payload["decision"],
        holdout_evidence_sha256=payload["holdout_evidence_sha256"],
        manifest_path=payload["source_manifest_path"],
        candidate_checkpoint_path=payload["source_candidate_checkpoint_path"],
        holdout_evidence_path=payload["source_holdout_evidence_path"],
        **parent_binding,
    )
    return snapshot, source


@dataclass(frozen=True)
class TrustedParentToken:
    owner_id: str
    pointer_sha256: str

    def __post_init__(self) -> None:
        _valid_owner(self.owner_id)
        if not isinstance(self.pointer_sha256, str) or _HEX64.fullmatch(
            self.pointer_sha256
        ) is None:
            raise ValueError("trusted parent token digest must be 64 lowercase hex")


@dataclass(frozen=True)
class ParentPointer:
    generation_id: str
    checkpoint_sha256: str
    manifest_sha256: str
    report_sha256: str
    owner_id: str = "default-owner"

    def __post_init__(self) -> None:
        _valid_owner(self.generation_id)
        _valid_owner(self.owner_id)
        for value in (self.checkpoint_sha256, self.manifest_sha256, self.report_sha256):
            if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
                raise ValueError("parent pointer digest must be 64 lowercase hex characters")

    def as_dict(self) -> dict[str, str]:
        return {
            "generation_id": self.generation_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "manifest_sha256": self.manifest_sha256,
            "report_sha256": self.report_sha256,
            "owner_id": self.owner_id,
        }

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.as_dict()))


def _parent_identity(pointer: ParentPointer) -> tuple[str, str, str, str]:
    """Return the immutable committed-generation identity, excluding commit owner."""
    return (
        pointer.generation_id,
        pointer.checkpoint_sha256,
        pointer.manifest_sha256,
        pointer.report_sha256,
    )


def _fsync_directory(path: Path) -> None:
    """Persist directory entries on POSIX; Windows has no portable directory fsync."""
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, payload: bytes) -> None:
    _reject_link_chain(path.parent, allow_missing=False)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _reject_link_chain(path, allow_missing=True)
        os.replace(name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def _acquire_file_lock(path: Path, error_message: str):
    """Acquire a regular-file lock that the OS releases when its holder dies."""
    _reject_link_chain(path.parent, allow_missing=False)
    _reject_link_chain(path, allow_missing=True)
    if path.exists() and (_is_link(path) or not path.is_file()):
        raise ValueError(error_message)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise ValueError(error_message) from exc
    try:
        handle = os.fdopen(fd, "r+b", buffering=0)
    except BaseException:
        os.close(fd)
        raise

    acquired = False
    try:
        _reject_link_chain(path, allow_missing=False)
        if _is_link(path):
            raise ValueError(error_message)
        descriptor_stat = os.fstat(handle.fileno())
        path_stat = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or not os.path.samestat(descriptor_stat, path_stat)
        ):
            raise ValueError(error_message)
        if descriptor_stat.st_size == 0:
            handle.seek(0)
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError(error_message) from exc
        acquired = True
        descriptor_stat = os.fstat(handle.fileno())
        path_stat = os.stat(path, follow_symlinks=False)
        if (
            _is_link(path)
            or not stat.S_ISREG(descriptor_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or not os.path.samestat(descriptor_stat, path_stat)
        ):
            raise ValueError("lock file identity changed")
        yield
    finally:
        if acquired:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


class GrowthRunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        _reject_link_chain(self.root, allow_missing=True)

    def _run(self, generation_id: str) -> Path:
        _valid_owner(generation_id)
        _reject_link_chain(self.root, allow_missing=True)
        path = self.root / generation_id
        root = Path(os.path.abspath(self.root))
        candidate = Path(os.path.abspath(path))
        if candidate.parent != root:
            raise ValueError("generation path escapes store root")
        _reject_link_chain(candidate, allow_missing=True)
        return candidate

    @contextmanager
    def _lock(self, generation_id: str, owner_id: str):
        run = self._run(generation_id)
        with _acquire_file_lock(run / ".lifecycle.guard", "run lifecycle is locked"):
            state = self._read_state(generation_id)
            self._check_owner(state, owner_id)
            lock = run / ".lifecycle.lock"
            _reject_link_chain(lock, allow_missing=True)
            if lock.exists():
                raise ValueError("run lifecycle is locked")
            lock.mkdir()
            try:
                _atomic_write(
                    lock / "owner.json",
                    canonical_json_bytes({"generation_id": generation_id, "owner_id": owner_id}),
                )
            except BaseException:
                if lock.is_dir() and not _is_link(lock):
                    lock.rmdir()
                raise
            try:
                yield lock
            finally:
                owner = lock / "owner.json"
                if owner.is_file() and not _is_link(owner):
                    owner.unlink()
                if lock.is_dir() and not _is_link(lock):
                    lock.rmdir()

    def _read_takeover_pending(self, run: Path) -> dict[str, Any] | None:
        path = run / "audit" / "lease-takeover.pending.json"
        if not path.exists():
            return None
        data = _regular_bytes(path)
        try:
            payload = _strict_json_bytes(data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("malformed lease takeover intent") from exc
        required = {
            "generation_id", "new_owner_id", "previous_expires_at", "previous_owner_id",
            "reason", "schema_version", "lease_seconds",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("invalid lease takeover intent contract")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("unsupported lease takeover intent version")
        _valid_owner(payload["generation_id"])
        _valid_owner(payload["new_owner_id"])
        _valid_owner(payload["previous_owner_id"])
        if not isinstance(payload["reason"], str) or not payload["reason"].strip():
            raise ValueError("lease takeover reason is required")
        if (
            isinstance(payload["previous_expires_at"], bool)
            or not isinstance(payload["previous_expires_at"], (int, float))
            or not math.isfinite(float(payload["previous_expires_at"]))
        ):
            raise ValueError("invalid lease takeover expiry")
        if (
            type(payload["lease_seconds"]) is not int
            or not 1 <= payload["lease_seconds"] <= _MAX_LEASE_SECONDS
        ):
            raise ValueError("invalid lease takeover duration")
        return payload

    def _clear_lifecycle_lock(self, lock: Path) -> None:
        if not lock.exists():
            return
        if _is_link(lock) or not lock.is_dir():
            raise ValueError("lifecycle lock is unsafe")
        entries = list(lock.iterdir())
        for entry in entries:
            if entry.name != "owner.json" and not (
                entry.name.startswith(".owner.json.") and entry.name != ".owner.json."
            ):
                raise ValueError("lifecycle lock contains unexpected entry")
            _reject_link_chain(entry, allow_missing=False)
            if _is_link(entry) or not entry.is_file():
                raise ValueError("lifecycle lock metadata is unsafe")
        for entry in entries:
            entry.unlink()
        lock.rmdir()

    def _publish_takeover_audit(
        self,
        audit_dir: Path,
        pending: dict[str, Any],
        *,
        status: str = "complete",
        superseded_by: str | None = None,
    ) -> Path:
        if status not in {"complete", "superseded"}:
            raise ValueError("invalid lease takeover audit status")
        takeover = {
            "generation_id": pending["generation_id"],
            "lease_seconds": pending["lease_seconds"],
            "new_owner_id": pending["new_owner_id"],
            "previous_expires_at": pending["previous_expires_at"],
            "previous_owner_id": pending["previous_owner_id"],
            "reason": pending["reason"],
            "schema_version": 1,
            "status": status,
        }
        if superseded_by is not None:
            takeover["superseded_by"] = _valid_owner(superseded_by)
        if (status == "superseded") != (superseded_by is not None):
            raise ValueError("superseded takeover audit requires superseded_by")
        payload = canonical_json_bytes(takeover)
        path = audit_dir / f"lease-takeover-{sha256_bytes(payload)}.json"
        self._publish_regular_no_replace(
            path,
            payload,
            "lease takeover audit must be regular non-symlink",
        )
        return path

    def _remove_takeover_pending(self, audit_dir: Path) -> None:
        path = audit_dir / "lease-takeover.pending.json"
        if not path.exists():
            return
        _reject_link_chain(path, allow_missing=False)
        if _is_link(path) or not path.is_file():
            raise ValueError("lease takeover intent is unsafe")
        path.unlink()

    @contextmanager
    def _recovery_locks(self, generation_id: str):
        run = self._run(generation_id)
        with _acquire_file_lock(
            run / ".lease-recovery.lock", "lease recovery is already locked"
        ):
            with _acquire_file_lock(run / ".lifecycle.guard", "run lifecycle is locked"):
                yield run

    def recover_stale_lock(
        self,
        generation_id: str,
        new_owner_id: str,
        reason: str,
        *,
        lease_seconds: int = _DEFAULT_LEASE_SECONDS,
    ) -> None:
        """Take over an expired generation lease only after the holder has released its lock."""
        _valid_owner(new_owner_id)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("stale-lock recovery reason is required")
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= _MAX_LEASE_SECONDS:
            raise ValueError("owner lease must be between 1 and 86400 seconds")
        with self._recovery_locks(generation_id) as run:
            state = self._read_state(generation_id)
            lease = _read_lease(run)
            if lease["generation_id"] != generation_id:
                raise ValueError("run owner mismatch")
            lifecycle = run / ".lifecycle.lock"
            _reject_link_chain(lifecycle, allow_missing=True)
            if lifecycle.exists() and (_is_link(lifecycle) or not lifecycle.is_dir()):
                raise ValueError("lifecycle lock is unsafe")
            metadata_owner: str | None = None
            metadata_path = lifecycle / "owner.json"
            if metadata_path.exists():
                metadata_bytes = _regular_bytes(metadata_path)
                try:
                    metadata = _strict_json_bytes(metadata_bytes)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    raise ValueError("malformed lifecycle lock metadata") from exc
                if (
                    not isinstance(metadata, dict)
                    or set(metadata) != {"generation_id", "owner_id"}
                    or metadata.get("generation_id") != generation_id
                ):
                    raise ValueError("stale lock generation mismatch")
                metadata_owner = metadata.get("owner_id")
                _valid_owner(metadata_owner)
            audit_dir = run / "audit"
            audit_dir.mkdir(exist_ok=True)
            _reject_link_chain(audit_dir, allow_missing=False)
            pending_path = audit_dir / "lease-takeover.pending.json"
            state_owner_before = state["owner_id"]
            lease_owner_before = lease["owner_id"]
            pending = self._read_takeover_pending(run)
            if pending is not None:
                if pending["generation_id"] != generation_id:
                    raise ValueError("conflicting lease takeover intent")
                pending_previous_owner = pending["previous_owner_id"]
                pending_new_owner = pending["new_owner_id"]
                if (
                    state_owner_before not in {pending_previous_owner, pending_new_owner}
                    or lease_owner_before not in {pending_previous_owner, pending_new_owner}
                ):
                    raise ValueError("conflicting lease takeover intent")
                if metadata_owner is not None and metadata_owner not in {
                    pending_previous_owner,
                    pending_new_owner,
                    state["owner_id"],
                    lease["owner_id"],
                }:
                    raise ValueError("conflicting lease takeover intent")
                if metadata_owner not in {None, state["owner_id"], lease["owner_id"]}:
                    self._clear_lifecycle_lock(lifecycle)
                matches_pending = (
                    pending["generation_id"] == generation_id
                    and pending_new_owner == new_owner_id
                    and pending["lease_seconds"] == lease_seconds
                    and pending["reason"] == reason
                )
                completed_pending = (
                    state["owner_id"] == pending_new_owner
                    and lease["generation_id"] == generation_id
                    and lease["owner_id"] == pending_new_owner
                )
                if completed_pending:
                    # A previous takeover reached state+lease but crashed before
                    # writing its audit or deleting the intent. A live completed
                    # lease may only be completed by the exact intended retry.
                    if lease["expires_at"] > time.time():
                        if not matches_pending:
                            if new_owner_id == pending_new_owner:
                                raise ValueError("conflicting lease takeover intent")
                            raise ValueError("owner lease is still active")
                        self._publish_takeover_audit(audit_dir, pending)
                        self._clear_lifecycle_lock(lifecycle)
                        self._remove_takeover_pending(audit_dir)
                        return
                    self._publish_takeover_audit(audit_dir, pending)
                    # The completed owner later crashed. Rotate the durable
                    # intent instead of deleting it, preserving an auditable
                    # chain across arbitrarily many compound crashes.
                    pending = {
                        "generation_id": generation_id,
                        "lease_seconds": lease_seconds,
                        "new_owner_id": new_owner_id,
                        "previous_expires_at": lease["expires_at"],
                        "previous_owner_id": lease["owner_id"],
                        "reason": reason,
                        "schema_version": 1,
                    }
                    _atomic_write(pending_path, canonical_json_bytes(pending))
                elif not matches_pending:
                    unclaimed = (
                        state_owner_before == pending_previous_owner
                        and lease_owner_before == pending_previous_owner
                    )
                    state_claimed = (
                        state_owner_before == pending_new_owner
                        and lease_owner_before == pending_previous_owner
                    )
                    if (
                        lease["expires_at"] > time.time()
                        or not (unclaimed or state_claimed)
                    ):
                        raise ValueError("conflicting lease takeover intent")
                    self._publish_takeover_audit(
                        audit_dir,
                        pending,
                        status="superseded",
                        superseded_by=new_owner_id,
                    )
                    pending = {
                        "generation_id": generation_id,
                        "lease_seconds": lease_seconds,
                        "new_owner_id": new_owner_id,
                        "previous_expires_at": lease["expires_at"],
                        "previous_owner_id": lease["owner_id"],
                        "reason": reason,
                        "schema_version": 1,
                    }
                    _atomic_write(pending_path, canonical_json_bytes(pending))
                previous_owner = pending["previous_owner_id"]
            else:
                previous_owner = metadata_owner or state["owner_id"]
                _valid_owner(previous_owner)
                if lease["expires_at"] > time.time():
                    raise ValueError("owner lease is still active")
                if state["owner_id"] != previous_owner or lease["owner_id"] != previous_owner:
                    raise ValueError("run owner mismatch")
                pending = {
                    "generation_id": generation_id,
                    "lease_seconds": lease_seconds,
                    "new_owner_id": new_owner_id,
                    "previous_expires_at": lease["expires_at"],
                    "previous_owner_id": previous_owner,
                    "reason": reason,
                    "schema_version": 1,
                }
                _atomic_write(pending_path, canonical_json_bytes(pending))
            if (
                lease["owner_id"] == previous_owner
                and lease["expires_at"] > time.time()
            ):
                raise ValueError("owner lease is still active")
            allowed_owners = {
                state_owner_before,
                lease_owner_before,
                new_owner_id,
            }
            if state["owner_id"] not in allowed_owners:
                raise ValueError("run owner mismatch")
            if lease["owner_id"] not in allowed_owners:
                raise ValueError("run owner mismatch")
            current = self._read_state(generation_id)
            current["owner_id"] = new_owner_id
            _atomic_write(run / "state.json", canonical_json_bytes(current))
            _atomic_write(
                run / ".owner.lease",
                canonical_json_bytes(_lease_payload(generation_id, new_owner_id, lease_seconds)),
            )
            self._publish_takeover_audit(audit_dir, pending)
            self._clear_lifecycle_lock(lifecycle)
            self._remove_takeover_pending(audit_dir)

    def _read_state(self, generation_id: str) -> dict[str, Any]:
        path = self._run(generation_id) / "state.json"
        _reject_link_chain(path, allow_missing=False)
        if _is_link(path) or not path.is_file():
            raise ValueError("state.json must be regular non-symlink")
        try:
            value = _strict_json_bytes(_regular_bytes(path))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("malformed state.json") from exc
        required = {
            "generation_id", "state", "owner_id",
            "prepared_report_sha256", "terminal_report_sha256",
        }
        if (
            not isinstance(value, dict)
            or set(value) != required
            or value.get("generation_id") != generation_id
            or not isinstance(value.get("state"), str)
            or RunState(value["state"]) not in RunState
            or not isinstance(value.get("owner_id"), str)
        ):
            raise ValueError("invalid state.json contract")
        _valid_owner(value["owner_id"])
        _valid_digest(value["prepared_report_sha256"], "prepared_report_sha256")
        _valid_digest(value["terminal_report_sha256"], "terminal_report_sha256")
        state = RunState(value["state"])
        if state in {RunState.RESERVED, RunState.RUNNING}:
            if value["prepared_report_sha256"] is not None or value["terminal_report_sha256"] is not None:
                raise ValueError("nonterminal state has terminal binding")
        elif state is RunState.PREPARED:
            if value["prepared_report_sha256"] is None or value["terminal_report_sha256"] is not None:
                raise ValueError("prepared state binding is incomplete")
        elif state in {RunState.TERMINAL_ACCEPTED, RunState.TERMINAL_REJECTED}:
            if value["prepared_report_sha256"] is None or value["terminal_report_sha256"] is None:
                raise ValueError("terminal state binding is incomplete")
        elif state is RunState.FAILED:
            if value["terminal_report_sha256"] is not None:
                raise ValueError("failed state binding is invalid")
        else:
            raise ValueError("unknown run state")
        return value

    def _check_owner(self, state: dict[str, Any], owner_id: str) -> None:
        _valid_owner(owner_id)
        if state.get("owner_id") != owner_id:
            raise ValueError("run owner mismatch")
        lease = _read_lease(self._run(state["generation_id"]))
        if lease["generation_id"] != state["generation_id"] or lease["owner_id"] != owner_id:
            raise ValueError("run owner mismatch")
        if lease["expires_at"] <= time.time():
            raise ValueError("run owner lease expired")

    def reserve(
        self,
        generation_id: str,
        *,
        owner_id: str = "default-owner",
        lease_seconds: int = _DEFAULT_LEASE_SECONDS,
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        _reject_link_chain(self.root, allow_missing=False)
        path = self._run(generation_id)
        staging = Path(tempfile.mkdtemp(prefix=f".reserve-{generation_id}.", dir=self.root))
        try:
            _atomic_write(
                staging / "state.json",
                canonical_json_bytes(
                    {
                        "generation_id": generation_id,
                        "state": RunState.RESERVED.value,
                        "owner_id": _valid_owner(owner_id),
                        "prepared_report_sha256": None,
                        "terminal_report_sha256": None,
                    }
                ),
            )
            _atomic_write(
                staging / ".owner.lease",
                canonical_json_bytes(_lease_payload(generation_id, owner_id, lease_seconds)),
            )
            _reject_link_chain(path, allow_missing=True)
            if path.exists():
                raise FileExistsError(path)
            # Unlike os.replace, rename cannot replace the populated run
            # directory on Windows and fails closed if a concurrent writer wins.
            os.rename(staging, path)
            return path
        except BaseException:
            if staging.is_dir() and not _is_link(staging):
                for child in staging.iterdir():
                    if child.is_file() and not _is_link(child):
                        child.unlink()
                staging.rmdir()
            raise

    def _transition(
        self, generation_id: str, owner_id: str, expected: RunState, target: RunState
    ) -> None:
        current = self._read_state(generation_id)
        self._check_owner(current, owner_id)
        if current.get("state") != expected.value:
            raise ValueError(f"invalid state transition from {current.get('state')}")
        allowed = {
            (RunState.RESERVED, RunState.RUNNING),
        }
        if (expected, target) not in allowed:
            raise ValueError("invalid state transition")
        current["state"] = target.value
        _atomic_write(self._run(generation_id) / "state.json", canonical_json_bytes(current))

    def _with_lock(self, generation_id: str, owner_id: str, action):
        with self._lock(generation_id, owner_id):
            return action()

    def renew_owner_lease(
        self,
        generation_id: str,
        *,
        owner_id: str = "default-owner",
        lease_seconds: int = _DEFAULT_LEASE_SECONDS,
    ) -> None:
        run = self._run(generation_id)
        with self._lock(generation_id, owner_id):
            _atomic_write(
                run / ".owner.lease",
                canonical_json_bytes(_lease_payload(generation_id, owner_id, lease_seconds)),
            )

    def mark_running(self, generation_id: str, *, owner_id: str = "default-owner") -> None:
        self._with_lock(
            generation_id, owner_id, lambda: self._transition(generation_id, owner_id, RunState.RESERVED, RunState.RUNNING)
        )

    def mark_prepared(
        self, generation_id: str, decision: PreparedDecision, *, owner_id: str = "default-owner"
    ) -> Path:
        if decision.generation_id != generation_id:
            raise ValueError("decision generation mismatch")
        run = self._run(generation_id)
        with self._lock(generation_id, owner_id):
            state = self._read_state(generation_id)
            self._check_owner(state, owner_id)
            if decision.decision == "accepted" and decision.parent_generation_id is None:
                raise ValueError("accepted schema v1 unsupported")
            if (
                (run / "failure.json").exists()
                or (run / "report.json").exists()
                or (run / "terminal-receipt.json").exists()
            ):
                raise ValueError("cannot prepare after terminal marker")
            prepared_path = run / "prepared-report.json"
            if state["state"] == RunState.PREPARED.value:
                _, _, _, source, prepared_sha256 = self._read_prepared_report(
                    run, generation_id
                )
                if prepared_sha256 != state["prepared_report_sha256"] or not _decision_equal(
                    source, decision
                ):
                    raise ValueError("prepared binding conflict")
                return prepared_path
            if state["state"] != RunState.RUNNING.value:
                raise ValueError("preparation requires RUNNING state")
            # A crash can leave the immutable marker after the state write;
            # bind it only after validating the complete source and snapshots.
            if prepared_path.exists():
                _, _, _, source, prepared_sha256 = self._read_prepared_report(
                    run, generation_id
                )
                if not _decision_equal(source, decision):
                    raise ValueError("prepared binding conflict")
            else:
                snapshot = self._snapshot_decision(run, decision)
                payload = _prepared_payload(decision, snapshot)
                self._publish_marker(
                    prepared_path,
                    canonical_json_bytes(payload),
                    "prepared report must be regular non-symlink",
                )
                prepared_sha256 = sha256_file(prepared_path)
            state = self._read_state(generation_id)
            state["state"] = RunState.PREPARED.value
            state["prepared_report_sha256"] = prepared_sha256
            _atomic_write(run / "state.json", canonical_json_bytes(state))
            return prepared_path

    def _publish_regular_no_replace(self, path: Path, data: bytes, error_message: str) -> None:
        _reject_link_chain(path.parent, allow_missing=False)
        if path.exists():
            if _is_link(path) or not path.is_file():
                raise ValueError(error_message)
            if path.read_bytes() == data:
                _fsync_directory(path.parent)
                return
            raise ValueError("existing immutable file conflict")
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            _reject_link_chain(path, allow_missing=True)
            try:
                os.link(name, path)
            except FileExistsError as exc:
                if not _is_link(path) and path.is_file() and path.read_bytes() == data:
                    _fsync_directory(path.parent)
                    return
                raise ValueError("existing immutable file conflict") from exc
            _fsync_directory(path.parent)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _publish_marker(self, path: Path, data: bytes, error_message: str) -> Path:
        self._publish_regular_no_replace(path, data, error_message)
        return path

    def _snapshot_decision(self, run: Path, decision: PreparedDecision) -> PreparedDecision:
        evidence = decision.validate_files()
        digest = {
            "manifest": decision.manifest_sha256,
            "candidate": decision.candidate_checkpoint_sha256,
            "holdout": decision.holdout_evidence_sha256,
        }
        snapshot_dir = run / "evidence"
        snapshot_dir.mkdir(exist_ok=True)
        _reject_link_chain(snapshot_dir, allow_missing=False)
        paths: dict[str, str] = {}
        for role in ("manifest", "candidate", "holdout"):
            target = snapshot_dir / f"{role}-{digest[role]}.bin"
            self._publish_regular_no_replace(
                target,
                evidence[role],
                "evidence snapshot must be regular non-symlink",
            )
            paths[role] = str(target)
        return decision.with_snapshot_paths(paths)

    def _read_prepared_report(
        self, run: Path, generation_id: str
    ) -> tuple[Path, dict[str, Any], PreparedDecision, PreparedDecision, str]:
        return _read_prepared_report(run, generation_id)

    def _terminal_contract(
        self, payload: Any, generation_id: str
    ) -> tuple[PreparedDecision, PreparedDecision]:
        return _terminal_contract(payload, generation_id)

    def _expected_snapshot(self, run: Path, decision: PreparedDecision) -> PreparedDecision:
        digest = {
            "manifest": decision.manifest_sha256,
            "candidate": decision.candidate_checkpoint_sha256,
            "holdout": decision.holdout_evidence_sha256,
        }
        return decision.with_snapshot_paths(
            {
                role: str(run / "evidence" / f"{role}-{digest[role]}.bin")
                for role in ("manifest", "candidate", "holdout")
            }
        )

    def _decision_matches_report(
        self, run: Path, decision: PreparedDecision, report: dict[str, str]
    ) -> bool:
        expected = self._expected_snapshot(run, decision)
        return all(
            report.get(key) == value
            for key, value in {
                "generation_id": decision.generation_id,
                "decision": decision.decision,
                "manifest_sha256": decision.manifest_sha256,
                "candidate_checkpoint_sha256": decision.candidate_checkpoint_sha256,
                "holdout_evidence_sha256": decision.holdout_evidence_sha256,
                "source_manifest_path": str(decision.manifest_path),
                "source_candidate_checkpoint_path": str(decision.candidate_checkpoint_path),
                "source_holdout_evidence_path": str(decision.holdout_evidence_path),
                "manifest_path": expected.manifest_path,
                "candidate_checkpoint_path": expected.candidate_checkpoint_path,
                "holdout_evidence_path": expected.holdout_evidence_path,
                "parent_generation_id": decision.parent_generation_id,
                "parent_checkpoint_sha256": decision.parent_checkpoint_sha256,
                "parent_manifest_sha256": decision.parent_manifest_sha256,
            }.items()
        )

    def _read_terminal_report(
        self, generation_id: str
    ) -> tuple[Path, dict[str, Any], str]:
        run = self._run(generation_id)
        if (run / "failure.json").exists() or (run / "terminal-receipt.json").exists():
            raise ValueError("conflicting terminal markers")
        report_path = run / "report.json"
        report_bytes = _regular_bytes(report_path)
        try:
            report = _strict_json_bytes(report_bytes)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("malformed terminal report") from exc
        if not (run / "prepared-report.json").is_file():
            raise ValueError("terminal report has no prepared report")
        _, prepared, prepared_snapshot, prepared_source, prepared_sha256 = (
            self._read_prepared_report(run, generation_id)
        )
        snapshot, source = _terminal_contract(report, generation_id)
        if not _decision_equal(snapshot, prepared_snapshot) or not _decision_equal(
            source, prepared_source
        ):
            raise ValueError("terminal/prepared binding mismatch")
        snapshot.validate_files()
        state = self._read_state(generation_id)
        if state["prepared_report_sha256"] != prepared_sha256:
            raise ValueError("prepared report/state digest contradiction")
        expected = _terminal_payload(prepared, prepared_sha256, prepared_snapshot)
        if report != expected:
            raise ValueError("terminal/prepared binding mismatch")
        return report_path, report, sha256_bytes(report_bytes)

    def _validate_terminal_state(
        self, generation_id: str, report: dict[str, Any], report_sha256: str
    ) -> RunState:
        state = self._read_state(generation_id)
        target = (
            RunState.TERMINAL_ACCEPTED
            if report["decision"] == "accepted"
            else RunState.TERMINAL_REJECTED
        )
        if state["state"] == target.value:
            if state["terminal_report_sha256"] != report_sha256:
                raise ValueError("state/terminal report digest contradiction")
            return target
        if (
            state["state"] == RunState.PREPARED.value
            and state["terminal_report_sha256"] is None
        ):
            return target
        raise ValueError("report/state contradiction")

    def _read_failure_marker(self, generation_id: str) -> dict[str, str]:
        path = self._run(generation_id) / "failure.json"
        data = _regular_bytes(path)
        try:
            payload = _strict_json_bytes(data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("malformed failure marker") from exc
        if (
            not isinstance(payload, dict)
            or set(payload) != {"generation_id", "reason"}
            or payload.get("generation_id") != generation_id
            or not isinstance(payload.get("reason"), str)
            or not payload["reason"]
        ):
            raise ValueError("invalid failure marker contract")
        return payload

    def _accepted_terminal_is_bound(self, report: dict[str, str]) -> bool:
        return (
            report.get("schema_version") == 2
            and report.get("decision") == "accepted"
            and report.get("parent_generation_id") is not None
        )

    def _require_committed_accepted_terminal(
        self,
        generation_id: str,
        report: dict[str, str],
        report_sha256: str,
        state: dict[str, Any],
    ) -> ParentPointer | None:
        if report.get("decision") != "accepted":
            return None
        current = self.read_parent()
        if (
            current is None
            or current.generation_id != generation_id
            or current.checkpoint_sha256 != report["candidate_checkpoint_sha256"]
            or current.manifest_sha256 != report["manifest_sha256"]
            or current.report_sha256 != report_sha256
        ):
            raise ValueError("accepted terminal recovery parent conflict")
        return current

    def publish_terminal(
        self, generation_id: str, *, owner_id: str = "default-owner"
    ) -> Path:
        run = self._run(generation_id)
        with self._lock(generation_id, owner_id):
            state = self._read_state(generation_id)
            self._check_owner(state, owner_id)
            if (run / "failure.json").exists():
                raise ValueError("cannot publish terminal after failure marker")
            if (run / "terminal-receipt.json").exists():
                raise ValueError("legacy terminal receipt is unsupported")
            report_path = run / "report.json"
            if report_path.exists():
                report_path, report, report_sha256 = self._read_terminal_report(
                    generation_id
                )
                if report["decision"] == "accepted":
                    self._require_committed_accepted_terminal(
                        generation_id, report, report_sha256, state
                    )
                target = self._validate_terminal_state(
                    generation_id, report, report_sha256
                )
                if state["state"] == RunState.PREPARED.value:
                    current = self._read_state(generation_id)
                    self._check_owner(current, owner_id)
                    current["state"] = target.value
                    current["terminal_report_sha256"] = report_sha256
                    _atomic_write(run / "state.json", canonical_json_bytes(current))
                return report_path
            if state["state"] != RunState.PREPARED.value:
                raise ValueError("terminal publication requires PREPARED state")
            _, prepared, snapshot, _, prepared_sha256 = (
                self._read_prepared_report(run, generation_id)
            )
            if state["prepared_report_sha256"] != prepared_sha256:
                raise ValueError("prepared report/state digest contradiction")
            if snapshot.decision == "accepted":
                raise ValueError("accepted terminal requires unified commit")
            report = _terminal_payload(prepared, prepared_sha256, snapshot)
            self._terminal_contract(report, generation_id)
            report_bytes = canonical_json_bytes(report)
            self._publish_marker(
                report_path,
                report_bytes,
                "terminal report must be regular non-symlink",
            )
            persisted = _regular_bytes(report_path)
            if persisted != report_bytes:
                raise ValueError("terminal report digest mismatch or tampered terminal report")
            report_sha256 = sha256_bytes(persisted)
            current = self._read_state(generation_id)
            self._check_owner(current, owner_id)
            current["state"] = (
                RunState.TERMINAL_ACCEPTED.value
                if snapshot.decision == "accepted"
                else RunState.TERMINAL_REJECTED.value
            )
            current["terminal_report_sha256"] = report_sha256
            _atomic_write(run / "state.json", canonical_json_bytes(current))
            return report_path

    def publish_failure(
        self, generation_id: str, reason: str, *, owner_id: str = "default-owner"
    ) -> Path:
        if not isinstance(reason, str) or not reason:
            raise ValueError("failure reason is required")
        run = self._run(generation_id)
        with self._lock(generation_id, owner_id):
            state = self._read_state(generation_id)
            self._check_owner(state, owner_id)
            if (run / "report.json").exists() or (run / "terminal-receipt.json").exists():
                raise ValueError("cannot publish failure after terminal marker")
            failure_path = run / "failure.json"
            if failure_path.exists():
                failure = self._read_failure_marker(generation_id)
                if failure.get("reason") != reason:
                    raise ValueError("failure marker conflicts with requested reason")
                if state["state"] == RunState.FAILED.value:
                    if state["prepared_report_sha256"] is not None:
                        _, _, _, _, prepared_sha256 = self._read_prepared_report(
                            run, generation_id
                        )
                        if state["prepared_report_sha256"] != prepared_sha256:
                            raise ValueError("prepared report/state digest contradiction")
                    return failure_path
                if state["state"] not in {
                    RunState.RESERVED.value,
                    RunState.RUNNING.value,
                    RunState.PREPARED.value,
                }:
                    raise ValueError("failure marker conflicts with run state")
                if state["state"] == RunState.PREPARED.value:
                    _, _, _, _, prepared_sha256 = self._read_prepared_report(
                        run, generation_id
                    )
                    if state["prepared_report_sha256"] != prepared_sha256:
                        raise ValueError("prepared report/state digest contradiction")
                elif (run / "prepared-report.json").exists():
                    raise ValueError("prepared marker conflicts with run state")
                current = self._read_state(generation_id)
                self._check_owner(current, owner_id)
                current["state"] = RunState.FAILED.value
                _atomic_write(run / "state.json", canonical_json_bytes(current))
                return failure_path
            if state["state"] == RunState.FAILED.value:
                raise ValueError("failure publication requires nonterminal state")
            if state["state"] not in {
                RunState.RESERVED.value,
                RunState.RUNNING.value,
                RunState.PREPARED.value,
            }:
                raise ValueError("failure publication requires nonterminal state")
            if state["state"] == RunState.PREPARED.value:
                _, _, _, _, prepared_sha256 = self._read_prepared_report(
                    run, generation_id
                )
                if state["prepared_report_sha256"] != prepared_sha256:
                    raise ValueError("prepared report/state digest contradiction")
            elif (run / "prepared-report.json").exists():
                raise ValueError("prepared marker conflicts with run state")
            data = canonical_json_bytes({"generation_id": generation_id, "reason": reason})
            self._publish_marker(
                failure_path,
                data,
                "failure marker must be regular non-symlink",
            )
            current = self._read_state(generation_id)
            self._check_owner(current, owner_id)
            current["state"] = RunState.FAILED.value
            _atomic_write(run / "state.json", canonical_json_bytes(current))
            return failure_path

    def recover(self, generation_id: str, *, owner_id: str = "default-owner") -> RunState:
        run = self._run(generation_id)
        with self._lock(generation_id, owner_id):
            state = self._read_state(generation_id)
            self._check_owner(state, owner_id)
            current = RunState(state["state"])
            report_path = run / "report.json"
            failure_path = run / "failure.json"
            prepared_path = run / "prepared-report.json"
            if report_path.exists() and failure_path.exists():
                raise ValueError("conflicting terminal markers")
            if (run / "terminal-receipt.json").exists():
                raise ValueError("legacy terminal receipt is unsupported")
            if report_path.exists():
                _, report, report_sha256 = self._read_terminal_report(generation_id)
                if report["decision"] == "accepted":
                    self._require_committed_accepted_terminal(
                        generation_id, report, report_sha256, state
                    )
                target = self._validate_terminal_state(
                    generation_id, report, report_sha256
                )
                if current is target:
                    return target
                if current is not RunState.PREPARED:
                    raise ValueError("report marker conflicts with run state")
                current_state = self._read_state(generation_id)
                self._check_owner(current_state, owner_id)
                current_state["state"] = target.value
                current_state["terminal_report_sha256"] = report_sha256
                _atomic_write(run / "state.json", canonical_json_bytes(current_state))
                return target
            if failure_path.exists():
                self._read_failure_marker(generation_id)
                if current is RunState.FAILED:
                    if state["prepared_report_sha256"] is not None:
                        _, _, _, _, prepared_sha256 = self._read_prepared_report(
                            run, generation_id
                        )
                        if state["prepared_report_sha256"] != prepared_sha256:
                            raise ValueError("prepared report/state digest contradiction")
                    return current
                if current not in {
                    RunState.RESERVED,
                    RunState.RUNNING,
                    RunState.PREPARED,
                }:
                    raise ValueError("failure marker conflicts with run state")
                if current is RunState.PREPARED:
                    _, _, _, _, prepared_sha256 = self._read_prepared_report(
                        run, generation_id
                    )
                    if state["prepared_report_sha256"] != prepared_sha256:
                        raise ValueError("prepared report/state digest contradiction")
                elif (run / "prepared-report.json").exists():
                    raise ValueError("prepared marker conflicts with run state")
                current_state = self._read_state(generation_id)
                self._check_owner(current_state, owner_id)
                current_state["state"] = RunState.FAILED.value
                _atomic_write(run / "state.json", canonical_json_bytes(current_state))
                return RunState.FAILED
            if prepared_path.exists():
                _, _, _, _, prepared_sha256 = self._read_prepared_report(
                    run, generation_id
                )
                if current is RunState.PREPARED:
                    if state["prepared_report_sha256"] != prepared_sha256:
                        raise ValueError("prepared report/state digest contradiction")
                    return current
                if current is RunState.RUNNING:
                    current_state = self._read_state(generation_id)
                    self._check_owner(current_state, owner_id)
                    current_state["state"] = RunState.PREPARED.value
                    current_state["prepared_report_sha256"] = prepared_sha256
                    _atomic_write(run / "state.json", canonical_json_bytes(current_state))
                    return RunState.PREPARED
                raise ValueError("prepared marker conflicts with run state")
            if current is RunState.PREPARED:
                raise ValueError("prepared report marker is missing")
            if current in _TERMINAL:
                raise ValueError("terminal marker is missing")
            return current

    def _accepted_decision_binds(
        self, generation_id: str, replacement: ParentPointer, decision: PreparedDecision
    ) -> None:
        if decision.generation_id != generation_id or decision.decision != "accepted":
            raise ValueError("parent CAS requires an accepted prepared decision")
        state = self._read_state(generation_id)
        self._check_owner(state, replacement.owner_id)
        _, report, report_sha256 = self._read_terminal_report(generation_id)
        self._validate_terminal_state(generation_id, report, report_sha256)
        parent_fields = (
            decision.parent_generation_id,
            decision.parent_checkpoint_sha256,
            decision.parent_manifest_sha256,
        )
        if parent_fields[0] is None or parent_fields != (
            report.get("parent_generation_id"),
            report.get("parent_checkpoint_sha256"),
            report.get("parent_manifest_sha256"),
        ):
            raise ValueError("parent lineage binding mismatch")
        if (
            report["candidate_checkpoint_sha256"] != replacement.checkpoint_sha256
            or report["manifest_sha256"] != replacement.manifest_sha256
            or not self._decision_matches_report(self._run(generation_id), decision, report)
        ):
            raise ValueError("accepted decision receipt mismatch")
        decision.validate_files()
        if report_sha256 != replacement.report_sha256:
            raise ValueError("accepted decision report digest mismatch")

    def _parent_pointer_lock(self):
        """Hold the crash-released OS lock used for parent compare-and-swap."""
        return _acquire_file_lock(self.root / ".parent.lock", "parent pointer CAS is locked")

    def _accepted_payload_preflight(
        self,
        expected: ParentPointer,
        replacement: ParentPointer,
        decision: PreparedDecision,
    ) -> tuple[PreparedDecision, dict[str, Any], str]:
        """Validate everything needed to commit before the first durable write.

        Returns the snapshot decision, exact canonical terminal payload, and
        the payload digest. Raises before any mutation on any mismatch.
        """
        generation_id = decision.generation_id
        if decision.decision != "accepted":
            raise ValueError("accepted commit requires an accepted prepared decision")
        if replacement.generation_id != generation_id:
            raise ValueError("accepted commit generation mismatch")
        parent_fields = (
            decision.parent_generation_id,
            decision.parent_checkpoint_sha256,
            decision.parent_manifest_sha256,
        )
        if parent_fields[0] is None:
            raise ValueError("accepted commit requires a bound parent lineage")
        if parent_fields != (
            expected.generation_id,
            expected.checkpoint_sha256,
            expected.manifest_sha256,
        ):
            raise ValueError("parent lineage binding mismatch")
        state = self._read_state(generation_id)
        if state["state"] not in {
            RunState.PREPARED.value,
            RunState.TERMINAL_ACCEPTED.value,
        }:
            raise ValueError(
                "accepted commit requires PREPARED or TERMINAL_ACCEPTED state"
            )
        if state["prepared_report_sha256"] is None:
            raise ValueError("prepared report/state digest contradiction")
        run = self._run(generation_id)
        if (run / "failure.json").exists():
            raise ValueError("cannot commit an accepted terminal after a failure marker")
        if (run / "terminal-receipt.json").exists():
            raise ValueError("legacy terminal receipt is unsupported")
        _, prepared, prepared_snapshot, prepared_source, prepared_sha256 = (
            self._read_prepared_report(run, generation_id)
        )
        if state["prepared_report_sha256"] != prepared_sha256:
            raise ValueError("prepared report/state digest contradiction")
        if not _decision_equal(decision, prepared_source):
            raise ValueError("accepted decision does not match the prepared source binding")
        if prepared_snapshot.decision != "accepted":
            raise ValueError("prepared report is not an accepted decision")
        prepared_snapshot.validate_files()
        report = _terminal_payload(prepared, prepared_sha256, prepared_snapshot)
        self._terminal_contract(report, generation_id)
        if (
            report["candidate_checkpoint_sha256"] != replacement.checkpoint_sha256
            or report["manifest_sha256"] != replacement.manifest_sha256
            or report["parent_generation_id"] != decision.parent_generation_id
            or report["parent_checkpoint_sha256"] != decision.parent_checkpoint_sha256
            or report["parent_manifest_sha256"] != decision.parent_manifest_sha256
        ):
            raise ValueError("accepted decision receipt mismatch")
        report_bytes = canonical_json_bytes(report)
        report_sha256 = sha256_bytes(report_bytes)
        if replacement.report_sha256 != report_sha256:
            raise ValueError("replacement report digest does not match the terminal payload")
        report_path = run / "report.json"
        if report_path.exists():
            persisted = _regular_bytes(report_path)
            if persisted != report_bytes:
                raise ValueError(
                    "terminal report digest mismatch or tampered terminal report"
                )
            if state["state"] == RunState.TERMINAL_ACCEPTED.value:
                if state["terminal_report_sha256"] != report_sha256:
                    raise ValueError("state/terminal report digest contradiction")
            elif state["terminal_report_sha256"] is not None:
                raise ValueError("state/terminal report digest contradiction")
        elif state["state"] == RunState.TERMINAL_ACCEPTED.value:
            raise ValueError("terminal report marker is missing")
        return prepared_snapshot, report, report_sha256

    def commit_accepted_terminal(
        self,
        expected: ParentPointer,
        replacement: ParentPointer,
        decision: PreparedDecision,
        *,
        owner_id: str = "default-owner",
    ) -> Path:
        """Commit an accepted generation with ``current-parent.json`` as the sole point.

        Ordering is normative: under ``.parent.lock`` then the generation
        recovery fence, revalidate the prepared snapshots and lineage, replace
        the parent pointer (the commit point), then no-replace publish the exact
        terminal report and finish ``state.json``. A crash before the pointer
        replacement leaves the incumbent pointer and ``PREPARED`` authoritative; a
        crash after it is an idempotent completion of a committed transaction.
        """
        _valid_owner(owner_id)
        self.root.mkdir(parents=True, exist_ok=True)
        _reject_link_chain(self.root, allow_missing=False)
        path = self.root / "current-parent.json"
        with self._parent_pointer_lock(), ExitStack() as locks:
            generation_id = decision.generation_id
            locks.enter_context(self._recovery_locks(generation_id))
            state = self._read_state(generation_id)
            self._check_owner(state, owner_id)
            # Full preflight runs under the locks so nothing can change between
            # validation and the first durable write.
            prepared_snapshot, report, report_sha256 = self._accepted_payload_preflight(
                expected, replacement, decision
            )
            current = self.read_parent()
            if current is None:
                raise ValueError("parent pointer CAS conflict")
            if current != expected and _parent_identity(current) != _parent_identity(replacement):
                raise ValueError("parent pointer CAS conflict")
            run = self._run(generation_id)
            report_path = run / "report.json"
            committed = _parent_identity(current) == _parent_identity(replacement)
            if not committed:
                if replacement.owner_id != owner_id:
                    raise ValueError("uncommitted accepted replacement owner mismatch")
                state = self._read_state(generation_id)
                if state["state"] != RunState.PREPARED.value:
                    raise ValueError(
                        "uncommitted accepted replacement requires PREPARED state"
                    )
                if report_path.exists():
                    raise ValueError("accepted terminal exists before parent commit")
                _atomic_write(path, canonical_json_bytes(replacement.as_dict()))
            # The commit point passed: finishing the transaction is idempotent.
            report_bytes = canonical_json_bytes(report)
            self._publish_marker(
                report_path,
                report_bytes,
                "terminal report must be regular non-symlink",
            )
            persisted = _regular_bytes(report_path)
            if persisted != report_bytes:
                raise ValueError("terminal report digest mismatch or tampered terminal report")
            state = self._read_state(generation_id)
            if state["terminal_report_sha256"] is None:
                state["state"] = RunState.TERMINAL_ACCEPTED.value
                state["terminal_report_sha256"] = report_sha256
                _atomic_write(run / "state.json", canonical_json_bytes(state))
            elif state["state"] != RunState.TERMINAL_ACCEPTED.value or state[
                "terminal_report_sha256"
            ] != report_sha256:
                raise ValueError("state/terminal report digest contradiction")
            return report_path

    def compare_and_swap_parent(
        self,
        expected: ParentPointer | None,
        replacement: ParentPointer,
        *,
        owner_id: str = "default-owner",
        accepted_decision: PreparedDecision | None = None,
        trusted_parent_token: TrustedParentToken | None = None,
    ) -> None:
        _valid_owner(owner_id)
        if replacement.owner_id != owner_id:
            raise ValueError("replacement parent owner mismatch")
        if expected is not None and expected.owner_id != owner_id:
            raise ValueError("expected parent owner mismatch")
        if expected is None:
            if accepted_decision is not None:
                raise ValueError("bootstrap does not accept a replacement decision")
            if trusted_parent_token is None:
                raise ValueError("bootstrap requires an explicit trusted parent token")
            if (
                trusted_parent_token.owner_id != owner_id
                or trusted_parent_token.pointer_sha256 != replacement.digest()
            ):
                raise ValueError("trusted parent token mismatch")
        else:
            if trusted_parent_token is not None:
                raise ValueError("replacement CAS does not accept bootstrap token")
            if accepted_decision is None:
                raise ValueError("replacement CAS requires an accepted prepared decision")
            self._accepted_decision_binds(replacement.generation_id, replacement, accepted_decision)
            if accepted_decision.parent_generation_id is not None and (
                expected.generation_id,
                expected.checkpoint_sha256,
                expected.manifest_sha256,
            ) != (
                accepted_decision.parent_generation_id,
                accepted_decision.parent_checkpoint_sha256,
                accepted_decision.parent_manifest_sha256,
            ):
                raise ValueError("parent lineage binding mismatch")
        self.root.mkdir(parents=True, exist_ok=True)
        _reject_link_chain(self.root, allow_missing=False)
        path = self.root / "current-parent.json"
        with self._parent_pointer_lock(), ExitStack() as locks:
            if accepted_decision is not None:
                # The generation fence must cover the complete pointer CAS,
                # not just the last validation. Lease takeover uses the same
                # fence, so it cannot transfer ownership between validation
                # and publication.
                locks.enter_context(self._recovery_locks(accepted_decision.generation_id))
                self._accepted_decision_binds(
                    accepted_decision.generation_id, replacement, accepted_decision
                )
            _reject_link_chain(path, allow_missing=True)
            if path.exists():
                try:
                    current = _strict_json_bytes(_regular_bytes(path))
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    raise ValueError("malformed current-parent.json") from exc
                if expected is None:
                    if not isinstance(current, dict):
                        raise ValueError("malformed current-parent.json")
                    if current == replacement.as_dict():
                        return
                    raise ValueError("parent pointer already exists")
                if not isinstance(current, dict) or current.get("owner_id") != owner_id:
                    raise ValueError("current parent owner mismatch")
                if current != expected.as_dict():
                    if current == replacement.as_dict():
                        return
                    raise ValueError("parent pointer CAS conflict")
            elif expected is not None:
                raise ValueError("parent pointer CAS conflict")
            _atomic_write(path, canonical_json_bytes(replacement.as_dict()))

    def read_parent(self) -> ParentPointer | None:
        """Read and validate the current parent pointer, if present."""
        self.root.mkdir(parents=True, exist_ok=True)
        _reject_link_chain(self.root, allow_missing=False)
        path = self.root / "current-parent.json"
        _reject_link_chain(path, allow_missing=True)
        if not path.exists():
            return None
        data = _regular_bytes(path)
        try:
            value = _strict_json_bytes(data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("malformed current-parent.json") from exc
        if not isinstance(value, dict):
            raise ValueError("invalid current-parent.json contract")
        required = {
            "generation_id", "checkpoint_sha256", "manifest_sha256",
            "report_sha256", "owner_id",
        }
        if set(value) != required:
            raise ValueError("invalid current-parent.json contract")
        return ParentPointer(**value)

    def bootstrap_parent(
        self,
        pointer: ParentPointer,
        *,
        owner_id: str = "default-owner",
        trusted_parent_token: TrustedParentToken,
    ) -> None:
        self.compare_and_swap_parent(
            None,
            pointer,
            owner_id=owner_id,
            trusted_parent_token=trusted_parent_token,
        )
