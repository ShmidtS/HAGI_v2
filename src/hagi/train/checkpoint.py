"""Checkpoint save and load with strict schema validation.

A checkpoint is a contract between two processes that may be weeks apart. Every
field is validated before the model is touched, because a partially-applied load
produces a model that runs, trains, and is silently wrong — the most expensive
failure mode available.

Format 12 (V41). Deliberately incompatible with earlier receiver semantics:
the interleaved conditional bank changes the training contract even though most
parameter shapes remain compatible. A permissive load would silently resume the
wrong objective.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from hagi.config import CHECKPOINT_FORMAT_VERSION, Config, _apply_dict, validate_config

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"format_version", "model", "config", "completed_steps"}
OPTIONAL_FIELDS = {"optimizer"}


class IncompatibleCheckpointError(RuntimeError):
    """The checkpoint cannot be loaded into this code version."""


def _fail(message: str) -> IncompatibleCheckpointError:
    return IncompatibleCheckpointError(f"incompatible checkpoint: {message}; fresh training required")


def config_to_dict(cfg: Config) -> dict:
    """Serialize a config to plain data."""
    return dataclasses.asdict(cfg)


def config_from_dict(data: Mapping) -> Config:
    """Rebuild and validate a config from checkpoint data."""
    cfg = Config()
    try:
        _apply_dict(cfg, dict(data))
        validate_config(cfg)
    except (TypeError, ValueError) as exc:
        raise _fail(f"stored config is invalid: {exc}") from exc
    return cfg


def load_payload(path: str | Path, device: str = "cpu") -> dict:
    """Deserialize and fully validate a checkpoint without mutating anything.

    Raises:
        IncompatibleCheckpointError: on any schema violation.
    """
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise _fail(f"payload cannot be read: {exc}") from exc
    if not isinstance(state, Mapping):
        raise _fail("payload root must be a mapping")

    unknown = set(state) - REQUIRED_FIELDS - OPTIONAL_FIELDS
    if unknown:
        raise _fail(f"unknown fields {sorted(unknown)}")
    missing = REQUIRED_FIELDS - set(state)
    if missing:
        raise _fail(f"missing fields {sorted(missing)}")
    if state["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise _fail(
            f"format_version {state['format_version']!r}, this code writes {CHECKPOINT_FORMAT_VERSION}"
        )
    if not isinstance(state["model"], Mapping) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state["model"].items()
    ):
        raise _fail("model must map names to tensors")
    if not isinstance(state["config"], Mapping):
        raise _fail("config must be a mapping")
    if type(state["completed_steps"]) is not int or state["completed_steps"] < 0:
        raise _fail("completed_steps must be a non-negative integer")
    return dict(state)


def load_model(path: str | Path, model: nn.Module, device: str = "cpu") -> tuple[int, Config]:
    """Validate a checkpoint fully, then load its weights.

    The key and shape comparison happens *before* any copy. ``load_state_dict``
    with ``strict=True`` reports mismatched keys, but it reports them at the end —
    it has already copied every key that did match, so a rejected checkpoint would
    leave the model half-overwritten with no indication of which half. Since the
    caller catches the exception and may keep training, that partially-loaded
    model is precisely the silent failure this module exists to prevent.

    Returns:
        ``(completed_steps, config)``.
    """
    state = load_payload(path, device)
    cfg = config_from_dict(state["config"])

    incoming = state["model"]
    current = model.state_dict()
    missing = sorted(set(current) - set(incoming))
    unexpected = sorted(set(incoming) - set(current))
    if missing or unexpected:
        raise _fail(
            f"state_dict does not match the model: missing {missing[:8]}, unexpected {unexpected[:8]}"
        )
    mismatched = [
        f"{name}: checkpoint {tuple(incoming[name].shape)} vs model {tuple(current[name].shape)}"
        for name in current
        if incoming[name].shape != current[name].shape
    ]
    if mismatched:
        raise _fail(f"tensor shapes differ: {mismatched[:8]}")

    try:
        model.load_state_dict(incoming, strict=True)
    except Exception as exc:  # pragma: no cover - the checks above are exhaustive
        raise _fail(f"state_dict could not be applied: {exc}") from exc
    return state["completed_steps"], cfg


def latest_checkpoint(directory: str | Path) -> Path | None:
    """Highest-numbered ``step-*.pt`` in ``directory``, or None."""
    found = sorted(
        Path(directory).glob("step-*.pt"), key=lambda p: int(p.stem.removeprefix("step-"))
    )
    return found[-1] if found else None


def save_checkpoint(
    model: nn.Module,
    cfg: Config,
    completed_steps: int,
    directory: str | Path,
    keep_last: int = 3,
    optimizer=None,
) -> Path:
    """Write a checkpoint atomically and rotate old ones.

    Written to a temporary file then moved into place: a crash mid-write leaves
    the previous checkpoint intact rather than a truncated file that fails to load
    at the exact moment it is needed.

    Args:
        model: the model to serialize.
        cfg: the config that produced it.
        completed_steps: optimizer steps finished.
        directory: destination.
        keep_last: how many checkpoints to retain.
        optimizer: when given, its state is stored so a resume keeps Muon
            momentum and AdamW second moments. Without it a resume restarts the
            optimizer cold, which shows up as a loss spike.

    Returns:
        The written path.
    """
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("completed_steps must be a non-negative integer")
    if type(keep_last) is not int or keep_last < 1:
        raise ValueError("keep_last must be >= 1")

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step-{completed_steps:07d}.pt"

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "config": config_to_dict(cfg),
        "completed_steps": completed_steps,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    handle = tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp", dir=root, delete=False)
    temp_path = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    logger.info("checkpoint saved: %s", target)

    existing = sorted(root.glob("step-*.pt"), key=lambda p: int(p.stem.removeprefix("step-")))
    for old in existing[:-keep_last]:
        old.unlink()
        logger.info("checkpoint pruned: %s", old)
    return target
