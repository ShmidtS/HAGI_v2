"""Strict checkpoint save and inference loading."""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from hagi_v4.config import Config, _apply_dict, validate_config

logger = logging.getLogger(__name__)
CHECKPOINT_FORMAT_VERSION = 4
CHECKPOINT_FIELDS = {"format_version", "model", "config", "completed_updates"}
# Optional field persisted only by training checkpoints for resume. Inference
# loaders ignore it.
OPTIONAL_FIELDS = {"optimizer"}


class IncompatibleCheckpointError(RuntimeError):
    pass


def _incompatible(message: str) -> IncompatibleCheckpointError:
    detail = " ".join(message.split())
    return IncompatibleCheckpointError(f"incompatible checkpoint: {detail}; fresh retraining required")


def _require_mapping(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise _incompatible(f"{name} must be a mapping")
    return value


def load_checkpoint_payload(path: str | Path, device: str = "cpu") -> dict:
    """Safely deserialize and validate a checkpoint before any model mutation."""
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise _incompatible(f"checkpoint payload cannot be loaded: {exc}") from exc
    state = _require_mapping(state, "root")
    extra = set(state) - CHECKPOINT_FIELDS - OPTIONAL_FIELDS
    if extra:
        raise _incompatible(f"unknown checkpoint fields {sorted(extra)}")
    missing = CHECKPOINT_FIELDS - set(state)
    if missing:
        raise _incompatible(f"checkpoint schema missing {sorted(missing)}")
    if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise _incompatible(f"checkpoint schema has unsupported format_version {state.get('format_version')!r}")
    model_state = _require_mapping(state.get("model"), "model")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in model_state.items()):
        raise _incompatible("model must map parameter names to tensors")
    if not isinstance(state.get("config"), Mapping):
        raise _incompatible("config must be a mapping")
    if type(state.get("completed_updates")) is not int or state["completed_updates"] < 0:
        raise _incompatible("completed_updates must be a non-negative integer")
    return dict(state)


def cfg_to_dict(cfg: Config) -> dict:
    """Serialize config to a plain dict via dataclasses.asdict."""
    return dataclasses.asdict(cfg)


def cfg_from_dict(data: dict) -> Config:
    """Reconstruct config from a checkpoint dict (recursive)."""
    cfg = Config()
    for top_key in ("model", "train", "inference"):
        top_data = data.get(top_key, {})
        if isinstance(top_data, dict) and hasattr(cfg, top_key):
            _apply_dict(getattr(cfg, top_key), top_data)
    try:
        validate_config(cfg)
    except (TypeError, ValueError) as exc:
        raise _incompatible(f"checkpoint config is invalid: {exc}") from exc
    return cfg


def assert_fresh_checkpoint_root(path: str | Path) -> Path:
    """Ensure the checkpoint root exists."""
    ckpt_dir = Path(path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def load_model_checkpoint(path: str | Path, model: nn.Module, device: str) -> tuple[int, Config]:
    """Validate a complete checkpoint, then strictly load its model state."""
    state = load_checkpoint_payload(path, device)
    cfg = cfg_from_dict(state["config"])
    try:
        model.load_state_dict(state["model"], strict=True)
    except Exception as exc:
        raise _incompatible(f"model state_dict is not compatible: {exc}") from exc
    return state["completed_updates"], cfg


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the highest-numbered ``step-*.pt`` in ``checkpoint_dir`` or None."""
    ckpts = sorted(Path(checkpoint_dir).glob("step-*.pt"), key=lambda p: int(p.stem.removeprefix("step-")))
    return ckpts[-1] if ckpts else None


def save_checkpoint(
    model: nn.Module,
    cfg: Config,
    completed_updates: int,
    checkpoint_dir: str | Path,
    keep_last: int = 3,
    optimizer=None,
) -> Path:
    """Save a training checkpoint. Returns path to the saved file.

    ``optimizer`` (a CombinedOptimizer with state_dict/load_state_dict) is
    optional: when provided, its state is persisted so a ``--resume`` run
    continues with Muon momentum buffers + AdamW second moments intact.
    """
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError("completed_updates must be a non-negative integer")
    if type(keep_last) is not int or keep_last < 1:
        raise ValueError("keep_last must be an integer of at least 1")
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / f"step-{completed_updates:06d}.pt"
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "completed_updates": completed_updates,
        "config": cfg_to_dict(cfg),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    temp_file = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=ckpt_dir, delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    try:
        torch.save(state, temp_path)
        # os.replace atomically overwrites an existing destination on both
        # POSIX and Windows (os.link fails with FileExistsError on rerun).
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    logger.info(f"Checkpoint saved: {path}")

    checkpoints = sorted(ckpt_dir.glob("step-*.pt"), key=lambda p: int(p.stem.removeprefix("step-")))
    for old in checkpoints[:-keep_last]:
        old.unlink()
        logger.info(f"Old checkpoint deleted: {old}")

    return path
