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

from hagi_v4.config import HAGIv4Config, validate_config

logger = logging.getLogger(__name__)
CHECKPOINT_FORMAT_VERSION = 3
CHECKPOINT_FIELDS = {"format_version", "model", "config", "completed_updates"}


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
    if set(state) != CHECKPOINT_FIELDS:
        raise _incompatible(f"checkpoint schema expected {sorted(CHECKPOINT_FIELDS)}, got {sorted(state)}")
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


def cfg_to_dict(cfg: HAGIv4Config) -> dict:
    """Serialize config to a plain dict via dataclasses.asdict."""
    return dataclasses.asdict(cfg)


def cfg_from_dict(data: dict) -> HAGIv4Config:
    """Reconstruct config from a checkpoint dict.

    Handles V22→V23 migration: new optional fields (exit_chart, cqi,
    uncertainty, kalman, hrm, msa) absent in V22 checkpoints default to None.
    Dict values for None-default fields are instantiated into dataclasses.
    """
    cfg = HAGIv4Config()

    for top_key in ("model", "train", "inference"):
        top_val = getattr(cfg, top_key)
        top_data = data.get(top_key, {})
        for f_name, fv in top_data.items():
            if not hasattr(top_val, f_name):
                continue
            current = getattr(top_val, f_name)
            if hasattr(current, "__dataclass_fields__") and isinstance(fv, dict):
                for sf, sv in fv.items():
                    setattr(current, sf, sv)
            elif current is None and isinstance(fv, dict):
                _instantiate_optional_field(top_val, f_name, fv)
            else:
                setattr(top_val, f_name, fv)
    try:
        validate_config(cfg)
    except (TypeError, ValueError) as exc:
        raise _incompatible(f"checkpoint config is invalid: {exc}") from exc
    return cfg


def _instantiate_optional_field(obj, key: str, value: dict) -> None:
    """Instantiate a None-default optional dataclass field from a dict."""
    import dataclasses
    import sys
    import typing

    if not dataclasses.is_dataclass(obj):
        setattr(obj, key, value)
        return
    fields_map = {f.name: f for f in dataclasses.fields(obj)}
    if key not in fields_map:
        setattr(obj, key, value)
        return
    raw_type = fields_map[key].type
    mod = sys.modules.get(type(obj).__module__)
    cls = None
    if isinstance(raw_type, str):
        clean = raw_type.replace(" | None", "").replace("Optional[", "").replace("]", "").strip()
        cls = getattr(mod, clean, None) if mod else None
    elif raw_type is not None:
        cls = raw_type
    if cls and hasattr(cls, "__dataclass_fields__"):
        instance = cls()
        for sf, sv in value.items():
            if hasattr(instance, sf):
                setattr(instance, sf, sv)
        setattr(obj, key, instance)
    else:
        setattr(obj, key, value)


def assert_fresh_checkpoint_root(path: str | Path) -> Path:
    """Reject a checkpoint root containing flat ``step-*.pt`` artifacts that a
    new run would overwrite. Unrelated subfolders and non-matching files do not
    conflict with flat ``checkpoints/step-XXXXXX.pt`` output and are ignored."""
    ckpt_dir = Path(path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def load_model_checkpoint(path: str | Path, model: nn.Module, device: str) -> tuple[int, HAGIv4Config]:
    """Validate a complete checkpoint, then strictly load its model state."""
    state = load_checkpoint_payload(path, device)
    cfg = cfg_from_dict(state["config"])
    try:
        model.load_state_dict(state["model"], strict=True)
    except Exception as exc:
        raise _incompatible(f"model state_dict is not compatible: {exc}") from exc
    return state["completed_updates"], cfg


def save_checkpoint(
    model: nn.Module,
    cfg: HAGIv4Config,
    completed_updates: int,
    checkpoint_dir: str | Path,
    keep_last: int = 3,
) -> Path:
    """Save training checkpoint. Returns path to saved file."""
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
    temp_file = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=ckpt_dir, delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    try:
        torch.save(state, temp_path)

        # V12: ``os.replace`` atomically overwrites an existing destination
        # on both POSIX and Windows, where ``os.link`` fails with
        # ``FileExistsError`` if the target file already exists. This allows
        # resuming/re-running training without manually clearing the
        # checkpoint directory first.
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
