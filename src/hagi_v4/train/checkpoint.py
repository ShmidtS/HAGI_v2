"""Checkpoint management — save, load, resume, cleanup."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from hagi_v4.config import HAGIv4Config
from hagi_v4.train.optim import CombinedOptimizer

logger = logging.getLogger(__name__)
CHECKPOINT_FORMAT_VERSION = 2


class IncompatibleCheckpointError(RuntimeError):
    pass


def _require_mapping(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise IncompatibleCheckpointError(f"incompatible checkpoint: {name} must be a mapping")
    return value


def load_checkpoint_payload(path: str, device: str = "cpu") -> dict:
    """Safely deserialize and validate a checkpoint before any model mutation."""
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise IncompatibleCheckpointError(f"incompatible checkpoint payload: {exc}") from exc
    state = _require_mapping(state, "root")
    if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise IncompatibleCheckpointError(
            "incompatible checkpoint; start fresh training with a v2 checkpoint directory"
        )
    model_state = _require_mapping(state.get("model"), "model")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in model_state.items()):
        raise IncompatibleCheckpointError("incompatible checkpoint: model must map parameter names to tensors")
    if not isinstance(state.get("config"), Mapping):
        raise IncompatibleCheckpointError("incompatible checkpoint: config must be a mapping")
    if not isinstance(state.get("completed_updates"), int) or state["completed_updates"] < 0:
        raise IncompatibleCheckpointError("incompatible checkpoint: completed_updates must be a non-negative integer")
    if "optimizer" in state:
        optimizer_state = _require_mapping(state["optimizer"], "optimizer")
        if set(optimizer_state) != {"muon", "adamw"} or not all(
            isinstance(optimizer_state[name], Mapping) for name in ("muon", "adamw")
        ):
            raise IncompatibleCheckpointError("incompatible checkpoint: optimizer must contain muon and adamw states")
    extra = _require_mapping(state.get("extra", {}), "extra")
    if "rng" in extra:
        rng = _require_mapping(extra["rng"], "rng")
        if not isinstance(rng.get("torch"), torch.Tensor) or rng["torch"].dtype != torch.uint8:
            raise IncompatibleCheckpointError("incompatible checkpoint: rng.torch must be a uint8 tensor")
        if rng.get("cuda") is not None and (
            not isinstance(rng["cuda"], torch.Tensor) or rng["cuda"].dtype != torch.uint8
        ):
            raise IncompatibleCheckpointError("incompatible checkpoint: rng.cuda must be a uint8 tensor or None")
    return dict(state)


def cfg_to_dict(cfg: HAGIv4Config) -> dict:
    """Serialize config to a plain dict via dataclasses.asdict."""
    return dataclasses.asdict(cfg)


def cfg_from_dict(data: dict) -> HAGIv4Config:
    """Reconstruct config from a plain dict, preserving nested dataclass structure."""
    cfg = HAGIv4Config()
    for top_key in ("model", "train", "inference"):
        if top_key not in data:
            continue
        top_val = getattr(cfg, top_key)
        for f_name, fv in data[top_key].items():
            if hasattr(top_val, f_name):
                current = getattr(top_val, f_name)
                if hasattr(current, "__dataclass_fields__") and isinstance(fv, dict):
                    for sf, sv in fv.items():
                        if hasattr(current, sf):
                            setattr(current, sf, sv)
                else:
                    setattr(top_val, f_name, fv)
    return cfg


def save_checkpoint(
    model: nn.Module,
    optimizer: CombinedOptimizer | None,
    cfg: HAGIv4Config,
    completed_updates: int,
    checkpoint_dir: str,
    keep_last: int = 3,
    extra: dict | None = None,
) -> str:
    """Save training checkpoint. Returns path to saved file."""
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
    if extra:
        state["extra"] = extra

    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")

    # Cleanup old checkpoints (by mtime — newest survive, never delete the one just saved)
    checkpoints = sorted(ckpt_dir.glob("step-*.pt"), key=lambda p: p.stat().st_mtime)
    for old in checkpoints[:-keep_last]:
        old.unlink()
        logger.info(f"Old checkpoint deleted: {old}")

    return str(path)


def _migrate_state_dict(state_dict: dict) -> dict:
    """Migrate old checkpoint keys to current model architecture."""
    renamed = {}
    for key, val in state_dict.items():
        new_key = key
        if "scale_weights" in key:
            new_key = key.replace("scale_weights", "scale_gates")
        renamed[new_key] = val
    to_delete = [k for k in renamed if "attn_norm" in k]
    for k in to_delete:
        del renamed[k]
    return renamed


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: CombinedOptimizer | None = None,
    device: str = "cpu",
) -> tuple[int, HAGIv4Config, dict]:
    """Load checkpoint. Returns (step, config, extra).

    Optimizer state is returned in extra["optimizer"] when present, so a caller
    that builds the optimizer after resume can still restore momentum buffers.
    """
    state = load_checkpoint_payload(path, device)
    model.load_state_dict(state["model"])
    cfg = cfg_from_dict(state["config"])
    next_step = state["completed_updates"]
    extra = state.get("extra", {})
    if "optimizer" in state:
        extra["optimizer"] = state["optimizer"]
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    logger.info(f"Checkpoint loaded: {path} (next step {next_step})")
    return next_step, cfg, extra


def get_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint in a directory."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("step-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def resume_from_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: CombinedOptimizer | None = None,
    device: str = "cpu",
) -> tuple[int, HAGIv4Config | None, dict]:
    """Resume from latest checkpoint. Returns (step, config, extra) or (0, None, {})."""
    latest = get_latest_checkpoint(checkpoint_dir)
    if latest is None:
        logger.info(f"No checkpoint found in {checkpoint_dir} — starting from scratch")
        return 0, None, {}
    return load_checkpoint(latest, model, optimizer, device)
