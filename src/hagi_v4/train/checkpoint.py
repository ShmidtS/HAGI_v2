"""Checkpoint management — save, load, resume, cleanup."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import torch
import torch.nn as nn

from hagi_v4.config import HAGIv4Config
from hagi_v4.train.optim import CombinedOptimizer

logger = logging.getLogger(__name__)


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
    step: int,
    checkpoint_dir: str,
    keep_last: int = 3,
    extra: dict | None = None,
) -> str:
    """Save training checkpoint. Returns path to saved file."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / f"step-{step:06d}.pt"
    state = {
        "model": model.state_dict(),
        "step": step,
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
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(_migrate_state_dict(state["model"]))
    cfg = cfg_from_dict(state["config"])
    step = state["step"]
    extra = state.get("extra", {})
    if "optimizer" in state:
        extra["optimizer"] = state["optimizer"]
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    logger.info(f"Checkpoint loaded: {path} (step {step})")
    return step, cfg, extra


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
