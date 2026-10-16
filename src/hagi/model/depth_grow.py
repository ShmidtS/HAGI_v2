"""Function-preserving depth growth (net2net-style layer insertion).

Grows a trained root from ``num_layers = L`` to ``L' > L`` by appending
``L' - L`` identity blocks: every new block gets a zeroed attention
``out_proj`` and a zeroed mixer ``down`` projection, so its residual
contribution is exactly zero and the grown model reproduces the source
model bit-for-bit (ARCHITECTURE.md principle 3: function-preserving
invariants are mandatory; verified by tests/test_depth_grow.py).

Layer placement: new identity blocks are appended after the trained
ones. Existing layer weights keep their trained BranchScale and
residual contributions; the depth-derived ``residual_scale`` only
affects initialization, which the copied weights override.
"""

from __future__ import annotations

import copy

import torch

from hagi.config import Config, _apply_dict, validate_config
from hagi.train.checkpoint import load_payload


def grow_config(cfg: Config, target_layers: int) -> Config:
    """Return a copy of ``cfg`` with ``model.num_layers = target_layers``."""
    if target_layers <= cfg.model.num_layers:
        raise ValueError(
            f"target_layers={target_layers} must exceed current "
            f"num_layers={cfg.model.num_layers}"
        )
    grown = copy.deepcopy(cfg)
    grown.model.num_layers = target_layers
    validate_config(grown)
    return grown


def grow_state_dict(
    old_state: dict[str, torch.Tensor],
    old_layers: int,
    new_layers: int,
) -> dict[str, torch.Tensor]:
    """Clone the source state dict for loading into the grown model.

    Trained layers keep their weights unchanged (new blocks are
    appended after them, so key names match); the appended blocks
    contribute zero through zeroed ``out_proj`` / ``down`` weights
    applied by :func:`zero_new_blocks_`.
    """
    del old_layers, new_layers  # appending keeps trained key names intact
    return {key: tensor.clone() for key, tensor in old_state.items()}


def grow_checkpoint(path: str, target_layers: int, device: str = "cpu") -> dict:
    """Load a checkpoint and return a grown payload (config + state dict).

    The caller saves it (keep the same ``completed_steps``; the
    optimizer state is dropped, the grown root re-enters the cycle
    through leaves/joint as usual).
    """
    payload = load_payload(path, device=device)
    cfg_dict = dict(payload["config"])
    base = Config()
    _apply_dict(base, cfg_dict, lenient=True)
    validate_config(base)
    if target_layers <= base.model.num_layers:
        raise ValueError(
            f"target_layers={target_layers} must exceed checkpoint "
            f"num_layers={base.model.num_layers}"
        )
    return {
        "format_version": payload["format_version"],
        "config": cfg_dict
        | {
            "model": dict(cfg_dict.get("model", {}))
            | {
                "num_layers": target_layers,
                # Runtime marker: layers < grown_from_layers keep the source
                # model's residual_scale (BranchScale clamp center / adapter
                # gate); see fix_old_residual_scale_. Applied on every load,
                # not only at grow time.
                "grown_from_layers": base.model.num_layers,
            }
        },
        "model": {k: v.clone() for k, v in payload["model"].items()},
        "completed_steps": payload["completed_steps"],
    }


def fix_old_residual_scale_(
    model: torch.nn.Module,
    old_layers: int,
    old_residual_scale: float,
) -> None:
    """Restore the source model's ``residual_scale`` on the trained layers.

    ``residual_scale = (2 * num_layers * loop_depth) ** -0.5`` is not
    only an init constant: at runtime it is the clamp center of every
    ``BranchScale`` and the adapter gate scale. Rebuilding with a
    larger ``num_layers`` silently changes the clamp bounds of the
    trained layers and breaks function preservation; this restores
    the source value on layers ``< old_layers`` so the grown model is
    bit-for-bit identical (new layers are zeroed anyway).
    """
    for idx in range(old_layers):
        block = model.blocks[idx]
        for module in block.modules():
            if hasattr(module, "residual_scale"):
                module.residual_scale = old_residual_scale


def old_residual_scale(num_layers: int, loop_depth: int) -> float:
    """The constructor formula from ``HAGI.__init__`` (single source kept
    in sync with it)."""
    loop = max(1, int(loop_depth))
    return (2.0 * num_layers * loop) ** -0.5


def zero_new_blocks_(model: torch.nn.Module, first_new_layer: int) -> None:
    """Zero the residual-producing weights of layers >= first_new_layer.

    Applied after ``load_state_dict(strict=False)`` so the appended
    blocks are exact identities even though their constructor init drew
    nonzero weights (it always does).
    """
    blocks = model.blocks  # nn.ModuleList on HAGI
    for idx in range(first_new_layer, len(blocks)):
        block = blocks[idx]
        torch.nn.init.zeros_(block.attn.out_proj.weight)
        torch.nn.init.zeros_(block.mixer.mixer.down.weight)
