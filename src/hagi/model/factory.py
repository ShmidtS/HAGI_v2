"""Single place that decides which model class a config asks for.

``train.py`` and ``eval_holdout.py`` both need to rebuild the exact
architecture a checkpoint was written with. When that logic lives in the
script, the two drift: a checkpoint produced by the ``scratch_block_norm``
arm (arm S, the architecture-matched scratch control) could not be evaluated
because the evaluator always built a plain ``HAGI``, and the mismatch surfaced
as an opaque ``IncompatibleCheckpointError`` about ``mixers.*``.

Keeping the dispatch here means a new architecture variant has to be added
once, and an evaluation of a trained arm cannot silently use a different
model than the one that produced it.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from hagi.config import Config


def build_model_for_config(cfg: Config, *, n_blocks: int | None = None) -> nn.Module:
    """Return the model class ``cfg`` selects, untrained and on CPU.

    Args:
        cfg: the loaded configuration.
        n_blocks: override for the block count, used by the merged arms where
            the block count is the expert count rather than ``model.n_layers``.

    Returns:
        A freshly constructed module. The caller decides the device.

    Raises:
        ValueError: if the config names a merge variant this dispatch does not
            know, rather than silently falling back to a plain ``HAGI`` -- a
            silent fallback is what made the arm-S evaluation fail.
    """
    merge = getattr(cfg, "merge", None)
    if merge is not None and getattr(merge, "enabled", False):
        # The recursive ternary tree is a different body from a flat merge:
        # its blocks carry BlockTreeNorm and RecursiveBranchScale over
        # 3**ternary_leaf leaves, and it binds provenance buffers that
        # MergedHAGI never writes. Dispatching it to MergedHAGI is what made
        # every recursive config die at construction with
        # "HAGI cannot instantiate merge.mixer_type='ternary_f3'" -- the tree
        # existed, passed its design gates, and could not be built from a
        # config, which is why no recursive measurement exists.
        # A recursive parent cannot be built from a config alone: its body IS
        # the lift of three specific children, so the tree structures exist
        # only once those child states are supplied. Construction goes through
        # build_model_from_payload, which detects the provenance markers.
        if getattr(merge, "mixer_type", None) == "ternary_f3":
            raise ValueError(
                "ternary_f3 has no untrained form; build it with "
                "hagi.model.merge.build_model_from_payload (from a checkpoint) "
                "or merge_recursive_f3 (from three child states)"
            )

        n_mixers = int(getattr(merge, "n_mixers", 1))
        if getattr(merge, "scratch_block_norm", False):
            from hagi.model.scratch_blocknorm import ScratchBlockNormHAGI

            return ScratchBlockNormHAGI(
                cfg, n_blocks=n_blocks if n_blocks is not None else int(merge.n_experts)
            )
        from hagi.model.merge import MergedHAGI

        return MergedHAGI(
            cfg, n_mixers=n_mixers, mixer_init_scale=merge.mixer_init_scale
        )

    from hagi.model.model import HAGI

    return HAGI(cfg)


def merge_block_count(cfg: Config) -> int | None:
    """Block count implied by a merge config, or None for a plain model."""
    merge: Any = getattr(cfg, "merge", None)
    if merge is None or not getattr(merge, "enabled", False):
        return None
    return int(merge.n_experts)
