"""Explicit ownership boundary for trainable side components.

HAGI can train its frozen base together with one or more opt-in adaptive
contours.  ``AdaptiveComponent`` is a small marker rather than a name-based
heuristic: model modules opt into the contract, and training/rollback code
collects parameters by type.  This keeps the base-frozen path independent of
where a future adapter is attached (block-local or model-global).
"""

from __future__ import annotations

import torch
from torch import nn


class AdaptiveComponent(nn.Module):
    """Marker base class for parameters trained during base-frozen adaptation."""


def adaptive_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """Return every parameter owned by an adaptive component, once each.

    Parameters are collected by component ownership, not by module path.  A
    nested adaptive component therefore cannot be assigned twice to an
    optimizer or snapshot during transactional rollback.
    """
    found: dict[int, nn.Parameter] = {}
    for module in model.modules():
        if not isinstance(module, AdaptiveComponent):
            continue
        for parameter in module.parameters():
            found.setdefault(id(parameter), parameter)
    return tuple(found.values())


def adaptive_parameter_names(model: nn.Module) -> set[str]:
    """Return fully-qualified names of parameters owned by adaptive modules.

    Resolve ownership from :func:`adaptive_parameters` so nested adaptive
    components are reported once under their actual model name, rather than
    once per enclosing component.
    """
    adaptive_ids = {id(parameter) for parameter in adaptive_parameters(model)}
    return {
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in adaptive_ids
    }


def freeze_base_in_place(model: nn.Module) -> None:
    """Freeze every non-adaptive parameter and keep adaptive parameters live."""
    adaptive_ids = {id(parameter) for parameter in adaptive_parameters(model)}
    if not adaptive_ids:
        raise RuntimeError("freeze_base=True requires at least one adaptive component")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.requires_grad_(id(parameter) in adaptive_ids)


__all__ = [
    "AdaptiveComponent",
    "adaptive_parameter_names",
    "adaptive_parameters",
    "freeze_base_in_place",
]
