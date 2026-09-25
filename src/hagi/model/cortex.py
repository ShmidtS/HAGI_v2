"""Directed same-token memory between contiguous layer levels.

The module adapts the Perceiver IO latent-bottleneck idea (Jaegle et al.,
2021, https://arxiv.org/abs/2107.14795) to HAGI's sequential block stack.  It
compresses each boundary state, lets only earlier levels feed a later target,
and reconstructs a gated residual.  It deliberately does **not** keep state
across tokens or generation steps, so incremental decode remains equivalent to
a full forward.

The zero-initialized directed links and additive residual follow the verified
local HAGI/Qwen identity-preserving adapter contract in
``scripts/qwen_pyramid.py``.  Links wake first; their feature projections can
receive gradients on the following optimizer step.
"""

from __future__ import annotations

import torch
from torch import nn

from hagi.config import PyramidalCortexConfig
from hagi.model.adaptive import AdaptiveComponent


def level_boundaries(num_layers: int, num_levels: int) -> tuple[int, ...]:
    """Return zero-based final block indices of balanced contiguous levels.

    Example: ``(24, 4) -> (5, 11, 17, 23)``.  Remainder blocks are assigned to
    the earliest levels so the mapping is deterministic and covers every block.
    """
    if type(num_layers) is not int or num_layers < 1:
        raise ValueError(f"num_layers must be a positive integer, got {num_layers!r}")
    if type(num_levels) is not int or num_levels < 1:
        raise ValueError(f"num_levels must be a positive integer, got {num_levels!r}")
    if num_levels > num_layers:
        raise ValueError(
            f"num_levels ({num_levels}) must not exceed num_layers ({num_layers})"
        )

    size, remainder = divmod(num_layers, num_levels)
    cursor = 0
    boundaries: list[int] = []
    for level in range(num_levels):
        cursor += size + (1 if level < remainder else 0)
        boundaries.append(cursor - 1)
    return tuple(boundaries)


class _DirectedLink(nn.Module):
    """A square bottleneck with the only zero-initialized parameter."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(rank, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight)


class PyramidalCortex(AdaptiveComponent):
    """Stateless, directed cross-level residual memory.

    At each boundary ``l`` the module first publishes ``z_l = R_l(h_l)`` from
    the pre-residual boundary state. It then adds only summaries from source
    levels ``k < l`` for configured strides ``s = l - k``::

        h'_l = h_l + residual_scale * U_l(sum_{k<l} P_{k,l} z_k)

    Every directed edge ``P_{k,l}`` has its own trainable matrix; adjacent and
    skip connections are not forced to share a target projector.

    All operations preserve leading ``[B, T]`` positions and mix only the
    bottleneck axis, so a one-token cached decode computes the same final
    position as a full sequence forward.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        cfg: PyramidalCortexConfig,
    ) -> None:
        super().__init__()
        if not cfg.enabled:
            raise ValueError("PyramidalCortex requires model.cortex.enabled=True")
        if num_layers < 2 or not 2 <= cfg.num_levels <= num_layers:
            raise ValueError(
                "cortex.num_levels must be in [2, num_layers] for a directed cortex"
            )
        if cfg.rank < 1:
            raise ValueError(f"cortex.rank must be >= 1, got {cfg.rank}")
        if cfg.residual_scale <= 0.0 or not torch.isfinite(torch.tensor(cfg.residual_scale)):
            raise ValueError("cortex.residual_scale must be finite and > 0")

        strides = tuple(cfg.link_strides)
        if not strides or any(type(stride) is not int or stride < 1 for stride in strides):
            raise ValueError("cortex.link_strides must contain positive integers")
        if len(set(strides)) != len(strides):
            raise ValueError("cortex.link_strides entries must be unique")
        if tuple(sorted(strides)) != strides:
            raise ValueError("cortex.link_strides must be sorted ascending")
        if strides[-1] >= cfg.num_levels:
            raise ValueError(
                "cortex.link_strides must be smaller than num_levels"
            )

        self.cfg = cfg
        self.num_layers = int(num_layers)
        self.num_levels = int(cfg.num_levels)
        self.hidden_size = int(hidden_size)
        self.rank = int(cfg.rank)
        self.link_strides = strides
        self.residual_scale = float(cfg.residual_scale)
        self.boundaries = level_boundaries(num_layers, self.num_levels)

        self._sources_by_level: tuple[tuple[int, ...], ...] = tuple(
            tuple(
                sorted(
                    level - stride
                    for stride in strides
                    if level - stride >= 0
                )
            )
            for level in range(self.num_levels)
        )
        used_sources = {
            source
            for sources in self._sources_by_level
            for source in sources
        }
        used_targets = {
            level for level, sources in enumerate(self._sources_by_level) if sources
        }
        edges = [
            (source, target)
            for target, sources in enumerate(self._sources_by_level)
            for source in sources
        ]

        self.down = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.rank, bias=False)
                for _ in sorted(used_sources)
            ]
        )
        self.up = nn.ModuleList(
            [
                nn.Linear(self.rank, self.hidden_size, bias=False)
                for _ in sorted(used_targets)
            ]
        )
        # One independent matrix per directed edge P_{source,target}.  A
        # shared target matrix would collapse adjacent and skip paths into a
        # single sum and lose the asymmetric highway contract.
        self.links = nn.ModuleList([_DirectedLink(self.rank) for _ in edges])
        self.edges = tuple(edges)
        # ``cast_model`` reads ``keep_fp32`` on the module that directly owns
        # a parameter. Mark every cortex weight module, not the parent, so a
        # bf16 base never rounds away the small online master updates. Physical
        # 4-bit/8-bit storage remains a separate, not-yet-implemented codec.
        for module in (*self.down, *self.up, *self.links):
            module.keep_fp32 = True
        self._down_index = {level: index for index, level in enumerate(sorted(used_sources))}
        self._target_index = {level: index for index, level in enumerate(sorted(used_targets))}
        self._edge_index = {edge: index for index, edge in enumerate(self.edges)}

        layer_to_level: list[int] = []
        previous_boundary = -1
        for level, boundary in enumerate(self.boundaries):
            layer_to_level.extend(
                [level] * (boundary - previous_boundary)
            )
            previous_boundary = boundary
        if len(layer_to_level) != num_layers:
            raise RuntimeError(
                f"cortex level map covers {len(layer_to_level)} layers, expected {num_layers}"
            )
        self.layer_to_level = tuple(layer_to_level)
        self.boundary_to_level = {
            boundary: level for level, boundary in enumerate(self.boundaries)
        }

    def start_sequence(self) -> list[torch.Tensor | None]:
        """Create a fresh same-forward state container.

        This method intentionally does not mutate persistent module state.  It
        gives gradient checkpointing a normal, functional boundary for one
        layer pass and makes ``loop_depth`` passes independent by construction.
        """
        return [None] * self.num_levels

    def is_boundary(self, layer_index: int) -> bool:
        """Whether ``layer_index`` completes a pyramid level."""
        return layer_index in self.boundary_to_level

    def level_for_layer(self, layer_index: int) -> int:
        """Return the contiguous level containing ``layer_index``."""
        if not 0 <= layer_index < self.num_layers:
            raise ValueError(f"layer {layer_index} is outside cortex geometry")
        return self.layer_to_level[layer_index]

    def level_for_boundary(self, layer_index: int) -> int:
        """Return the level completed by ``layer_index``."""
        try:
            return self.boundary_to_level[layer_index]
        except KeyError as exc:
            raise ValueError(f"layer {layer_index} is not a cortex boundary") from exc

    def apply_boundary(
        self,
        x: torch.Tensor,
        level: int,
        states: list[torch.Tensor | None],
    ) -> torch.Tensor:
        """Publish a level and add its directed incoming cross-level residual."""
        if type(level) is not int or not 0 <= level < self.num_levels:
            raise ValueError(f"invalid cortex level {level!r}")
        if len(states) != self.num_levels:
            raise ValueError(
                f"expected {self.num_levels} states, got {len(states)}"
            )
        if x.ndim < 2 or x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"cortex input must end in hidden size {self.hidden_size}, got {tuple(x.shape)}"
            )

        if level in self._down_index:
            # Publish the raw boundary, not the post-incoming residual.  This
            # prevents a skip edge from accidentally re-adding a target's
            # previous dependency when composing multiple strides.
            down = self.down[self._down_index[level]]
            states[level] = down(x.to(dtype=down.weight.dtype))

        sources = self._sources_by_level[level]
        if not sources:
            return x

        missing = [source for source in sources if states[source] is None]
        if missing:
            raise RuntimeError(
                f"cortex level {level} is missing summaries from levels {missing}"
            )
        edges: list[torch.Tensor] = []
        for source in sources:
            edge_index = self._edge_index[(source, level)]
            link = self.links[edge_index]
            state = states[source]
            assert state is not None
            edges.append(link(state.to(dtype=link.weight.dtype)))
        context = (
            edges[0]
            if len(edges) == 1
            else torch.stack(edges, dim=0).sum(dim=0)
        )
        target_index = self._target_index[level]
        up = self.up[target_index]
        delta = up(context.to(dtype=up.weight.dtype)).to(dtype=x.dtype)
        return x + self.residual_scale * delta


__all__ = ["PyramidalCortex", "level_boundaries"]
