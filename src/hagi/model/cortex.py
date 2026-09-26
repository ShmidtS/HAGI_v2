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

Two switches extend the module for a recursive ternary parent without touching
the flat path.  ``mode`` says **where** the side channel writes: ``"dense"``
writes the whole stream (the flat model's role, and the default), ``"root"``
writes only the tree's all-equal mode, i.e. the diagonal the parent-preserving
lift fixes -- see ``docs/ARCHITECTURE_V2.md``.  ``gate`` says **how much**:
``"fixed"`` keeps the constant ``residual_scale`` multiplier, ``"gumbel"``
adds one annealed learnable scalar per directed edge, the mechanism credited
by the Cache-to-Cache ablation (arXiv:2510.03215; see
``docs/EXTERNAL_IDEAS.md``).
"""

from __future__ import annotations

import torch
from torch import nn

from hagi.config import CORTEX_GATES, CORTEX_MODES, PyramidalCortexConfig
from hagi.model.adaptive import AdaptiveComponent


# A gate logit at this value opens to ``sigmoid(-3) ~= 0.047`` at temperature
# 1, i.e. essentially closed.  It guards a zero-initialized link, so a fresh
# cortex injects nothing on top of the edge's own zero start.
_GATE_INIT_LOGIT = -3.0

# ``torch.rand`` can return exactly 0.0, whose log is -inf; the clamp keeps the
# logistic noise finite.
_NOISE_EPS = 1e-6


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


class _EdgeGate(nn.Module):
    """One learnable scalar gate on one directed edge.

    While training, logistic noise is added to the logit and the sum is divided
    by an annealed temperature, so the edge samples open-or-closed rather than
    sitting halfway; at evaluation the same logit passes through a plain
    sigmoid at the floor temperature, so the gate is deterministic and hardened
    toward binary.  The sampled form is the logistic-noise identity of
    Gumbel-sigmoid (Maddison et al., https://arxiv.org/abs/1611.01144); the
    per-edge gate on a cross-level additive channel is what Cache-to-Cache
    (https://arxiv.org/abs/2510.03215) credits for the gain -- its ablation
    reads overwrite 20.70 -> +residual 44.88 -> +gate 47.95, i.e. the channel
    alone is worth little and the gate is what makes injection safe.

    The logit starts at :data:`_GATE_INIT_LOGIT`, so the gate is essentially
    closed at initialization.  It guards a zero-initialized link, which makes
    the injection off twice over: a fresh cortex reproduces the parent's logits
    exactly, the property the flat path already relies on.

    ``keep_fp32`` marks the module for ``cast_model``: a bf16 scalar near zero
    cannot represent the tiny updates a nearly-closed gate receives, and would
    freeze at its initialization value for the whole run.
    """

    def __init__(self) -> None:
        super().__init__()
        self.keep_fp32 = True
        self.logit = nn.Parameter(torch.tensor(_GATE_INIT_LOGIT))

    def forward(self, temperature: torch.Tensor) -> torch.Tensor:
        """Return the gate value for this forward at ``temperature``.

        ``temperature`` is a scalar tensor while training (the annealed clock
        of the owning cortex) and the floor value at evaluation; both divide
        the logit the same way.
        """
        logit = self.logit
        if self.training:
            uniform = torch.rand_like(logit).clamp(_NOISE_EPS, 1.0 - _NOISE_EPS)
            # log(u) - log(1-u) is logistic, i.e. the difference of two
            # standard Gumbel draws.
            logit = logit + (torch.log(uniform) - torch.log1p(-uniform))
        return torch.sigmoid(logit / temperature)


class PyramidalCortex(AdaptiveComponent):
    """Stateless, directed cross-level residual memory.

    At each boundary ``l`` the module first publishes ``z_l = R_l(h_l)`` from
    the pre-residual boundary state. It then adds only summaries from source
    levels ``k < l`` for configured strides ``s = l - k``::

        h'_l = h_l + residual_scale * U_l(sum_{k<l} g_{k,l} P_{k,l} z_k)

    Every directed edge ``P_{k,l}`` has its own trainable matrix; adjacent and
    skip connections are not forced to share a target projector. ``g_{k,l}`` is
    the per-edge gate: absent (``gate="fixed"``) or an annealed Gumbel-sigmoid
    scalar (``gate="gumbel"``). ``U_l`` is linear, so gating the rank-width
    edge output before the sum is the same scalar placement on the injection
    term ``residual_scale`` multiplies, and ``gate="fixed"`` reproduces the
    pre-gate arithmetic bit for bit.

    ``mode`` selects the width ``R_l``/``U_l`` see. In ``"dense"`` mode it is
    the whole stream. In ``"root"`` mode the stream is a recursive ternary
    body: ``R_l`` reads the mean over the leaf axis (the all-equal mode, i.e.
    the diagonal the parent-preserving lift fixes) and ``U_l`` writes that
    single leaf-width vector back to **every** leaf identically. An equal term
    per leaf leaves the per-leaf ``BlockTreeNorm`` statistic unchanged and
    crosses the residual stream untouched, so the parent-preserving invariant
    survives; a dense term differs between leaves and breaks it at step 0. See
    ``docs/ARCHITECTURE_V2.md`` for the argument.

    All operations preserve leading ``[B, T]`` positions and mix only the
    bottleneck axis, so a one-token cached decode computes the same final
    position as a full sequence forward. Root mode adds no cross-leaf mixing:
    the reshape it uses is a view of the leaf axis, the mean reduces it, and
    the write broadcasts a single vector, so every position is still computed
    from its own state alone.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        cfg: PyramidalCortexConfig,
        *,
        n_leaves: int | None = None,
        leaf_hidden: int | None = None,
    ) -> None:
        """Build the side channel.

        Args:
            num_layers: unique block count the levels partition.
            hidden_size: full residual-stream width.
            cfg: cortex config.
            n_leaves: leaf count of a recursive tree body. Required when
                ``cfg.mode == "root"``; the caller resolves it from the merge
                geometry, because ``PyramidalCortexConfig`` describes the
                cortex only and never sees the tree.
            leaf_hidden: per-leaf width, paired with ``n_leaves``. Forbidden
                in ``"dense"`` mode, where the cortex has no leaf axis.
        """
        super().__init__()
        if not cfg.enabled:
            raise ValueError("PyramidalCortex requires model.cortex.enabled=True")
        if cfg.mode not in CORTEX_MODES:
            raise ValueError(
                f"cortex.mode must be one of {sorted(CORTEX_MODES)}, got {cfg.mode!r}"
            )
        if cfg.gate not in CORTEX_GATES:
            raise ValueError(
                f"cortex.gate must be one of {sorted(CORTEX_GATES)}, got {cfg.gate!r}"
            )
        if num_layers < 2 or not 2 <= cfg.num_levels <= num_layers:
            raise ValueError(
                "cortex.num_levels must be in [2, num_layers] for a directed cortex"
            )
        if cfg.rank < 1:
            raise ValueError(f"cortex.rank must be >= 1, got {cfg.rank}")
        if cfg.residual_scale <= 0.0 or not torch.isfinite(torch.tensor(cfg.residual_scale)):
            raise ValueError("cortex.residual_scale must be finite and > 0")

        # The leaf decomposition is the one ``BlockTreeNorm`` and
        # ``RecursiveBranchScale`` are built against, so root mode is defined
        # against it and against nothing else.
        if cfg.mode == "root":
            if n_leaves is None or leaf_hidden is None:
                raise ValueError(
                    "cortex.mode='root' requires the leaf decomposition "
                    "(n_leaves, leaf_hidden); a root-mode cortex in a flat "
                    "model has no trunk to write to"
                )
            if type(n_leaves) is not int or type(leaf_hidden) is not int:
                raise ValueError(
                    f"cortex leaf geometry must be integers, got {n_leaves!r}, {leaf_hidden!r}"
                )
            # A ternary tree has 3**depth leaves. Anything else means the
            # caller and the tree norms disagree about the decomposition, and
            # a silently wrong reshape would corrupt every leaf statistic.
            leaves, power = n_leaves, 0
            while leaves > 1 and leaves % 3 == 0:
                leaves //= 3
                power += 1
            if leaves != 1 or power < 1:
                raise ValueError(
                    f"cortex.mode='root' requires n_leaves to be a power of three "
                    f">= 3, got {n_leaves}"
                )
            if leaf_hidden < 1:
                raise ValueError(f"cortex leaf_hidden must be >= 1, got {leaf_hidden}")
            if n_leaves * leaf_hidden != hidden_size:
                raise ValueError(
                    f"cortex leaf geometry {n_leaves} x {leaf_hidden} does not tile "
                    f"hidden_size {hidden_size}"
                )
            width = leaf_hidden
        elif n_leaves is not None or leaf_hidden is not None:
            raise ValueError(
                "cortex.mode='dense' takes no leaf geometry: a dense cortex writes "
                "the whole stream, which is exactly what breaks leaf equality in a "
                "ternary tree"
            )
        else:
            width = hidden_size
        if cfg.rank > width:
            raise ValueError(
                f"cortex.rank ({cfg.rank}) exceeds the cortex write width "
                f"{width}; the bottleneck would be wider than the trunk it feeds"
            )
        if cfg.gate == "gumbel":
            if cfg.gate_temp_start <= 0.0 or cfg.gate_temp_floor <= 0.0:
                raise ValueError("cortex gate temperatures must be > 0")
            if cfg.gate_temp_start < cfg.gate_temp_floor:
                raise ValueError(
                    "cortex.gate_temp_start must be >= gate_temp_floor "
                    f"({cfg.gate_temp_start} < {cfg.gate_temp_floor})"
                )
            if type(cfg.gate_temp_steps) is not int or cfg.gate_temp_steps < 1:
                raise ValueError(
                    f"cortex.gate_temp_steps must be a positive integer, got {cfg.gate_temp_steps!r}"
                )

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
        self.mode = cfg.mode
        self.gate = cfg.gate
        self.n_leaves = None if n_leaves is None else int(n_leaves)
        self.leaf_hidden = None if leaf_hidden is None else int(leaf_hidden)
        self.width = int(width)
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
                nn.Linear(self.width, self.rank, bias=False)
                for _ in sorted(used_sources)
            ]
        )
        self.up = nn.ModuleList(
            [
                nn.Linear(self.rank, self.width, bias=False)
                for _ in sorted(used_targets)
            ]
        )
        # One independent matrix per directed edge P_{source,target}.  A
        # shared target matrix would collapse adjacent and skip paths into a
        # single sum and lose the asymmetric highway contract.
        self.links = nn.ModuleList([_DirectedLink(self.rank) for _ in edges])
        self.edges = tuple(edges)
        # One scalar gate per directed edge, or no gate at all.  ``None`` keeps
        # the module tree of the fixed path byte-identical to the pre-gate
        # module tree: no extra entry, no extra state_dict key.
        self.gates: nn.ModuleList | None = (
            nn.ModuleList([_EdgeGate() for _ in edges]) if cfg.gate == "gumbel" else None
        )
        if self.gates is not None:
            # The annealing clock counts cortex passes.  It is a persistent
            # buffer so a resumed run continues the schedule instead of
            # restarting the anneal, and it is an integer, so ``cast_model``
            # leaves it alone.
            self.register_buffer("gate_step", torch.zeros((), dtype=torch.int64), persistent=True)
            # The clock is advanced by the terminal boundary, which is the only
            # boundary reached exactly once per pass.  That placement is sound
            # only because the terminal level always has a source.
            if not self._sources_by_level[self.num_levels - 1]:
                raise RuntimeError(
                    "cortex terminal level has no sources; the gate clock would "
                    "never advance"
                )
        # ``cast_model`` reads ``keep_fp32`` on the module that directly owns
        # a parameter. Mark every cortex weight module, not the parent, so a
        # bf16 base never rounds away the small online master updates. Physical
        # 4-bit/8-bit storage remains a separate, not-yet-implemented codec.
        for module in (*self.down, *self.up, *self.links, *(self.gates or ())):
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

    def _root_mode(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse a tree stream to its all-equal (root) mode.

        The leaf axis is the decomposition :class:`BlockTreeNorm` and
        :class:`RecursiveBranchScale` are built against, so the mean over it is
        the diagonal direction the parent-preserving lift fixes -- the trunk.
        Only the trailing axes are touched: the reduction is per ``(B, T)``
        position and never mixes tokens, so the decode parity the dense path
        promises is preserved.
        """
        leaves = x.reshape(*x.shape[:-1], self.n_leaves, self.leaf_hidden)
        return leaves.mean(dim=-2)

    def _write_root(self, root: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """Expand a root-mode term to the full stream, identically per leaf.

        The identical broadcast is the whole invariant: an equal addend in
        every leaf leaves each leaf's own ``BlockTreeNorm`` statistic
        unchanged, so the term rides the residual stream untouched and the
        leaves stay equal.  ``root`` is a single leaf-width vector per position
        and the expand has no leaf axis to vary over, so leaf dependence is
        structurally impossible; ``scripts/root_cortex_check.py`` measures the
        equality on a built module instead of trusting this docstring.
        """
        expanded = root.unsqueeze(-2).expand(
            *root.shape[:-1], self.n_leaves, self.leaf_hidden
        )
        return expanded.reshape(shape)

    def _gate_temperature(self) -> torch.Tensor:
        """Return the annealed Gumbel-sigmoid temperature for this pass.

        The schedule is a linear ramp from ``gate_temp_start`` to
        ``gate_temp_floor`` over ``gate_temp_steps`` cortex passes, clamped at
        the floor.  It is counted in passes, not optimizer steps: one pass of
        the block stack advances the clock once, so a ``loop_depth = L`` model
        anneals ``L`` times faster in step terms.  Everything stays on device
        in fp32, so no host sync enters a level boundary.
        """
        span = float(self.cfg.gate_temp_steps)
        progress = self.gate_step.to(dtype=torch.float32).clamp(max=span) / span
        start = float(self.cfg.gate_temp_start)
        floor = float(self.cfg.gate_temp_floor)
        return start + (floor - start) * progress

    def _advance_gate_clock(self, level: int) -> None:
        """Tick the annealing schedule once per model pass.

        The terminal boundary is the only boundary reached exactly once per
        pass, so the clock counts passes.  This is the single piece of state
        the side channel mutates, and it is training progress rather than token
        state: no summary, gate value or noise draw survives the forward, so
        incremental decode stays equivalent to a full sequence pass.
        """
        if self.gates is not None and self.training and level == self.num_levels - 1:
            self.gate_step.add_(1)

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
            # previous dependency when composing multiple strides.  Root mode
            # publishes the trunk only: a leaf-dependent component never
            # leaves the level it belongs to.
            down = self.down[self._down_index[level]]
            source = self._root_mode(x) if self.mode == "root" else x
            states[level] = down(source.to(dtype=down.weight.dtype))

        sources = self._sources_by_level[level]
        if not sources:
            return x

        missing = [source for source in sources if states[source] is None]
        if missing:
            raise RuntimeError(
                f"cortex level {level} is missing summaries from levels {missing}"
            )
        gates = self.gates
        temperature = self._gate_temperature() if gates is not None else None
        edges: list[torch.Tensor] = []
        for source in sources:
            edge_index = self._edge_index[(source, level)]
            link = self.links[edge_index]
            state = states[source]
            assert state is not None
            passed = link(state.to(dtype=link.weight.dtype))
            if gates is not None:
                assert temperature is not None
                # The gate scales the rank-width edge output before the sum.
                # ``up`` is linear, so this is the same scalar placement on the
                # injection term as a gate on the reconstructed residual, and
                # it costs the narrow multiply instead of the wide one.
                passed = passed * gates[edge_index](temperature).to(dtype=passed.dtype)
            edges.append(passed)
        context = (
            edges[0]
            if len(edges) == 1
            else torch.stack(edges, dim=0).sum(dim=0)
        )
        target_index = self._target_index[level]
        up = self.up[target_index]
        delta = up(context.to(dtype=up.weight.dtype)).to(dtype=x.dtype)
        if self.mode == "root":
            # Broadcast the leaf-width reconstruction to every leaf before the
            # add, so the term is equal across leaves by construction.
            delta = self._write_root(delta, x.shape)
        self._advance_gate_clock(level)
        return x + self.residual_scale * delta


__all__ = ["PyramidalCortex", "level_boundaries"]
