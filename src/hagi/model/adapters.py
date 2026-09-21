"""Opt-in residual adapters for HAGI: pyramid contour and TTT-LoRA.

**Attachment point.** Each adapter attaches as an *additive residual delta*
after the original :class:`~hagi.model.block.Block` forward -- i.e. after the
``x + attention(x)`` and ``x + mixer(x)`` updates have both been applied. The
Block computes::

    x_post_attn = x + self.attn(x, positions, mask)
    x_post_mixer = x_post_attn + self.mixer(x_post_attn)
    adapter = getattr(self, "adapters", None)
    if adapter is not None:
        x_post_mixer = x_post_mixer + adapter(x_post_attn, x_post_mixer)

The adapter receives the post-attention residual that the frozen mixer sees,
and returns a zero-initialized delta added to the post-mixer stream. The
default path (``adapters is None``) is a pure no-op: no extra parameters, no
extra computation, and no change to the output.

**Default-path invariant.** When ``cfg.model.adapters.enabled`` is ``False``
(the default), :class:`~hagi.model.model.HAGI` never creates an adapter, so
``Block.adapters`` is ``None`` and the original forward is reproduced
bit-for-bit -- numerically and shape-wise identical, including dtype and
device. Zero-init ``scale`` (pyramid) and ``lora_B`` (LoRA) also make an
enabled-but-untrained adapter an exact zero delta.

The adapters do not change the frozen base tensors, tokenizer, QKV layout, or
LM head, and do not rewrite frozen GGUF/safetensors artifacts.

Reference-first: the LoRA form follows the standard PEFT layout
``delta(x) = scaling * (x @ A) @ B.T`` with ``alpha/r`` scaling and a
zero-init ``B``. The frozen orthonormal ``A`` follows the local
:mod:`scripts.qwen_ttt_lora` contract. The pyramid topology follows the
verified invariant in :mod:`scripts.qwen_pyramid`: shared frozen weights on a
common input, mean reduction, and additive residual.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from hagi.config import AdaptersConfig, PyramidAdapterConfig, TttLoraConfig


def _qr_orthonormal(
    rows: int,
    cols: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a reduced QR basis for the LoRA analysis projection.

    For the adapter contract ``rows >= cols``. The rows-orthonormal branch is
    retained as a defensive fallback for direct helper use, but adapter
    construction rejects ``rank > hidden_size`` before reaching it.

    The draw and the factorisation are pinned to CPU fp32 and the result is
    moved to the ambient device afterwards. Two reasons, both measured:

    * a CPU ``torch.Generator`` cannot feed a cuda ``torch.randn`` (it raises
      "Expected a 'cuda' device type for generator but found 'cpu'"), and
      callers pass one in for reproducibility;
    * ``torch.linalg.qr`` is not implemented for bf16 on CPU ("geqrf_cpu not
      implemented for 'BFloat16'"), so the ambient dtype has to be overridden
      for the factorisation, not just the device.

    The tensor is [rows, cols] -- 40K elements at real dimensions -- so the
    round trip costs nothing, while it is what lets a 27B model be constructed
    *directly on the GPU*. That is not a convenience: this host has 31.6 GiB of
    RAM, and building 22.4B params on the host needs 83.5 GiB fp32 (41.8 bf16),
    which pages into a crash. Precision is not lost: ``lora_A`` is a buffer,
    so the production ``cast_model`` already narrows it to bf16 anyway.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"QR dimensions must be positive, got {rows}x{cols}")
    # Capture the ambient target BEFORE pinning to cpu: inside that context the
    # ambient device is cpu again, so reading it after would silently leave the
    # basis on the host (measured: "cuda ctx: cpu").
    device = torch.get_default_device()
    dtype = torch.get_default_dtype()
    with torch.device("cpu"):
        if rows >= cols:
            q, _ = torch.linalg.qr(
                torch.randn(rows, cols, generator=generator, dtype=torch.float32))
        else:
            q, _ = torch.linalg.qr(
                torch.randn(rows, cols, generator=generator, dtype=torch.float32).T)
            q = q.T
    return q.to(device=device, dtype=dtype)


class PyramidAdapter(nn.Module):
    """Shared-weight multi-branch pyramid residual contour.

    ``cfg.levels`` is a tuple of per-level branch counts, e.g. ``(1, 2, 4)``:
    level 0 has 1 branch, level 1 has 2 branches, and level 2 has 4 branches.
    Every level runs its branches on the *same* input (the original mixer
    input), mean-reduces that level, and contributes to one additive delta.
    The shared mixer is evaluated under ``torch.no_grad()`` so the repeated
    pyramid recomputation cannot alter gradients of the frozen base mixer; the
    learnable scalar ``scale`` still receives the residual delta gradient.

    The only extra parameter is ``scale``, initialized to 0.0, so a freshly
    attached adapter is an exact zero delta. ``is_channel_weight`` is
    explicitly ``False`` on ``scale`` so the optimizer routes it to AdamW (a
    1D gain, not a 2D channel matrix that would go to Muon).
    """

    def __init__(self, cfg: PyramidAdapterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.scale = nn.Parameter(torch.zeros(1))
        self.scale.is_channel_weight = False
        # Keep the frozen mixer outside the adapter module tree. Assigning it
        # through object.__setattr__ prevents its parameters from being
        # re-exposed by BlockAdapter.parameters() during freeze_base.
        self._branch: nn.Module | None = None

        if not isinstance(cfg.levels, (tuple, list)) or not cfg.levels:
            raise ValueError(
                "adapters.pyramid.levels must be a non-empty tuple/list of "
                "positive integers"
            )
        levels: list[int] = []
        for level in cfg.levels:
            if type(level) is not int or level < 1:
                raise ValueError(
                    "adapters.pyramid.levels entries must be positive "
                    f"integers, got {level!r}"
                )
            levels.append(level)
        self.levels = tuple(levels)
        self.n_branches = sum(self.levels)
        if not math.isfinite(cfg.residual_scale) or cfg.residual_scale < 0.0:
            raise ValueError(
                "adapters.pyramid.residual_scale must be finite and >= 0, "
                f"got {cfg.residual_scale!r}"
            )

    def attach(self, mixer: nn.Module) -> None:
        """Bind the frozen mixer without registering it as an adapter child."""
        object.__setattr__(self, "_branch", mixer)

    def forward(self, x: torch.Tensor, mixer_out: torch.Tensor) -> torch.Tensor:
        branch = self._branch
        if branch is None:
            raise RuntimeError(f"{type(self).__name__} was not attached to a mixer")

        # The shared base mixer is frozen for this repeated recomputation. The
        # adapter's scalar gate remains differentiable and receives the delta
        # gradient; the base mixer's parameters do not.
        with torch.no_grad():
            out: torch.Tensor | None = None
            for n in self.levels:
                if n == 1:
                    level_out = branch(x)
                else:
                    outs = torch.stack([branch(x) for _ in range(n)], dim=0)
                    level_out = outs.mean(dim=0)
                out = level_out if out is None else out + level_out
            assert out is not None

        gate = (self.cfg.residual_scale * self.scale).to(dtype=out.dtype)
        return out * gate


class TttLoraAdapter(nn.Module):
    """Low-rank LoRA residual delta: ``scaling * (x @ A) @ B.T``.

    ``A`` is a frozen orthonormal buffer (QR init); ``B:[H, r]`` is the only
    trainable parameter, zero-init so the delta is exactly zero at
    construction. The anchored-RLS contour that *fits* ``B`` lives in
    :mod:`scripts.qwen_ttt_lora`; this module is only the forward/inspect seam
    wired into the Block contract.

    Reference: PEFT LoRA layout ``lora_B @ lora_A`` with ``alpha/r`` scaling and
    a zero-init ``B``. Here the projection goes through a frozen orthonormal
    basis ``A[Fin, r]`` (so ``phi = x @ A`` is the LoRA feature), matching
    :mod:`scripts.qwen_ttt_lora.LowRankLoRA`, which uses ``A:[Fin,r]``,
    ``B:[Fout,r]``, ``delta = scaling * (phi @ B.T)``.
    """

    def __init__(
        self,
        hidden_size: int,
        cfg: TttLoraConfig,
        init_seed: int = 0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if type(cfg.rank) is not int or cfg.rank < 1:
            raise ValueError(f"Lora r must be an integer >= 1, got {cfg.rank!r}")
        if cfg.rank > hidden_size:
            raise ValueError(
                f"Lora r ({cfg.rank}) must be <= hidden_size ({hidden_size})"
            )
        if not math.isfinite(cfg.alpha) or cfg.alpha <= 0.0:
            raise ValueError(
                f"Lora alpha must be finite and > 0, got {cfg.alpha!r}"
            )
        if not math.isfinite(cfg.dropout) or not 0.0 <= cfg.dropout < 1.0:
            raise ValueError(
                f"Lora dropout must be in [0, 1), got {cfg.dropout!r}"
            )

        self.hidden_size = int(hidden_size)
        self.r = int(cfg.rank)
        self.scaling = float(cfg.alpha) / float(self.r)

        generator = torch.Generator().manual_seed(init_seed)
        # A[Fin, r] orthonormal (Fin >= r for the analysis projection).
        self.register_buffer(
            "lora_A",
            _qr_orthonormal(hidden_size, self.r, generator),
            persistent=False,
        )
        # B[Fout, r] with Fout == hidden_size. PEFT layout:
        # delta = scaling * (x @ A) @ B.T => [N, r] @ [r, H] = [N, H].
        self.lora_B = nn.Parameter(torch.zeros(hidden_size, self.r))
        self.lora_B.is_channel_weight = False
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else None

    def forward(self, x: torch.Tensor, mixer_out: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x_in = self.dropout(x)
        else:
            x_in = x
        # Match phi (and the matmul accumulator) to B's dtype so an
        # enabled-but-untrained LoRA emits an exact zero delta in the input
        # dtype, and an enabled adapter never alters the post-mixer dtype.
        x_in = x_in.to(dtype=self.lora_A.dtype)
        b_dtype = self.lora_B.dtype
        phi = x_in @ self.lora_A.to(dtype=x_in.dtype)
        delta = self.scaling * (
            phi.to(dtype=b_dtype) @ self.lora_B.to(dtype=b_dtype).T
        )
        return delta.to(dtype=x.dtype)


class BlockAdapter(nn.Module):
    """Composable residual adapter for :class:`~hagi.model.block.Block`.

    Holds an optional :class:`PyramidAdapter` and an optional
    :class:`TttLoraAdapter`. The combined delta is the sum of enabled
    sub-adapters. When ``cfg.enabled`` is ``False`` the Block holds no adapter
    at all and the default path is a pure no-op.

    The Block calls ``forward(mixer_input, mixer_output)`` and adds the
    returned delta to the post-mixer residual stream. ``mixer_input`` is the
    post-attention residual captured immediately before the frozen mixer runs;
    the adapter therefore receives exactly the same input as the base mixer.
    Adapters are attached post-construction by
    :class:`~hagi.model.model.HAGI`.
    """

    def __init__(self, cfg: AdaptersConfig, hidden_size: int) -> None:
        super().__init__()
        self._cfg = cfg
        self.hidden_size = hidden_size
        self.pyramid: PyramidAdapter | None = (
            PyramidAdapter(cfg.pyramid) if cfg.pyramid.enabled else None
        )
        self.ttt_lora: TttLoraAdapter | None = (
            TttLoraAdapter(hidden_size, cfg.ttt_lora)
            if cfg.ttt_lora.enabled
            else None
        )
        # Require exactly one contour when the master switch is on. Built
        # here (not only in validate_config) so a directly-constructed adapter
        # also fails fast instead of silently summing two zero-init deltas.
        if cfg.enabled and self.pyramid is not None and self.ttt_lora is not None:
            raise ValueError(
                "pyramid and ttt_lora adapters are currently mutually exclusive: "
                "enable only one of model.adapters.pyramid.enabled or "
                "model.adapters.ttt_lora.enabled"
            )
        if cfg.enabled and self.pyramid is None and self.ttt_lora is None:
            raise ValueError(
                "adapters.enabled is True but no contour is enabled: "
                "enable model.adapters.pyramid.enabled or "
                "model.adapters.ttt_lora.enabled"
            )

    @property
    def has_adapter(self) -> bool:
        """True when at least one configured sub-adapter is attached."""
        return self.pyramid is not None or self.ttt_lora is not None

    def attach(self, mixer: nn.Module) -> None:
        """Bind the frozen mixer to the pyramid contour."""
        if self.pyramid is not None:
            self.pyramid.attach(mixer)

    def forward(self, x: torch.Tensor, mixer_out: torch.Tensor) -> torch.Tensor:
        if not self.has_adapter:
            return torch.zeros_like(mixer_out)
        delta = torch.zeros_like(mixer_out)
        if self.pyramid is not None:
            delta = delta + self.pyramid(x, mixer_out)
        if self.ttt_lora is not None:
            delta = delta + self.ttt_lora(x, mixer_out)
        return delta
