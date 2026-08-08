"""grow.py — Expert concatenation + Cross-Expert Mixer for HAGI growing experiment.

Constructs a wide HAGI from N identically-architectured small experts:

    H_new = N * H                           (hidden dimension)
    L unchanged                             (same depth)
    head_dim unchanged                      (RoPE unchanged, theta unchanged)
    num_heads_new  = N * num_heads          (proportionally more heads)
    num_kv_heads_new = N * num_kv_heads     (GQA ratio preserved)
    intermediate_new = N * intermediate     (FFN width scales with H)

Block-diagonal property at step 0 (before any integration training):
    F_big(x) is equivalent to each expert running in its own H-subspace.
    The LM head output is the average of all expert logits (1/sqrt(N) scale).

After concat_experts(), CrossExpertMixer modules can be attached to every
Block via attach_cross_mixers(). These are tiny bottleneck residual layers
that learn inter-subspace communication.

Reference: Branch-Train-MiX (Sukhbaatar et al. 2024, arXiv:2403.07816),
but with full H concatenation instead of FFN-only MoE extraction.
"""
from __future__ import annotations

import copy
import math
from typing import Literal

import torch
from torch import nn

from hagi.config import Config, ffn_width
from hagi.model.model import HAGI
from hagi.model.norms import RMSNorm


# ---------------------------------------------------------------------------
# Cross-Expert Mixer
# ---------------------------------------------------------------------------

class CrossExpertMixer(nn.Module):
    """Lightweight residual bottleneck for cross-subspace communication.

    Inserted after every Block's FFN. The up-projection is zero-initialized,
    so the module is a strict identity at step 0 — it cannot corrupt the
    block-diagonal property before the first integration step.

        h' = h + up(silu(down(norm(h))))

    Bottleneck size defaults to max(64, H_new // 8).  At H_new=512 that is
    64 parameters connecting 512 channels — cheap to train, sufficient to
    propagate cross-expert routing signals.

    Args:
        hidden_size: H_new (total hidden size of the grown model).
        bottleneck_dim: inner width; None → max(64, H_new // 8).
    """

    def __init__(self, hidden_size: int, bottleneck_dim: int | None = None) -> None:
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = max(64, hidden_size // 8)
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.norm = RMSNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        # near-zero down, zero up → identity at init
        nn.init.normal_(self.down.weight, std=0.02 / math.sqrt(hidden_size))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(torch.nn.functional.silu(self.down(self.norm(x))))

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, bottleneck_dim={self.bottleneck_dim}"


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def make_grown_config(expert_cfg: Config, n_experts: int) -> Config:
    """Build a Config for the N-expert concatenated model.

    Scales H, heads, and FFN intermediate by N. Everything else (depth,
    head_dim, rope_theta, sliding window ratios, vocab, etc.) is unchanged.

    Args:
        expert_cfg: config of one expert (all must be identical).
        n_experts: how many experts to fuse.

    Returns:
        New Config valid for instantiation with HAGI().
    """
    if n_experts < 2:
        raise ValueError("need at least 2 experts to grow")
    m = expert_cfg.model
    a = m.attention
    N = n_experts

    cfg = copy.deepcopy(expert_cfg)
    cfg.model.hidden_size = m.hidden_size * N
    cfg.model.attention.num_query_heads = a.num_query_heads * N
    cfg.model.attention.num_kv_heads = a.num_kv_heads * N
    cfg.model.ffn.intermediate_size = ffn_width(m) * N
    cfg.model.ffn.multiple_of = 1      # already exact multiple, skip rounding
    cfg.model.target_params = 0        # disable auto_configure
    cfg.model.init_orthogonal = False  # block-diagonal init overrides this

    # Sanity check: the fundamental GQA constraint must hold after scaling.
    new_h = cfg.model.hidden_size
    new_nq = cfg.model.attention.num_query_heads
    assert new_nq * a.head_dim == new_h, (
        f"head geometry broken after scaling: "
        f"{new_nq} * {a.head_dim} = {new_nq * a.head_dim} ≠ {new_h}"
    )
    return cfg


# ---------------------------------------------------------------------------
# Tensor fill helpers (no torch.block_diag because it only takes square blocks)
# ---------------------------------------------------------------------------

def _block_diagonal(weights: list[torch.Tensor]) -> torch.Tensor:
    """Stack 2-D tensors into a block-diagonal matrix.

    Given [out_0, in_0], [out_1, in_1], ... returns a matrix of shape
    [sum_out, sum_in] with each block on the main diagonal and zeros elsewhere.
    """
    total_out = sum(w.shape[0] for w in weights)
    total_in = sum(w.shape[1] for w in weights)
    result = torch.zeros(total_out, total_in, dtype=weights[0].dtype)
    out_off = 0
    in_off = 0
    for w in weights:
        out_sz, in_sz = w.shape
        result[out_off: out_off + out_sz, in_off: in_off + in_sz].copy_(w)
        out_off += out_sz
        in_off += in_sz
    return result


def _concat(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    return torch.cat(tensors, dim=dim)


def _mean(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(tensors, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Main concatenation
# ---------------------------------------------------------------------------

def concat_experts(experts: list[HAGI]) -> HAGI:
    """Concatenate N experts into one wide model with block-diagonal init.

    All experts must have the same Config (shape-wise). The resulting model
    satisfies:

        logits_big ≈ average(logits_k)   at step 0 (before training)

    because the LM head is initialized with logit_scale / sqrt(N), and the
    tied embedding projects [h_A, h_B, ...] against [emb_A, emb_B, ...] to
    give the sum of expert dot-products, which is divided by sqrt(N).

    No CrossExpertMixers are attached. Call attach_cross_mixers() afterwards.

    Args:
        experts: list of trained HAGI models, all with identical architecture.

    Returns:
        New HAGI instance with H_new = N * H.
    """
    if len(experts) < 2:
        raise ValueError("need at least 2 experts")
    ref_cfg = experts[0].cfg
    for i, e in enumerate(experts[1:], 1):
        _validate_same_arch(ref_cfg, e.cfg, expert_idx=i)

    N = len(experts)
    new_cfg = make_grown_config(ref_cfg, N)
    big = HAGI(new_cfg)

    with torch.no_grad():
        _fill_encoder(big, experts)
        _fill_blocks(big, experts)
        _fill_out_norm(big, experts)
        _fill_head(big, experts, N)

    return big


def _validate_same_arch(ref: Config, other: Config, expert_idx: int) -> None:
    m_ref, m_oth = ref.model, other.model
    for attr in ("hidden_size", "num_layers", "vocab_size"):
        v_ref = getattr(m_ref, attr)
        v_oth = getattr(m_oth, attr)
        if v_ref != v_oth:
            raise ValueError(f"expert[{expert_idx}] {attr}={v_oth} ≠ ref {v_ref}")
    for attr in ("num_query_heads", "num_kv_heads", "head_dim"):
        v_ref = getattr(m_ref.attention, attr)
        v_oth = getattr(m_oth.attention, attr)
        if v_ref != v_oth:
            raise ValueError(f"expert[{expert_idx}] attention.{attr}={v_oth} ≠ ref {v_ref}")


def _fill_encoder(big: HAGI, experts: list[HAGI]) -> None:
    # embedding: [V, H] → [V, H_new] by column-concatenation.
    # Each expert's tokens map to their own H-slice so the body blocks
    # (which are block-diagonal in the H dimension) see exactly the same
    # representation they saw during independent training.
    big.encoder.embedding.weight.copy_(
        _concat([e.encoder.embedding.weight for e in experts], dim=1)
    )

    # depthwise conv weight: [H, 1, K] → [H_new, 1, K] along channels
    if big.encoder.conv is not None and experts[0].encoder.conv is not None:
        big.encoder.conv.weight.copy_(
            _concat([e.encoder.conv.weight for e in experts], dim=0)
        )
        big.encoder.conv.bias.copy_(
            _concat([e.encoder.conv.bias for e in experts], dim=0)
        )
        if big.encoder.norm is not None and experts[0].encoder.norm is not None:
            big.encoder.norm.weight.copy_(
                _concat([e.encoder.norm.weight for e in experts], dim=0)
            )


def _fill_blocks(big: HAGI, experts: list[HAGI]) -> None:
    for layer_idx, big_block in enumerate(big.blocks):
        exp_blocks = [e.blocks[layer_idx] for e in experts]

        # Pre-norm (attn branch): [H] → [H_new]
        big_block.attn.attn_norm.weight.copy_(
            _concat([b.attn.attn_norm.weight for b in exp_blocks], dim=0)
        )

        # Fused QKV: [n_q_out + 2*n_kv_out, H] → block_diagonal [same*N, H_new]
        # Each expert's QKV sub-block sits on the diagonal, attending only to its
        # own H-subspace of the residual stream (the off-diagonal blocks are zero).
        big_block.attn.qkv_proj.weight.copy_(
            _block_diagonal([b.attn.qkv_proj.weight for b in exp_blocks])
        )

        # out_proj: [H_q, H] → block_diagonal [H_q*N, H_new] = [H_new, H_new]
        big_block.attn.out_proj.weight.copy_(
            _block_diagonal([b.attn.out_proj.weight for b in exp_blocks])
        )

        # QK-norm: [head_dim] → same (head_dim is unchanged); average gains
        if big_block.attn.q_norm is not None:
            big_block.attn.q_norm.weight.copy_(
                _mean([b.attn.q_norm.weight for b in exp_blocks])
            )
            big_block.attn.k_norm.weight.copy_(
                _mean([b.attn.k_norm.weight for b in exp_blocks])
            )

        # branch_scale (attention): scalar residual cap; average
        with torch.no_grad():
            vals = torch.stack([b.attn.branch_scale.scale.data for b in exp_blocks])
            big_block.attn.branch_scale.scale.copy_(vals.mean())

        # Pre-norm (FFN branch): [H] → [H_new]
        big_block.mixer.norm.weight.copy_(
            _concat([b.mixer.norm.weight for b in exp_blocks], dim=0)
        )

        # SwiGLU: gate/up/down all block-diagonal
        big_block.mixer.mixer.gate.weight.copy_(
            _block_diagonal([b.mixer.mixer.gate.weight for b in exp_blocks])
        )
        big_block.mixer.mixer.up.weight.copy_(
            _block_diagonal([b.mixer.mixer.up.weight for b in exp_blocks])
        )
        big_block.mixer.mixer.down.weight.copy_(
            _block_diagonal([b.mixer.mixer.down.weight for b in exp_blocks])
        )

        # branch_scale (FFN): average
        with torch.no_grad():
            vals = torch.stack([b.mixer.mixer.branch_scale.scale.data for b in exp_blocks])
            big_block.mixer.mixer.branch_scale.scale.copy_(vals.mean())


def _fill_out_norm(big: HAGI, experts: list[HAGI]) -> None:
    big.out_norm.weight.copy_(
        _concat([e.out_norm.weight for e in experts], dim=0)
    )


def _fill_head(big: HAGI, experts: list[HAGI], N: int) -> None:
    # The tied head projection is the embedding, already filled above.
    # For untied heads, fill the projection weight the same way.
    if not big.head.is_tied and experts[0].head.projection is not None:
        big.head.projection.weight.copy_(
            _concat([e.head.projection.weight for e in experts], dim=1)
        )

    # logit_scale: the big head's dot product sums N contributions
    # (each of roughly equal magnitude). To keep the initial output
    # distribution stable (same entropy as one expert), divide by sqrt(N).
    ref_scale = float(experts[0].head.logit_scale)
    big.head.logit_scale.copy_(torch.tensor(ref_scale / math.sqrt(N)))


# ---------------------------------------------------------------------------
# Attach CrossExpertMixers
# ---------------------------------------------------------------------------

def attach_cross_mixers(
    model: HAGI,
    bottleneck_dim: int | None = None,
) -> nn.ModuleList:
    """Attach a CrossExpertMixer after the FFN of every Block.

    Requires that Block.forward checks self.cross_mixer (the version in
    the modified block.py). Mixers start as identity (zero up-projection).

    Args:
        model: grown HAGI model.
        bottleneck_dim: bottleneck width per mixer; None → max(64, H//8).

    Returns:
        The ModuleList of CrossExpertMixers (also registered as model.cross_mixers).
    """
    H_new = model.cfg.model.hidden_size
    mixers = nn.ModuleList(
        [CrossExpertMixer(H_new, bottleneck_dim) for _ in model.blocks]
    )
    # Register as a submodule so state_dict() and .to(device) include them.
    model.cross_mixers = mixers

    # Attach one mixer per block — Block.forward() checks self.cross_mixer.
    for block, mixer in zip(model.blocks, mixers):
        block.cross_mixer = mixer

    return mixers


# ---------------------------------------------------------------------------
# Training-mode helpers
# ---------------------------------------------------------------------------

def set_integration_mode(
    model: HAGI,
    mode: Literal["mixer_only", "mixer_plus_slow_body", "full"],
) -> None:
    """Freeze / unfreeze parameters for integration training.

    mixer_only          — only CrossExpertMixers train; the entire body is frozen.
    mixer_plus_slow_body — everything trains; body LR should be scaled down
                           (e.g. ×0.1) in the optimizer to avoid overwriting
                           the specialist knowledge.
    full                — everything trains at full LR.
    """
    has_mixers = hasattr(model, "cross_mixers") and model.cross_mixers is not None
    for name, param in model.named_parameters():
        is_mixer = has_mixers and "cross_mixers" in name
        if mode == "mixer_only":
            param.requires_grad_(is_mixer)
        else:
            param.requires_grad_(True)


# ---------------------------------------------------------------------------
# Numerical verification
# ---------------------------------------------------------------------------

@torch.no_grad()
def verify_block_diagonal(
    model: HAGI,
    experts: list[HAGI],
    tol: float = 1e-3,
    n_tokens: int = 32,
) -> dict[str, float]:
    """Check that the body subspace-k output matches expert-k's output.

    Runs a random batch through both the big model and each expert, then
    compares hidden[:, :, k*H:(k+1)*H] with expert_k(hidden).

    Works only when no CrossExpertMixers have been trained (they break the
    exact block-diagonal property by design). If mixers are attached, detach
    them first or use a freshly merged model.

    Returns:
        Dict expert_k → max absolute deviation over the batch.
    """
    H = experts[0].cfg.model.hidden_size
    device = next(model.parameters()).device
    vocab = experts[0].cfg.model.vocab_size

    batch = torch.randint(0, vocab, (2, n_tokens), device=device)

    model.eval()
    for e in experts:
        e.eval().to(device)

    out_big = model(batch, return_logits=False).hidden   # [B, T, H_new]
    deviations: dict[str, float] = {}
    for k, expert in enumerate(experts):
        out_k = expert(batch, return_logits=False).hidden  # [B, T, H]
        big_slice = out_big[..., k * H: (k + 1) * H]
        dev = (big_slice - out_k).abs().max().item()
        deviations[f"expert_{k}"] = dev
        if dev > tol:
            import warnings
            warnings.warn(
                f"expert_{k}: block-diagonal deviation {dev:.4e} > tol {tol}. "
                "This is expected if CrossExpertMixers are non-zero.",
                stacklevel=2,
            )
    return deviations
