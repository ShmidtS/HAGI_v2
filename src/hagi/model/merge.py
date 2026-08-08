"""Block-diagonal expert merge: train-many-small, merge-into-big.

The hypothesis (information-theoretic): a large model trained from scratch
spends most of its early compute discovering *specialized* representations
(per-domain features) that are then composed. If we instead train N small
experts to saturation on N different corpora, each expert's hidden space is
already a well-formed, domain-tuned code. Concatenating those spaces
block-diagonally into one wide model gives the big model a head start: it
inherits N pre-formed specialized subspaces instead of having to discover them.

The merged model is built so that at step 0 it is *exactly* N independent
experts:

    W_Q = diag(W_Q^A, W_Q^B, ..., W_Q^N)   (and likewise for KV, out, FFN)

so the off-diagonal blocks are zero and each block reproduces its expert's
forward pass bit-for-bit. The only new parameters are a small number of
cross-block ``CrossMixer`` layers (identity-init, gain 0), which are the
connections that let the blocks interact. Short joint training then teaches
the blocks to compose — the mixer gain rises from 0 as the blocks learn to
exchange information.

Why block-diagonal rather than a dense re-init: a dense big model would
scramble the experts' learned geometry at step 0 (every weight is a linear
combination of unrelated experts), destroying the very representations we
merged to preserve. Block-diagonal preserves them exactly and adds only the
cheap mixer connections.

The mixer is placed *after* the expert stack, on the residual stream, so it
does not perturb the experts' internal dynamics — it only mixes their outputs.
"""

from __future__ import annotations

import math
import torch
from torch import nn

from hagi.config import Config
from hagi.model.ffn import BranchScale
from hagi.model.model import HAGI


class CrossMixer(nn.Module):
    """A cross-block mixing layer on the residual stream.

    ``y = x + gain * down(silu(gate(x)) * up(x))`` with ``gain`` initialized to
    ``mixer_init_scale`` (0 by default). At gain 0 the mixer is the identity, so
    a merged model with zero-init mixers is exactly N independent experts. The
    gain is a single learnable scalar (kept in fp32) that rises as joint
    training teaches the blocks to interact; it is the "how much do the blocks
    talk to each other" knob.

    The mixer is a full-width (H = N * expert_hidden) SwiGLU, so it can mix
    across all blocks. Its ``down`` projection is scaled by ``residual_scale``
    like any other residual branch.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        norm_eps: float = 1e-5,
        residual_scale: float = 1.0,
        mixer_init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
        nn.init.normal_(self.gate.weight, std=hidden_size**-0.5)
        nn.init.normal_(self.up.weight, std=hidden_size**-0.5)
        nn.init.normal_(self.down.weight, std=residual_scale / intermediate_size**0.5)
        self.branch_scale = BranchScale(residual_scale)
        self.keep_fp32 = True
        self.gain = nn.Parameter(torch.tensor(float(mixer_init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        out = self.down(torch.nn.functional.silu(self.gate(h)) * self.up(h))
        return x + self.branch_scale(out) * self.gain.to(x.dtype)


class MergedHAGI(HAGI):
    """A HAGI whose body is N block-diagonally merged experts.

    The encoder, out-norm and head are the *merged* (wide) versions: the
    codebook is ``[V, N*H_i]`` and the head projection is ``[V, N*H_i]``. The
    body blocks are the block-diagonal concatenation of the experts' blocks,
    followed by ``n_mixers`` cross-block mixers.

    The merged model is constructed by :func:`merge_experts` from N expert
    checkpoints. When no checkpoints are given, the current model's weights are
    replicated N times (for machinery smoke tests).
    """

    def __init__(
        self,
        cfg: Config,
        n_mixers: int = 1,
        mixer_init_scale: float = 0.0,
    ) -> None:
        # Ternary quantization does not commute with block-diagonal merging:
        # ternarize normalizes each row by its absmean, and a merged row
        # includes the zero off-diagonal blocks, which changes the scale and
        # breaks the exact-expert equivalence. The merged body therefore uses
        # plain fp16 linear layers (block-diagonal weights applied exactly);
        # ternary can be re-enabled during joint training if desired.
        m = cfg.model
        _saved_ternary = m.ternary.enabled
        m.ternary.enabled = False
        try:
            super().__init__(cfg)
        finally:
            m.ternary.enabled = _saved_ternary
        h = m.hidden_size
        n = cfg.merge.n_experts
        if h % n != 0:
            raise ValueError(f"hidden_size {h} must be divisible by n_experts {n}")
        self.n_experts = n
        self.expert_hidden = h // n

        # Replace the wide RMSNorms with block-wise RMSNorms so each expert's
        # block is normalized independently (a plain wide RMSNorm would
        # normalize the whole concatenated stream at once, which is not the
        # same as per-expert normalization).
        from hagi.model.norms import BlockRMSNorm

        for block in self.blocks:
            block.attn.attn_norm = BlockRMSNorm(n, self.expert_hidden, m.norm_eps)
            block.mixer.norm = BlockRMSNorm(n, self.expert_hidden, m.norm_eps)
        self.out_norm = BlockRMSNorm(n, self.expert_hidden, m.norm_eps)

        # Cross-block mixers on the residual stream.
        inter = max(64, int(2.0 * h))
        residual_scale = (2.0 * m.num_layers * max(1, int(m.loop_depth))) ** -0.5
        self.mixers = nn.ModuleList(
            [
                CrossMixer(h, inter, m.norm_eps, residual_scale, mixer_init_scale)
                for _ in range(n_mixers)
            ]
        )

    def _run_blocks(
        self,
        h: torch.Tensor,
        positions: torch.Tensor | None,
        doc_ids: torch.Tensor | None,
        prefix_len: int,
        t_total: int,
        use_state: bool = False,
    ) -> torch.Tensor:
        h = super()._run_blocks(h, positions, doc_ids, prefix_len, t_total, use_state=use_state)
        for mixer in self.mixers:
            h = mixer(h)
        return h


def _ternarize_block(weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Ternarize a 2D weight per output row (BitNet b1.58), matching the
    experts' BitLinear forward. Returns the effective quantized weight."""
    scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    return (weight / scale).clamp(-1.0, 1.0).round() * scale


def _block_diag(blocks: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate 2D weight blocks into a block-diagonal matrix.

    ``blocks`` are ``[out_i, in_i]``; the result is ``[sum out_i, sum in_i]``
    with the blocks on the diagonal and zeros elsewhere.
    """
    out_total = sum(b.shape[0] for b in blocks)
    in_total = sum(b.shape[1] for b in blocks)
    result = torch.zeros(out_total, in_total, dtype=blocks[0].dtype, device=blocks[0].device)
    row = 0
    col = 0
    for b in blocks:
        result[row : row + b.shape[0], col : col + b.shape[1]] = b
        row += b.shape[0]
        col += b.shape[1]
    return result


def _merge_2d(weights: list[torch.Tensor], block_diag: bool) -> torch.Tensor:
    """Merge a list of same-shaped 2D weights.

    ``block_diag=True``: block-diagonal concatenation (for hidden-mixing
    matrices, where each expert's weight acts on its own subspace).
    ``block_diag=False``: row-wise concatenation (for codebooks / head
    projections, where the output rows are the vocabulary and the input is the
    full hidden space).
    """
    if block_diag:
        return _block_diag(weights)
    return torch.cat(weights, dim=1)


def _merge_1d(weights: list[torch.Tensor]) -> torch.Tensor:
    """Merge 1D gains by concatenation (norms, per-expert gains)."""
    return torch.cat(weights, dim=0)


def merge_experts(
    cfg: Config,
    expert_states: list[dict],
    n_mixers: int = 1,
    mixer_init_scale: float = 0.0,
    drop_expert_mixers: bool = False,
) -> MergedHAGI:
    """Build a merged model from N expert state dicts.

    Args:
        cfg: config with ``model.hidden_size = N * expert_hidden`` and
            ``merge.n_experts = N``.
        expert_states: list of N expert ``state_dict`` mappings (the ``model``
            payload of each checkpoint), in block order.
        n_mixers: number of cross-block mixers to add.
        mixer_init_scale: initial mixer gain (0 = exact independent experts).
        drop_expert_mixers: when True, ignore any ``mixers.*`` keys in the
            expert states (used for hierarchical merging: the level-1 experts
            are themselves merged models with trained mixers, which must be
            dropped and replaced by a fresh level-2 mixer).

    Returns:
        A :class:`MergedHAGI` with the experts' weights block-diagonally
        concatenated and zero-init mixers.
    """
    n = cfg.merge.n_experts
    if len(expert_states) != n:
        raise ValueError(f"expected {n} expert states, got {len(expert_states)}")
    m = cfg.model
    h = m.hidden_size
    if h % n != 0:
        raise ValueError(f"hidden_size {h} must be divisible by n_experts {n}")
    expert_hidden = h // n

    model = MergedHAGI(cfg, n_mixers=n_mixers, mixer_init_scale=mixer_init_scale)
    sd = model.state_dict()

    # Group expert tensors by key. Each expert has the same key set.
    keys = list(expert_states[0].keys())
    for k in keys:
        for st in expert_states:
            if k not in st:
                raise ValueError(f"expert state missing key {k!r}")
            if tuple(st[k].shape) != tuple(expert_states[0][k].shape):
                raise ValueError(f"expert shape mismatch on {k!r}: {tuple(st[k].shape)}")

    for k in keys:
        if k not in sd:
            continue
        if drop_expert_mixers and k.startswith("mixers."):
            # Drop the experts' own mixers; the fresh level-2 mixer (zero-init
            # from the constructor) replaces them.
            continue
        target = sd[k]
        blocks = [st[k] for st in expert_states]
        # Buffers that are shared across the whole model (not per-expert) keep
        # the merged model's own value: the unigram log-prior is a function of
        # the vocabulary, not of any single expert.
        if k.endswith("log_prior"):
            continue
        if target.ndim == 2 and (k.endswith("q_norm.weight") or k.endswith("k_norm.weight")):
            # Per-head QK gains: each expert's gain applies to its own heads.
            # The merged model has per_head_qk=True, so the target is
            # [n_heads, head_dim]. Each expert has q_per_exp (or kv_per_exp)
            # heads, so repeat each expert's gain that many times. For a
            # hierarchical merge the experts are themselves merged models with
            # 2D per-head gains [n_heads_exp, head_dim]; concatenate those
            # along the head axis instead of repeating.
            if blocks[0].ndim == 2:
                merged = torch.cat(blocks, dim=0)
            else:
                q_per_exp = m.attention.num_query_heads // n
                kv_per_exp = m.attention.num_kv_heads // n
                rep = q_per_exp if k.endswith("q_norm.weight") else kv_per_exp
                merged = torch.cat([b.unsqueeze(0).repeat(rep, 1) for b in blocks], dim=0)
        elif target.ndim == 2 and (
            k.endswith("attn_norm.weight")
            or k.endswith("mixer.norm.weight")
            or k.endswith("out_norm.weight")
        ):
            # Block-wise RMSNorm gains: [n_blocks, block_dim]. Stack the
            # experts' [block_dim] gains along the block axis. For a
            # hierarchical merge the experts are themselves merged models with
            # 2D block norms [n_blocks, block_dim]; concatenate those along
            # the block_dim axis so each expert's blocks stay contiguous.
            if blocks[0].ndim == 2:
                merged = torch.cat(blocks, dim=1)
            else:
                merged = torch.stack(blocks, dim=0)
        elif target.ndim == 2:
            # Hidden-mixing matrices (qkv/out/gate/up/down) merge block-diagonal;
            # codebooks and head projections merge row-wise (concat over input).
            if k.endswith((".weight",)) and (
                "qkv_proj" in k or "out_proj" in k or "mixer.gate" in k or "mixer.up" in k or "mixer.down" in k
            ):
                # The experts' BitLinear layers ternarize their weights at
                # forward time. The merged body uses plain fp16 linear layers,
                # so to reproduce the experts exactly we ternarize each expert
                # block *individually* (per-block absmean, matching the expert's
                # own per-row normalization) before the block-diagonal merge.
                tern_blocks = [
                    _ternarize_block(b, m.ternary.eps) for b in blocks
                ]
                if "qkv_proj" in k:
                    # The fused QKV projection has a q block on top and a kv
                    # block below. Each expert contributes its own q and kv
                    # blocks; they must be merged separately so all q blocks
                    # stay on top and all kv blocks below (the merged model's
                    # layout), not interleaved along the diagonal.
                    q_per_exp = m.attention.num_query_heads // n
                    q_out = q_per_exp * m.attention.head_dim
                    q_blocks = [b[:q_out] for b in tern_blocks]
                    kv_blocks = [b[q_out:] for b in tern_blocks]
                    # Each expert's kv block is laid out as [k_all, v_all]
                    # (2 * n_kv_exp * head_dim). The merged model views the
                    # whole kv region as [B, T, 2, n_kv, head_dim], i.e. all
                    # k heads first, then all v heads. So we must collect all
                    # k blocks, then all v blocks -- not interleave per expert.
                    n_kv_exp = m.attention.num_kv_heads // n
                    hd = m.attention.head_dim
                    k_blocks = [b[: n_kv_exp * hd] for b in kv_blocks]
                    v_blocks = [b[n_kv_exp * hd :] for b in kv_blocks]
                    merged = torch.cat(
                        [
                            _block_diag(q_blocks),
                            _block_diag(k_blocks),
                            _block_diag(v_blocks),
                        ],
                        dim=0,
                    )
                else:
                    merged = _merge_2d(tern_blocks, block_diag=True)
            else:
                merged = _merge_2d(blocks, block_diag=False)
        elif target.ndim == 1:
            # Per-hidden-dim gains (attn_norm, mixer.norm, out_norm) concatenate
            # so each expert's norm applies to its own block.
            merged = _merge_1d(blocks)
        else:
            # Scalars (branch_scale, logit_scale): take the first expert's value.
            # logit_scale must be divided by sqrt(N): the merged head projection
            # is the column-concatenation of the experts' codebooks, so
            # ``hidden @ weight.T`` sums N contributions of roughly equal
            # magnitude. Without the 1/sqrt(N) the logits are ~Nx sharper and
            # the output distribution collapses (too confident), which is why a
            # freshly merged model scores poorly (AVG~9) before joint training.
            # This mirrors grow.py's ``_fill_head`` (ref_scale / sqrt(N)).
            if k.endswith("logit_scale"):
                merged = blocks[0] / math.sqrt(n)
            else:
                merged = blocks[0]
        if tuple(merged.shape) != tuple(target.shape):
            raise ValueError(
                f"merged shape {tuple(merged.shape)} != target {tuple(target.shape)} for {k!r}"
            )
        sd[k] = merged

    # Zero-init the mixer gains (already 0 from constructor) and keep the
    # merged model's own out_norm / head as-is (they are the merged versions).
    model.load_state_dict(sd, strict=True)
    return model
