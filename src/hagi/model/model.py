"""HAGI V31 — a causal language model stated as a communication system.

    tokens
      -> source coder        (codebook + causal pulse-shaping filter)
      -> [multimodal prefix] (per-modality coders + fixed-rate bridge)
      -> channel             (L ternary blocks: QK-normed GQA + SwiGLU/MoE)
      -> output norm
      -> receiver            (tied head + unigram prior + chunked CE)

One path. Every tensor that leaves the source coder reaches the receiver; there
are no auxiliary branches reading detached copies of the hidden state, and no
loss terms competing with the coding objective. That is a deliberate reversal of
V28, which had five off-path modules (variational bottleneck, latent memory bank,
HEP refiner, EXIT halt, water-filling allocator) and shipped with all of them
disabled in both production configs — the code paths remained, the complexity
remained, and the failure modes they introduced remained.

The objective:

    loss = CE + w_z * z_loss + w_router_z * router_z_loss [+ w_ground * grounding]

CE is the channel's coding cost. The two z-losses bound log-partition drift,
which is numerical conditioning rather than a modelling preference. Grounding
appears only when a second modality is present. Load balance is *not* a loss: it
is a bias controller inside the router, so nothing competes with CE for gradient.
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint as checkpoint_util
from torch import nn

from hagi.config import (
    Config,
    count_params,
    ffn_width,
    layer_windows,
    moe_layers,
)
from hagi.model.attention import AttentionConfig, build_attention_mask
from hagi.model.block import Block, build_mixer
from hagi.model.embedding import SourceEncoder
from hagi.model.head import LMHead
from hagi.model.kv_cache import KVCache
from hagi.model.norms import RMSNorm
from hagi.model.outputs import ModelOutput


class HAGI(nn.Module):
    """Causal LM: source coder, ternary channel, receiver.

    Args:
        cfg: top-level :class:`~hagi.config.Config`.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        h = m.hidden_size
        use_ternary = m.ternary.enabled

        self.encoder = SourceEncoder(
            vocab_size=m.vocab_size,
            hidden_size=h,
            conv_kernel=m.embedding.conv_kernel,
            norm_eps=m.norm_eps,
            init_std=m.embedding.init_std,
        )

        # Residual scaling: 2L branches each of variance s^2 keep the stream at
        # O(1) when s = 1/sqrt(2L). Without it the stream's variance grows
        # linearly in depth and the first phase of training is spent undoing it.
        residual_scale = (2.0 * m.num_layers) ** -0.5
        windows = layer_windows(m)
        is_moe = moe_layers(m)
        intermediate = ffn_width(m)

        self.blocks = nn.ModuleList()
        for layer in range(m.num_layers):
            attn_cfg = AttentionConfig(
                num_heads=m.attention.num_query_heads,
                num_kv_heads=m.attention.num_kv_heads,
                head_dim=m.attention.head_dim,
                rope_theta=m.attention.rope_theta,
                qk_norm=m.attention.qk_norm,
                sliding_window=windows[layer],
            )
            mixer = build_mixer(
                h, intermediate, m.moe, is_moe[layer], m.norm_eps, use_ternary, residual_scale
            )
            self.blocks.append(Block(h, attn_cfg, mixer, m.norm_eps, use_ternary, residual_scale))

        self.out_norm = RMSNorm(h, eps=m.norm_eps)
        self.head = LMHead(
            h,
            m.vocab_size,
            m.head,
            tied_weight=self.encoder.weight if m.embedding.tie_lm_head else None,
        )

        self.bridge = None
        if m.multimodal.enabled:
            from hagi.model.multimodal import MultimodalBridge

            self.bridge = MultimodalBridge(cfg)

        self._window_layers = windows
        self._uniform_window = windows[0] if len(set(windows)) == 1 else None

    def param_summary(self) -> dict[str, int]:
        """Analytic parameter counts by group (see :func:`~hagi.config.count_params`)."""
        return count_params(self.cfg.model)

    def allocate_cache(self, dtype: torch.dtype, device: torch.device) -> list[KVCache]:
        """Attach a fresh KV-cache to every layer and return the list."""
        a = self.cfg.model.attention
        caches = []
        for block in self.blocks:
            cache = KVCache(a.max_seq_len, a.num_kv_heads, a.head_dim, dtype, device)
            block.attn.attach_cache(cache)
            caches.append(cache)
        return caches

    def reset_cache(self) -> None:
        """Detach all KV-caches and clear the source filter's decode state."""
        for block in self.blocks:
            block.attn.detach_cache()
        self.encoder.reset_state()

    def commit_controller_updates(self) -> None:
        """Apply every MoE router's deferred bias update.

        Called once per optimizer step after backward. Deferring keeps the
        forward pure, which activation checkpointing requires: the recomputed
        forward must select the same experts as the original.
        """
        for block in self.blocks:
            if block.is_moe:
                block.mixer.commit_bias_update()

    def _run_blocks(
        self,
        h: torch.Tensor,
        positions: torch.Tensor | None,
        doc_ids: torch.Tensor | None,
        prefix_len: int,
        t_total: int,
    ) -> torch.Tensor:
        """Run the stack, building one mask per distinct window size.

        Masks depend only on ``(t_q, t_total, window, doc_ids, prefix_len)``, so
        layers sharing a window share a mask. With the default 1:3 relay pattern
        that is two mask builds per forward instead of L.
        """
        t_q = h.shape[1]
        checkpointing = self.training and self.cfg.train.grad_checkpointing
        mask_by_window: dict[int, torch.Tensor | None] = {}

        for block, window in zip(self.blocks, self._window_layers, strict=True):
            if window not in mask_by_window:
                mask_by_window[window] = build_attention_mask(
                    t_q,
                    t_total,
                    window=window,
                    doc_ids=doc_ids,
                    prefix_len=prefix_len,
                    device=h.device,
                    dtype=h.dtype,
                )
            mask = mask_by_window[window]
            if checkpointing:
                h = checkpoint_util.checkpoint(block, h, positions, mask, use_reentrant=False)
            else:
                h = block(h, positions, mask)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        doc_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        spectrograms: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        use_cache: bool = False,
        return_logits: bool = False,
    ) -> ModelOutput:
        """Encode, transmit, decode.

        Args:
            input_ids: ``[B, T]`` token IDs.
            targets: ``[B, T]`` next-token targets, already shifted by the data
                pipeline. When None only ``hidden``/``logits`` are produced.
            doc_ids: ``[B, T]`` document id per position, for packed batches. The
                mask then forbids attention across document boundaries.
            loss_mask: ``[B, T]`` positions to score. None scores everything.
            images: ``[B, C, H, W]`` (multimodal).
            spectrograms: ``[B, n_mels, T_frames]`` (multimodal).
            positions: ``[T]`` absolute positions for RoPE; None derives them
                from the cache length.
            use_cache: incremental decode (source filter state + KV-cache).
            return_logits: also return full logits. Costs ``B*T*V`` floats — for
                generation and diagnostics only.

        Returns:
            :class:`ModelOutput`.
        """
        h = self.encoder(input_ids, use_state=use_cache)
        t_text = h.shape[1]

        prefix_len = 0
        modal_pooled = None
        if self.bridge is not None and (images is not None or spectrograms is not None):
            prefix, modal_pooled = self.bridge(images, spectrograms)
            if prefix is not None:
                h = torch.cat([prefix.to(h.dtype), h], dim=1)
                prefix_len = prefix.shape[1]
                if doc_ids is not None:
                    pad = doc_ids.new_full((doc_ids.shape[0], prefix_len), -1)
                    doc_ids = torch.cat([pad, doc_ids], dim=1)

        cache_len = self.blocks[0].attn._kv_cache.length if use_cache and self.blocks else 0
        t_total = cache_len + h.shape[1]
        h = self._run_blocks(h, positions, doc_ids, prefix_len, t_total)
        h = self.out_norm(h)

        text_hidden = h[:, prefix_len:] if prefix_len else h
        out = ModelOutput(hidden=text_hidden)

        if return_logits:
            out.logits = self.head.logits(text_hidden)

        if targets is None:
            return out

        if targets.shape[:2] != (input_ids.shape[0], t_text):
            raise ValueError(
                f"targets shape {tuple(targets.shape)} does not match input_ids "
                f"{tuple(input_ids.shape)}"
            )

        flat_hidden = text_hidden.reshape(-1, text_hidden.shape[-1])
        flat_targets = targets.reshape(-1)
        if loss_mask is not None:
            keep = loss_mask.reshape(-1).nonzero(as_tuple=True)[0]
            flat_hidden = flat_hidden.index_select(0, keep)
            flat_targets = flat_targets.index_select(0, keep)

        ce, z_loss = self.head.loss(flat_hidden, flat_targets)
        loss = ce
        if self.cfg.train.z_loss_weight > 0:
            loss = loss + self.cfg.train.z_loss_weight * z_loss

        router_z = None
        for block in self.blocks:
            if block.is_moe and block.mixer.last_router_z_loss is not None:
                term = block.mixer.last_router_z_loss
                router_z = term if router_z is None else router_z + term
        if router_z is not None and self.cfg.train.moe_z_loss_weight > 0:
            loss = loss + self.cfg.train.moe_z_loss_weight * router_z

        grounding = None
        if self.bridge is not None and modal_pooled is not None:
            text_pooled = text_hidden.float().mean(dim=1)
            grounding = self.bridge.grounding(text_pooled, modal_pooled.float())
            loss = loss + grounding

        out.loss = loss
        out.ce = ce
        out.z_loss = z_loss
        out.router_z_loss = router_z
        out.grounding = grounding
        out.n_tokens = int(flat_targets.numel())
        return out

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        """Scalar observables that identify *which* failure mode is active.

        * ``moe/entropy_ratio`` — usable fraction of the expert channels;
          decaying toward ``1/E`` means routing collapse.
        * ``qk_gain`` — mean QK-norm gain; a rising value is the leading
          indicator of softmax saturation.
        * ``residual_gain`` — mean output-norm gain; tracks stream scale drift.
        * ``logit_scale`` — receiver gain. It starts at ``1/sqrt(H)`` (the
          receiver sitting exactly at the unigram prior) and should rise as the
          conditional part of the code becomes informative; a value falling back
          toward 0 means the head has given up and is emitting the prior.
        """
        stats: dict[str, float] = {}
        moe_blocks = [b for b in self.blocks if b.is_moe]
        if moe_blocks:
            per_block = [b.mixer.load_stats() for b in moe_blocks]
            for key in per_block[0]:
                stats[f"moe/{key}"] = sum(s[key] for s in per_block) / len(per_block)

        gains = [b.attn.q_norm.weight.abs().mean() for b in self.blocks if b.attn.q_norm is not None]
        if gains:
            stats["qk_gain"] = float(torch.stack(gains).mean())
        stats["residual_gain"] = float(self.out_norm.weight.abs().mean())
        stats["logit_scale"] = float(self.head.logit_scale)
        return stats
