"""HAGI V41: source-matched conditional ternary channel.

    tokens (+ optional fixed-rate image/audio prefix)
      -> source coder       (codebook + causal pulse-shaping filter)
      -> ternary channel    (L=4, local/global QK-normalized GQA + SwiGLU)
      -> conditional head  (shared-bank NCE, q = unigram source prior)
      -> exact receiver    (full alphabet for generation and calibration)

The training path scores every text symbol. Conditional NCE provides the fast
gradient signal; periodic exact CE is the coding-cost SSOT.
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint as checkpoint_util
from torch import nn

from hagi.config import Config, count_params, ffn_width, layer_windows
from hagi.model.attention import AttentionConfig, build_attention_mask
from hagi.model.block import Block
from hagi.model.embedding import SourceEncoder
from hagi.model.ffn import FeedForward
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

        # Residual scaling: 2 * L_eff branches each of variance s^2 keep the stream
        # at O(1) when s = 1/sqrt(2*L_eff). With loop_depth > 1 the same unique
        # blocks are applied multiple times, so the effective branch count is
        # num_layers * loop_depth (not just num_layers).
        loop = max(1, int(m.loop_depth))
        residual_scale = (2.0 * m.num_layers * loop) ** -0.5
        windows = layer_windows(m)
        intermediate = ffn_width(m)

        self.blocks = nn.ModuleList()
        init_ortho = m.init_orthogonal
        # One RoPE table shared by every layer (identical head_dim / theta).
        from hagi.model.rope import RotaryEmbedding

        shared_rope = RotaryEmbedding(m.attention.head_dim, rope_theta=m.attention.rope_theta)
        self.rope = shared_rope  # registered so buffers move with .to(device)
        for layer in range(m.num_layers):
            attn_cfg = AttentionConfig(
                num_heads=m.attention.num_query_heads,
                num_kv_heads=m.attention.num_kv_heads,
                head_dim=m.attention.head_dim,
                rope_theta=m.attention.rope_theta,
                qk_norm=m.attention.qk_norm,
                sliding_window=windows[layer],
                history_stride=m.sliding.history_stride,
            )
            mixer = FeedForward(
                h,
                intermediate,
                m.norm_eps,
                use_ternary,
                residual_scale,
                init_ortho,
            )
            self.blocks.append(
                Block(
                    h,
                    attn_cfg,
                    mixer,
                    m.norm_eps,
                    use_ternary,
                    residual_scale,
                    init_orthogonal=init_ortho,
                    rope=shared_rope,
                )
            )


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
        self._loop_depth = loop

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
        """Clamp the sampled receiver gain after an optimizer update."""
        gain_max = float(self.cfg.model.head.logit_scale_max)
        if gain_max > 0:
            with torch.no_grad():
                self.head.logit_scale.clamp_(max=gain_max)

    def _run_blocks(
        self,
        h: torch.Tensor,
        positions: torch.Tensor | None,
        doc_ids: torch.Tensor | None,
        prefix_len: int,
        t_total: int,
        use_state: bool = False,
    ) -> torch.Tensor:
        """Run the stack, building one mask per distinct window size.

        Masks depend only on ``(t_q, t_total, window, doc_ids, prefix_len)``, so
        layers sharing a window share a mask. With the default 1:3 relay pattern
        that is two mask builds per forward instead of L.

        With ``loop_depth > 1`` the unique blocks are applied repeatedly
        (weight-tied depth). Each pass reuses the same mask cache; only the
        first pass may write into a KV-cache (decode is single-pass).
        """
        t_q = h.shape[1]
        checkpointing = self.training and self.cfg.train.grad_checkpointing
        mask_by_window: dict[int, torch.Tensor | None] = {}

        # Prefetch masks once; loops reuse them.
        # Pure window (no docs / no multimodal prefix): leave mask=None so the
        # layer runs correct O(T·W) local_window_attention instead of building
        # a dense T×T band and paying the math-SDPA full-score tax.
        for window in self._window_layers:
            if window not in mask_by_window:
                if window > 0 and doc_ids is None and prefix_len <= 0:
                    mask_by_window[window] = None
                else:
                    mask_by_window[window] = build_attention_mask(
                        t_q,
                        t_total,
                        window=window,
                        doc_ids=doc_ids,
                        prefix_len=prefix_len,
                        device=h.device,
                        dtype=h.dtype,
                    )

        loops = self._loop_depth if not use_state else 1
        for _ in range(loops):
            for block, window in zip(self.blocks, self._window_layers, strict=True):
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
        h = self._run_blocks(h, positions, doc_ids, prefix_len, t_total, use_state=use_cache)
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

        grounding = None
        if self.bridge is not None and modal_pooled is not None:
            text_pooled = text_hidden.float().mean(dim=1)
            gw = float(self.cfg.model.multimodal.grounding_weight)
            grounding = self.bridge.grounding(text_pooled, modal_pooled.float())
            if gw != 0.0:
                loss = loss + gw * grounding

        out.loss = loss
        out.ce = ce
        out.z_loss = z_loss
        out.grounding = grounding
        out.n_tokens = int(flat_targets.numel())
        return out

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        """Scalar observables that identify *which* failure mode is active.

        * ``qk_gain`` — mean QK-norm gain; a rising value is the leading
          indicator of softmax saturation.
        * ``residual_gain`` — mean output-norm gain; tracks stream scale drift.
        * ``logit_scale`` — receiver gain. It starts at ``1/sqrt(H)`` (the
          receiver sitting exactly at the unigram prior) and should rise as the
          conditional part of the code becomes informative; a value falling back
          toward 0 means the head has given up and is emitting the prior.
        """
        stats: dict[str, float] = {}
        gains = [b.attn.q_norm.weight.abs().mean() for b in self.blocks if b.attn.q_norm is not None]
        if gains:
            stats["qk_gain"] = float(torch.stack(gains).mean())
        stats["residual_gain"] = float(self.out_norm.weight.abs().mean())
        stats["logit_scale"] = float(self.head.logit_scale)
        return stats
