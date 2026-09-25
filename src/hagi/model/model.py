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
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_util
from torch import nn

from hagi.config import Config, count_params, ffn_width, layer_windows
from hagi.model.attention import AttentionConfig, build_attention_mask
from hagi.model.block import Block
from hagi.model.cortex import PyramidalCortex
from hagi.model.decision import DecisionHead
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
        if type(self) is HAGI and cfg.merge.mixer_type == "ternary_f3":
            raise ValueError(
                "HAGI cannot instantiate merge.mixer_type='ternary_f3'; "
                "use RecursiveF3HAGI or build_model_from_payload"
            )
        self.cfg = cfg
        m = cfg.model
        h = m.hidden_size
        use_ternary = m.ternary.enabled

        # Block-diagonal expert merge: the body is built from N experts whose
        # hidden spaces are concatenated. The merged model's head geometry must
        # keep each expert's query heads attending only to that expert's kv
        # heads (block-diagonal attention), so num_query_heads = N * q_per_exp
        # and num_kv_heads = N * kv_per_exp.
        n_exp = max(1, int(getattr(cfg.merge, "n_experts", 1) or 1))
        if getattr(cfg.merge, "enabled", False) and h % n_exp != 0:
            raise ValueError(f"hidden_size {h} must be divisible by n_experts {n_exp}")
        # Per-head QK gains are only needed when the body is a block-diagonal
        # merge of multiple experts; a plain model keeps one shared gain.
        self._n_experts = n_exp if getattr(cfg.merge, "enabled", False) else 1

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
                per_head_qk=self._n_experts > 1,
                fp32_softmax=m.attention.fp32_softmax,
                sink_len=m.attention.sink_len,
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

        # Opt-in residual adapters. Only built when the master switch is on;
        # otherwise blocks.adapters stays None and the original forward is
        # reproduced bit-for-bit (no extra parameters, no extra computation).
        self._attach_adapters(h)

        # Directed same-token memory between contiguous layer levels. The
        # module is absent by default, so the baseline module tree and forward
        # remain unchanged. State is created per pass in _run_blocks, never
        # stored across tokens or generation calls.
        self.cortex: PyramidalCortex | None = (
            PyramidalCortex(m.num_layers, h, m.cortex) if m.cortex.enabled else None
        )
        self.decision_head: DecisionHead | None = (
            DecisionHead(h, m.decision.num_options) if m.decision.enabled else None
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

    def _attach_adapters(self, hidden_size: int) -> None:
        """Attach an opt-in residual adapter to every block.

        No-op when ``model.adapters.enabled`` is False (the default): the blocks
        keep ``adapters = None`` and the original forward is unchanged. When
        enabled, a :class:`~hagi.model.adapters.BlockAdapter` is built per block
        with the shared frozen mixer and attached to the block's ``adapters``
        slot so :meth:`Block.forward` adds the delta after the mixer.
        """
        from hagi.model.adapters import BlockAdapter

        ad = self.cfg.model.adapters
        if not ad.enabled:
            return
        for block in self.blocks:
            adapter = BlockAdapter(ad, hidden_size)
            adapter.attach(block.mixer)
            block.adapters = adapter

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
        sink_len = self.cfg.model.attention.sink_len

        # Prefetch masks once; loops reuse them.
        # Pure window (no docs / no multimodal prefix): leave mask=None so the
        # layer runs correct O(T·W) local_window_attention instead of building
        # a dense T×T band and paying the math-SDPA full-score tax.
        # Exception: when sinks are on, the mask is required so the leading
        # keys stay visible inside the window (local_window_attention has no
        # sink handling).
        for window in self._window_layers:
            if window not in mask_by_window:
                if window > 0 and doc_ids is None and prefix_len <= 0 and sink_len == 0:
                    mask_by_window[window] = None
                else:
                    mask_by_window[window] = build_attention_mask(
                        t_q,
                        t_total,
                        window=window,
                        doc_ids=doc_ids,
                        prefix_len=prefix_len,
                        sink_len=sink_len,
                        device=h.device,
                        dtype=h.dtype,
                    )

        loops = self._loop_depth if not use_state else 1
        for _ in range(loops):
            # A pass-local state container keeps the side channel causal and
            # prevents stale summaries from leaking across loop_depth repeats.
            cortex_states = self.cortex.start_sequence() if self.cortex is not None else None
            for layer_index, (block, window) in enumerate(
                zip(self.blocks, self._window_layers, strict=True)
            ):
                mask = mask_by_window[window]
                if checkpointing:
                    h = checkpoint_util.checkpoint(block, h, positions, mask, use_reentrant=False)
                else:
                    h = block(h, positions, mask)
                if self.cortex is not None and cortex_states is not None and self.cortex.is_boundary(layer_index):
                    level = self.cortex.level_for_boundary(layer_index)
                    h = self.cortex.apply_boundary(h, level, cortex_states)
        return h

    def _apply_mixers(self, h: torch.Tensor) -> torch.Tensor:
        """Post-norm cross-block mixers (identity for a plain model).

        Overridden by :class:`~hagi.model.merge.MergedHAGI` to run the
        cross-block mixers on the normalized residual stream. The hook lives
        here so the ordering (blocks -> out_norm -> mixers -> head) is shared
        by every model and the head always sees ``mixer(out_norm(h))``.
        """
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
        decision_targets: torch.Tensor | None = None,
        decision_mask: torch.Tensor | None = None,
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
            decision_targets: optional ``[B]`` finite-option labels. One
                decision is scored from the last text position per sequence.
                This is the explicit sequence-summary contract: the final
                causal state has attended to all prior text, so later-token
                bias is intentional for incremental decoding.
            decision_mask: optional ``[B]`` bool mask selecting decision rows.

        Returns:
            :class:`ModelOutput`.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] < 1:
            raise ValueError(
                f"input_ids must have shape [B, T] with B >= 1, got {tuple(input_ids.shape)}"
            )
        if self.decision_head is None:
            if decision_targets is not None or decision_mask is not None:
                raise ValueError(
                    "decision_targets/decision_mask require model.decision.enabled=True"
                )
        else:
            batch = input_ids.shape[0]
            if decision_targets is not None:
                if decision_targets.dtype != torch.long:
                    raise ValueError("decision_targets must have dtype torch.long")
                if tuple(decision_targets.shape) != (batch,):
                    raise ValueError(
                        f"decision_targets must have shape [{batch}], got {tuple(decision_targets.shape)}"
                    )
                if decision_targets.device != input_ids.device:
                    raise ValueError("decision_targets must be on the same device as input_ids")
                if int(decision_targets.min()) < 0 or int(decision_targets.max()) >= self.decision_head.num_options:
                    raise ValueError(
                        f"decision_targets must be in [0, {self.decision_head.num_options})"
                    )
            if decision_mask is not None:
                if decision_mask.dtype != torch.bool:
                    raise ValueError("decision_mask must have dtype torch.bool")
                if tuple(decision_mask.shape) != (batch,):
                    raise ValueError(
                        f"decision_mask must have shape [{batch}], got {tuple(decision_mask.shape)}"
                    )
                if decision_mask.device != input_ids.device:
                    raise ValueError("decision_mask must be on the same device as input_ids")
                if decision_targets is None and bool(decision_mask.any()):
                    raise ValueError("a selected decision_mask row requires decision_targets")

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
        # Hook for merged models: cross-block mixers run *after* the output
        # norm so the head sees ``mixer(out_norm(h))``. This ordering is what
        # makes the Hadamard mixer's head pre-rotation exact (see
        # :meth:`MergedHAGI._apply_mixers`).
        h = self._apply_mixers(h)

        text_hidden = h[:, prefix_len:] if prefix_len else h
        out = ModelOutput(hidden=text_hidden)

        if self.decision_head is not None:
            out.decision_logits = self.decision_head(text_hidden[:, -1])

        if return_logits:
            out.logits = self.head.logits(text_hidden)

        lm_loss = None
        if targets is not None:
            if targets.shape[:2] != (input_ids.shape[0], t_text):
                raise ValueError(
                    f"targets shape {tuple(targets.shape)} does not match input_ids "
                    f"{tuple(input_ids.shape)}"
                )

            flat_hidden = text_hidden.reshape(-1, text_hidden.shape[-1])
            flat_targets = targets.reshape(-1)
            if loss_mask is not None:
                # Boolean fancy indexing gathers the scored rows in one op. The
                # previous ``nonzero()`` + ``index_select`` pair forced a GPU-to-CPU
                # sync (variable-length result) and showed up as ~1.1 s/step of
                # host time in the profiler on this ROCm build.
                keep = loss_mask.reshape(-1)
                flat_hidden = flat_hidden[keep]
                flat_targets = flat_targets[keep]

            ce, z_loss = self.head.loss(flat_hidden, flat_targets)
            lm_loss = ce
            if self.cfg.train.z_loss_weight > 0:
                lm_loss = lm_loss + self.cfg.train.z_loss_weight * z_loss

            grounding = None
            if self.bridge is not None and modal_pooled is not None:
                text_pooled = text_hidden.float().mean(dim=1)
                gw = float(self.cfg.model.multimodal.grounding_weight)
                grounding = self.bridge.grounding(text_pooled, modal_pooled.float())
                if gw != 0.0:
                    lm_loss = lm_loss + gw * grounding

            out.ce = ce
            out.z_loss = z_loss
            out.grounding = grounding
            out.n_tokens = int(flat_targets.numel())
            out.lm_loss = lm_loss

        decision_loss = None
        if out.decision_logits is not None and decision_targets is not None:
            row_loss = F.cross_entropy(
                out.decision_logits.float(), decision_targets, reduction="none"
            )
            if decision_mask is None:
                n_decisions = int(decision_targets.numel())
                decision_loss = row_loss.mean()
                out.decision_targets = decision_targets
            else:
                selected = row_loss * decision_mask.to(row_loss.dtype)
                n_decisions = int(decision_mask.sum().item())
                if n_decisions:
                    decision_loss = selected.sum() / n_decisions
                    out.decision_targets = decision_targets[decision_mask]
                else:
                    out.decision_targets = decision_targets[:0]
            out.n_decisions = n_decisions

        total = None
        if lm_loss is not None:
            total = lm_loss
        if decision_loss is not None:
            weighted = self.cfg.model.decision.loss_weight * decision_loss
            total = weighted if total is None else total + weighted
        out.decision_loss = decision_loss
        out.loss = total
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
