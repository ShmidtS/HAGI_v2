"""HAGI V24 — integrated information-theoretic multimodal model.

Pipeline (docs/ARCHITECTURE_V24.md):
  source encode (ConvEmbed + sinusoidal PE + TransformerBlock stack)
    -> InformationBottleneck (H->C, variational RD, §3.1)
    -> PredictiveDecoder (extrinsic error highway, §3.2)
    -> source decode (rate_up C->H, expression stack, factored LM head)

The V23 AWGN/LDPC SCS codec (the self-inflicted-channel flaw) is replaced.
When ``model.v24.enabled`` is False the model still constructs the V23 codec
modules for ablation, but the default forward uses the honest V24 path.

V24 modules are reused, not rewritten: the bottleneck/predictive-decoder/
hebbian-FFN live in their own files; ``uncertainty.py`` (V23) is reused
inside PredictiveDecoder for the K=P/(P+R) blend.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.block import AttentionConfig, HebbianFFNConfig, TransformerBlock
from hagi_v4.model.bottleneck import BottleneckConfig, InformationBottleneck
from hagi_v4.model.conv_embedding import ConvEmbedding
from hagi_v4.model.multimodal_input import MultimodalInput
from hagi_v4.model.norms import RMSNorm
from hagi_v4.model.outputs import AuxLosses, ModelOutput
from hagi_v4.model.predictive_decoder import PredictiveConfig, PredictiveDecoder

class HAGIv4(nn.Module):
    """V24 integrated model. Optional V23 codec ablation via ``cfg.model.v24.enabled=False``."""

    def __init__(self, cfg: HAGIv4Config) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        self.v24_enabled = bool(m.v24.enabled)
        self.v25_enabled = bool(getattr(m, "v25", None) and m.v25.enabled)
        H = m.hidden_size
        C = m.core_hidden_size if (self.v24_enabled or self.v25_enabled) else m.core_hidden_size
        self._H, self._C = H, C
        self.inference_config = type("Inf", (), {"vocab_size": m.vocab_size})()

        # ---- Source encoder (V9 factorized ConvEmbedding — kept) ----
        self.embed = ConvEmbedding(
            vocab_size=m.vocab_size,
            hidden_size=H,
            factor_rank=m.embeddings.factor_rank,
            kernel_size=m.embeddings.kernel_size,
            norm_eps=m.norm_eps,
        )
        self.semantic_unknown_embed = nn.Parameter(torch.empty(H))
        nn.init.normal_(self.semantic_unknown_embed, std=1.0 / math.sqrt(H))

        self._pilot_pos_cache: dict[tuple[int, int], torch.Tensor] = {}

        # Attention/FFN config objects for the TransformerBlock stack.
        attn_cfg = AttentionConfig(
            num_heads=max(1, m.attention.num_query_heads),
            head_dim=m.attention.head_dim,
            rope_theta=m.attention.rope_theta,
            attn_entropy_floor=cfg.train.attn_entropy_floor,
        )
        v24 = m.v24
        v25 = m.v25
        if self.v25_enabled:
            # V25 ternary body (BitNet b1.58). Hebbian expansion comes from the
            # V25 config; ternary masters route to Muon via is_muon_param.
            ffn_cfg = HebbianFFNConfig(expansion=v25.hebbian_expansion, dropout=v25.hebbian_dropout)
            use_ternary = bool(v25.use_ternary)
        else:
            ffn_cfg = HebbianFFNConfig(expansion=v24.hebbian_expansion, dropout=v24.hebbian_dropout)
            use_ternary = False
        self._v25_use_ternary = use_ternary
        # V25 main-path-bypass fix: the variational IB + PredictiveDecoder are
        # OUT of the main LM signal path by default (they deadlocked from-scratch
        # training). bottleneck_in_path=True restores the failed original design
        # for the ablation comparison only.
        self._v25_bottleneck_in_path = bool(getattr(v25, "bottleneck_in_path", False))

        # Context stack (perception) + expression stack.
        if self.v25_enabled:
            # V25 ternary blocks. Loop counts come from the v25 config
            # (context_layers / expression_layers) when positive, else fall back
            # to the model-level perception/expression layers.
            ctx_n = v25.context_layers if v25.context_layers > 0 else m.perception_layers
            expr_n = v25.expression_layers if v25.expression_layers > 0 else m.expression_layers
            from hagi_v4.model.block_v25 import TransformerBlock as V25Block

            self.context_stack = nn.ModuleList(
                V25Block(H, attn_cfg, ffn_cfg, m.norm_eps, use_ternary=use_ternary) for _ in range(ctx_n)
            )
            self.expression = nn.ModuleList(
                V25Block(H, attn_cfg, ffn_cfg, m.norm_eps, use_ternary=use_ternary) for _ in range(expr_n)
            )
        else:
            self.context_stack = nn.ModuleList(
                TransformerBlock(H, attn_cfg, ffn_cfg, m.norm_eps) for _ in range(m.perception_layers)
            )
            self.expression = nn.ModuleList(
                TransformerBlock(H, attn_cfg, ffn_cfg, m.norm_eps) for _ in range(m.expression_layers)
            )

        if self.v25_enabled:
            # ---- §3.2 variational information bottleneck (V25) ----
            from hagi_v4.model.bottleneck_v25 import BottleneckConfig as V25BnCfg
            from hagi_v4.model.bottleneck_v25 import InformationBottleneck as V25IB

            bn_cfg = V25BnCfg(
                dim=C,
                ib_beta=v25.ib_beta,
                distortion_weight=v25.distortion_weight,
                perception_weight=v25.perception_weight,
                kl_free_bits=v25.kl_free_bits,
                logvar_clamp=v25.logvar_clamp,
                distortion_eps=v25.distortion_eps,
            )
            self.bottleneck = V25IB(H, bn_cfg, m.norm_eps)

            # ---- §3.3 predictive decoder (extrinsic error highway + HEP) ----
            from hagi_v4.model.predictive_v25 import PredictiveConfig as V25PdCfg
            from hagi_v4.model.predictive_v25 import PredictiveDecoder as V25PD

            pd_cfg = V25PdCfg(
                train_iterations=v25.predictive_train_iterations,
                infer_iterations=v25.predictive_infer_iterations,
                convergence_threshold=v25.predictive_convergence_threshold,
                update_hidden=v25.predictive_update_hidden,
            )
            self.predictive_decoder = V25PD(
                C, H, pd_cfg, m.norm_eps,
                use_kalman_blend=v25.use_kalman_blend,
                use_ternary=use_ternary,
                hep_enabled=v25.hep_enabled,
            )

            # Source-decode rate-up C->H. Ternary 2D hidden weight when the body
            # is ternarized (the only source-decode mixing matrix on the V25
            # path per §5). The FP master routes to Muon (2D, not excluded).
            if use_ternary:
                from hagi_v4.model.ternary import BitLinear

                self.rate_up = BitLinear(C, H, bias=False)
                nn.init.normal_(self.rate_up.weight, std=1.0 / math.sqrt(C))
            else:
                self.rate_up = nn.Linear(C, H, bias=False)
                nn.init.normal_(self.rate_up.weight, std=1.0 / math.sqrt(C))
        elif self.v24_enabled:
            # ---- §3.1 information bottleneck ----
            bn_cfg = BottleneckConfig(
                dim=C,
                ib_beta=v24.ib_beta,
                distortion_weight=v24.distortion_weight,
                perception_weight=v24.perception_weight,
                kl_free_bits=v24.kl_free_bits,
            )
            self.bottleneck = InformationBottleneck(H, bn_cfg, m.norm_eps)

            # ---- §3.2 predictive decoder (reuses uncertainty.py) ----
            pd_cfg = PredictiveConfig(
                train_iterations=v24.predictive_train_iterations,
                infer_iterations=v24.predictive_infer_iterations,
                convergence_threshold=v24.predictive_convergence_threshold,
                update_hidden=v24.predictive_update_hidden,
            )
            self.predictive_decoder = PredictiveDecoder(
                C, H, pd_cfg, m.norm_eps, use_kalman_blend=v24.use_kalman_blend
            )

            # Source-decode rate-up C->H (zero-bypass: only the refined latent).
            self.rate_up = nn.Linear(C, H, bias=False)
            nn.init.normal_(self.rate_up.weight, std=1.0 / math.sqrt(C))
        else:
            # V23 ablation path: build the full SCS codec (AWGN/LDPC) so the
            # model can be A/B compared. Constructed lazily to avoid importing
            # codec/ when only V24/V25 is used.
            self._build_v23_codec(cfg)

        self.final_norm = RMSNorm(H, eps=m.norm_eps)

        # Factored LM head (independent rank-r factorization, V12 win).
        r = m.embeddings.factor_rank
        self.lm_compress = nn.Linear(H, r, bias=False)
        self.lm_expand = nn.Linear(r, m.vocab_size, bias=False)
        nn.init.normal_(self.lm_compress.weight, std=1.0 / math.sqrt(H))
        nn.init.normal_(self.lm_expand.weight, std=1.0 / math.sqrt(r))

        self.multimodal_input: nn.Module | None = None
        mm_on = m.multimodal.enabled or (self.v25_enabled and v25.multimodal_enabled)
        if mm_on:
            if self.v25_enabled:
                # V25 §3.4: shared/specific subspace + inverse-variance fusion
                # (replaces V24's naive additive MultimodalInput).
                from hagi_v4.model.multimodal_v25 import MultimodalFusion

                self.multimodal_input = MultimodalFusion(cfg, text_encoder=self.embed)
            else:
                self.multimodal_input = MultimodalInput(cfg, text_encoder=self.embed)
        self._mm_on = bool(self.multimodal_input is not None)

        self._init_weights()
        for mod in self.modules():
            if hasattr(mod, "set_attn_entropy_floor"):
                mod.set_attn_entropy_floor(cfg.train.attn_entropy_floor)

    def _build_v23_codec(self, cfg: HAGIv4Config) -> None:
        """Construct the V23 SCS codec (ablation). Only when v24.enabled=False."""
        from hagi_v4.model.codec import ChannelDecoder, ChannelEncoder, SourceDecoder, SourceEncoder
        from hagi_v4.model.codec_contracts import CodecShapeConfig

        codec_shape = CodecShapeConfig.from_hagi_config(cfg)
        self.source_encoder = SourceEncoder(cfg)
        self.channel_encoder = ChannelEncoder(cfg, codec_shape, core_mask_embed=self.source_encoder.core_mask_embed)
        self.channel_decoder = ChannelDecoder(
            cfg, codec_shape,
            interleaver=self.channel_encoder.interleaver,
            shared_parity_weights=self.channel_encoder.parity_encoder.edge_log_scale,
            shared_sparse_mask=self.channel_encoder.parity_encoder.sparse_mask,
            shared_edge_log_scale=self.channel_encoder.parity_encoder.edge_log_scale,
            shared_parity_base=self.channel_encoder.parity_encoder.parity_base,
        )
        self.source_decoder = SourceDecoder(
            cfg,
            rate_up=self.source_encoder.rate_up,
            token_compress_weight=self.source_encoder.embed.token_compress.weight
            if cfg.model.embeddings.use_conv_embedding
            else None,
        )

    def _init_weights(self) -> None:
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                if "rate_up" in name or "lm_compress" in name or "lm_expand" in name:
                    continue
                if name.endswith("out_proj") or name.endswith("W") or "update_out" in name:
                    continue
                std = 1.0 / math.sqrt(max(1, mod.weight.shape[1]))
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self.lm_expand.weight @ self.lm_compress.weight

    def _pilot_pe(self, T: int, H: int, device: torch.device) -> torch.Tensor:
        key = (T, H)
        if key not in self._pilot_pos_cache:
            if len(self._pilot_pos_cache) > 8:
                self._pilot_pos_cache.clear()
            pos = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, H, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / max(H, 1))
            )
            pe = torch.zeros(T, H, device=device, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(pos * div[: pe[:, 0::2].shape[1]])
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
            self._pilot_pos_cache[key] = pe
        return self._pilot_pos_cache[key]

    def _stack_forward(self, h, stack, attention_mode="bidir", prefix_len=None, soft_beta=None):
        entropy_pen = None
        # Gradient checkpointing: recompute each block's activations in backward
        # instead of holding them. Activations drop from O(layers) to O(1), which
        # removes the dominant VRAM pressure (the [B,H,T,T] attention matrices
        # and FFN intermediates of every layer are recomputed, not stored). The
        # attention entropy penalty is read from the block's side-effect after
        # the (cached) call so it still contributes to the loss.
        checkpointing = self.training and len(stack) > 1
        for blk in stack:
            if checkpointing:
                def run(b_in, *, blk=blk, am=attention_mode, pl=prefix_len, sb=soft_beta):
                    return blk(b_in, attention_mode=am, prefix_len=pl, soft_beta=sb)
                h = ckpt.checkpoint(run, h, use_reentrant=False)
            else:
                h = blk(h, attention_mode=attention_mode, prefix_len=prefix_len, soft_beta=soft_beta)
            pen = getattr(blk, "_last_attn_entropy_penalty", None)
            if pen is not None:
                entropy_pen = pen if entropy_pen is None else entropy_pen + pen
        self._last_attn_entropy_penalty = entropy_pen
        return h

    def _encode_text(self, input_ids, semantic_unknown_mask):
        embed_dev = self.embed.weight.device
        ids_dev = input_ids.device
        unknown = self.semantic_unknown_embed.to(device=embed_dev)
        if embed_dev != ids_dev:
            ids_on = input_ids.to(embed_dev)
            mask_on = semantic_unknown_mask.to(embed_dev) if semantic_unknown_mask is not None else None
            h = self.embed.forward_with_erasure(ids_on, unknown, mask_on).to(ids_dev)
        else:
            h = self.embed.forward_with_erasure(input_ids, unknown, semantic_unknown_mask)
        h = h + (1.0 / math.sqrt(self._H)) * self._pilot_pe(h.shape[1], self._H, h.device).to(h.dtype).unsqueeze(0)
        return h

    def forward(
        self,
        input_ids=None,
        targets=None,
        *,
        semantic_unknown_mask=None,
        prediction_mask=None,
        valid_target_mask=None,
        physical_corruption_mask=None,
        step=0,
        images=None,
        spectrograms=None,
        attention_mode="bidir",
        prefix_len=None,
        soft_beta=None,
        refinement_iterations=None,
        awgn_sigma=None,
        cache=None,
        **_unused,
    ):
        """V25/V24/V23 forward. ``physical_corruption_mask`` / ``awgn_sigma`` /
        ``cache`` are accepted for call-site compatibility with the V23 training
        loop; only the V23 ablation path uses them (V25/V24 have no self-noise)."""
        if input_ids is None:
            raise ValueError("input_ids is required")
        if awgn_sigma is not None:
            self.set_awgn_sigma(awgn_sigma)
        del cache

        if self.v25_enabled:
            return self._forward_v25(
                input_ids, targets, semantic_unknown_mask, prediction_mask, valid_target_mask,
                images, spectrograms, attention_mode, prefix_len, soft_beta, refinement_iterations,
            )
        if self.v24_enabled:
            return self._forward_v24(
                input_ids, targets, semantic_unknown_mask, prediction_mask, valid_target_mask,
                images, spectrograms, attention_mode, prefix_len, soft_beta, refinement_iterations,
            )
        return self._forward_v23(
            input_ids, targets, semantic_unknown_mask, prediction_mask, valid_target_mask,
            physical_corruption_mask, step, attention_mode, prefix_len, soft_beta, refinement_iterations,
        )

    def _forward_v25(self, input_ids, targets, semantic_unknown_mask, prediction_mask,
                     valid_target_mask, images, spectrograms, attention_mode, prefix_len, soft_beta,
                     refinement_iterations):
        """V25 ternary RD-channel forward (ARCHITECTURE_V25.md, main-path-bypass fix).

        Main LM path (PROVEN to learn — ablation masked_ce 11.4->0.39):
          source-encode -> ternary context stack -> causal expression stack
          -> final_norm -> factored LM head.

        The variational InformationBottleneck is an AUXILIARY rate regularizer:
        it computes KL/distortion/perception on h_ctx (the only genuine "rate")
        but does NOT intercept the LM signal. Inserting it + the zero-init
        PredictiveDecoder into the main path starves the LM head of a stable
        signal at init -> deadlock (root cause of the V25 step-1500 garbage).

        Opt-in ablation (``v25.bottleneck_in_path=True``) restores the original
        IB -> PD -> rate_up in-path design for the failed-path comparison.
        """
        # STAGE 1 — modal source encode.
        if self._mm_on and (images is not None or spectrograms is not None):
            h, _mod_ids = self.multimodal_input(input_ids, images, spectrograms)[:2]
            T_text = input_ids.shape[1]
            h = h + (1.0 / math.sqrt(self._H)) * self._pilot_pe(h.shape[1], self._H, h.device).to(h.dtype).unsqueeze(0)
        else:
            h = self._encode_text(input_ids, semantic_unknown_mask)
            T_text = input_ids.shape[1]

        # STAGE 2 — ternary context stack (the genuine channel).
        h_ctx = self._stack_forward(h, self.context_stack, attention_mode, prefix_len, soft_beta)

        # STAGE 3 — auxiliary information bottleneck: rate/distortion/perception
        # on h_ctx as KL regularization. Does NOT intercept the LM signal.
        _, bn_info = self.bottleneck(h_ctx)

        # STAGE 4/5 — MAIN LM PATH: causal expression stack directly on h_ctx
        # (no bottleneck reparam, no predictive decoder). The expression stack
        # IS the source-decode + refinement in one (it is a full ternary block
        # stack with its own Hebbian FFN mixing).
        if self._v25_bottleneck_in_path:
            # Opt-in ablation: the failed original design (IB -> PD -> rate_up).
            z = bn_info["z"]
            z_refined, pred_info = self.predictive_decoder(
                z, h_ctx, self.training, iterations=refinement_iterations
            )
            h_dec = self.rate_up(z_refined)
            h_dec = self._stack_forward(h_dec, self.expression, attention_mode="causal")
            iters_used = pred_info["iterations_used"]
        else:
            h_dec = self._stack_forward(h_ctx, self.expression, attention_mode="causal")
            iters_used = torch.zeros(h_ctx.shape[:2], dtype=torch.long, device=h_ctx.device)
        h_dec = self.final_norm(h_dec)

        # STAGE 6 — decision device: slice text positions, factored LM head.
        h_text = h_dec[:, :T_text]
        idx, logits = self._gather_logits(h_text, prediction_mask, valid_target_mask)
        ce_loss = self._ce(idx, logits, targets)

        # STAGE 7 — aux losses. Rate = KL[q(z|h_ctx)||N(0,I)] (the only rate),
        # computed on h_ctx but NOT blocking the LM path. rate_distortion=None
        # (no V23 double-count; distortion is its own term here).
        aux = AuxLosses()
        aux.rate = bn_info["rate"]
        aux.distortion = bn_info["distortion"]
        aux.perception = bn_info["perception"]
        aux.rate_distortion = None
        aux.attn_entropy = getattr(self, "_last_attn_entropy_penalty", None)


        return ModelOutput(
            logits=logits, hidden=h_text, aux=aux, ce_loss=ce_loss,
            iterations_used=iters_used, prediction_indices=idx,
        )

    def _forward_v24(self, input_ids, targets, semantic_unknown_mask, prediction_mask,
                     valid_target_mask, images, spectrograms, attention_mode, prefix_len, soft_beta,
                     refinement_iterations):
        if self.multimodal_input is not None and (images is not None or spectrograms is not None):
            h, _ = self.multimodal_input(input_ids, images, spectrograms)
            T_text = input_ids.shape[1]
            h = h + (1.0 / math.sqrt(self._H)) * self._pilot_pe(h.shape[1], self._H, h.device).to(h.dtype).unsqueeze(0)
        else:
            h = self._encode_text(input_ids, semantic_unknown_mask)
            T_text = input_ids.shape[1]

        h_ctx = self._stack_forward(h, self.context_stack, attention_mode, prefix_len, soft_beta)

        # §3.1 variational information bottleneck.
        z, bn_info = self.bottleneck(h_ctx)
        # §3.2 predictive decode (extrinsic error highway).
        z_refined, pred_info = self.predictive_decoder(z, h_ctx, self.training, iterations=refinement_iterations)

        # Source decode: rate_up then causal expression stack.
        h_dec = self.rate_up(z_refined)
        h_dec = self._stack_forward(h_dec, self.expression, attention_mode="causal")
        h_dec = self.final_norm(h_dec)
        h_text = h_dec[:, :T_text]

        idx, logits = self._gather_logits(h_text, prediction_mask, valid_target_mask)
        ce_loss = self._ce(idx, logits, targets)

        aux = AuxLosses()
        aux.rate = bn_info["rate"]
        aux.distortion = bn_info["distortion"]
        aux.rate_distortion = bn_info["distortion"]
        aux.attn_entropy = getattr(self, "_last_attn_entropy_penalty", None)

        return ModelOutput(
            logits=logits, hidden=h_text, aux=aux, ce_loss=ce_loss,
            iterations_used=pred_info["iterations_used"], prediction_indices=idx,
        )

    def _forward_v23(self, input_ids, targets, semantic_unknown_mask, prediction_mask,
                     valid_target_mask, physical_corruption_mask, step, attention_mode,
                     prefix_len, soft_beta, refinement_iterations):
        """V23 ablation path (AWGN/LDPC). physical_corruption_mask IS used here."""
        from hagi_v4.model.codec_contracts import DecodeState, SemanticMaskBatch

        awgn_sigma = float(getattr(self, "_awgn_sigma", 0.0) or 0.0)
        masks = SemanticMaskBatch(
            semantic_unknown_mask if semantic_unknown_mask is not None else torch.zeros_like(input_ids, dtype=torch.bool),
            prediction_mask if prediction_mask is not None else valid_target_mask,
            valid_target_mask,
            physical_corruption_mask if physical_corruption_mask is not None else torch.zeros_like(input_ids, dtype=torch.bool),
        )
        masks.validate(input_ids, None)
        state = DecodeState()
        encoded = self.source_encoder.forward(
            input_ids, semantic_unknown_mask, None,
            attention_mode=attention_mode, prefix_len=prefix_len, soft_beta=soft_beta,
        )
        ch_encoded = self.channel_encoder.apply_erasure(
            self.channel_encoder.forward(encoded), physical_corruption_mask, awgn_sigma
        )
        self.channel_decoder.set_awgn_sigma(awgn_sigma if awgn_sigma > 0.0 else None)
        decoded = self.channel_decoder.forward(
            ch_encoded, encoded, state, self.training,
            bp_iterations=getattr(self.cfg.train, "bp_iterations", None),
            refinement_iterations=refinement_iterations,
        )
        h = self.source_decoder.forward(decoded, attention_mode="causal")
        h = self.final_norm(h)
        idx, logits = self._gather_logits(h, prediction_mask, valid_target_mask)
        ce_loss = self._ce(idx, logits, targets)

        aux = AuxLosses()
        si = decoded.side_info
        aux.rate_distortion = (encoded.systematic.float() - decoded.latent.float()).pow(2).mean().to(h.dtype)
        if si.get("parity_strength") is not None:
            aux.parity = si["parity_strength"]
        aux.attn_entropy = getattr(self, "_last_attn_entropy_penalty", None)
        return ModelOutput(
            logits=logits, hidden=h, aux=aux, ce_loss=ce_loss,
            iterations_used=si.get("iterations_used"), prediction_indices=idx,
        )

    def _gather_logits(self, h_text, prediction_mask, valid_target_mask):
        if prediction_mask is not None and valid_target_mask is not None:
            selected = prediction_mask & valid_target_mask
            idx = selected.flatten().nonzero(as_tuple=False).squeeze(-1)
            sel_h = h_text.flatten(0, 1).index_select(0, idx.to(h_text.device))
            logits = self.lm_expand(self.lm_compress(sel_h))
        else:
            idx = None
            logits = self.lm_expand(self.lm_compress(h_text))
        return idx, logits

    @staticmethod
    def _ce(idx, logits, targets):
        if targets is None or idx is None or idx.numel() == 0:
            return None
        sel_t = targets.flatten().index_select(0, idx.to(targets.device))
        return F.cross_entropy(logits, sel_t.to(logits.device))

    def set_awgn_sigma(self, sigma):
        """Used by the V23 ablation path; ignored by V24."""
        self._awgn_sigma = sigma
