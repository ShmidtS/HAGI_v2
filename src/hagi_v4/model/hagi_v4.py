"""HAGI V18 — Strict Shannon Source-Channel Separation codec LM.

V21 refactor: the monolithic 787-LOC class has been decomposed into a
``codec/`` package with four specialised modules. This file is now a thin
orchestrator that wires the four stages together and owns only the
global concerns (weight init, distill alignment, forward orchestration,
LM head assembly). No behavioural changes vs V18 — pure decomposition.

Pipeline:
  Source Encode (ConvEmbed + sinusoidal PE + AttentionBlock × N_perc + LayerScale + rate_down)
    -> z_sys in R^C
  Channel Encode (SparseParity + BlockInterleaver) -> codeword
  Physical Channel (AWGN + erasure on systematic only)
  Channel Decode (pure LDPC BP: syndrome -> H^T back-projection -> gated correction)
  Source Decode (rate_up + AttentionBlock × N_expr + final_norm + factored tied lm_head)

V18 changes vs V17 (see docs/ARCHITECTURE.md):
  - No source_skip_scale bypass (strict SCS)
  - No FreqBlock in decoder (pure LDPC BP)
  - No bottleneck_norm (LayerScale only)
  - No Lorentz sphere, HARQ buffer, EXIT halt, mutation branch, multimodal
  - rate_up.expand init N(0, 1/sqrt(C)) for cold-start gradient flow
  - corr_gate_w/b init neutral (0.1, 0.0)
  - Channel always open (no channel_open schedule)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.algebra.clifford import geometric_product
from hagi_v4.config import HAGIv4Config
from hagi_v4.model.codec import (
    ChannelDecoder,
    ChannelEncoder,
    SourceDecoder,
    SourceEncoder,
)
from hagi_v4.model.codec_contracts import (
    CodecShapeConfig,
    DecodeState,
    InferenceShapeConfig,
    SemanticMaskBatch,
    TurboDecodeConfig,
)
from hagi_v4.model.contrastive import ContrastiveAlignment
from hagi_v4.model.multimodal_input import MultimodalInput
from hagi_v4.model.outputs import AuxLosses, ModelOutput, compute_parity_diversity_loss, compute_whiteness_loss

if TYPE_CHECKING:
    from hagi_v4.inference.spectral_cache import SpectralCache


class HAGIv4(nn.Module):
    """HAGI V18 — Strict Source-Channel Separation codec language model.

    Thin orchestrator over the ``codec/`` package. Owns only:
      - codec shape / turbo / inference configs
      - the four codec stage modules (SourceEncoder, ChannelEncoder,
        ChannelDecoder, SourceDecoder)
      - distill_align projection (lazy, for distillation)
      - global weight init (needs full module tree)
      - forward() orchestration and LM head assembly
    """

    def __init__(self, cfg: HAGIv4Config) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        self.codec_shape = CodecShapeConfig.from_hagi_config(cfg)
        self.turbo_config = TurboDecodeConfig.from_hagi_config(cfg)
        self.inference_config = InferenceShapeConfig.from_hagi_config(cfg)
        H = m.hidden_size
        C = m.core_hidden_size
        self._H = H
        self._C = C

        # ---- Source encoder ----
        self.source_encoder = SourceEncoder(cfg)

        # ---- Channel encoder ----
        # core_mask_embed is owned by SourceEncoder but needed by
        # ChannelEncoder for erasure fill-in; pass by reference.
        self.channel_encoder = ChannelEncoder(
            cfg,
            self.codec_shape,
            core_mask_embed=self.source_encoder.core_mask_embed,
        )

        # ---- Channel decoder ----
        # Shared Tanner-graph params come from ChannelEncoder's parity_encoder;
        # pass them so the decoder's SparseParityChecker aliases the same
        # tensors (fixed mask + frozen parity_base + learnable edge_log_scale).
        self.channel_decoder = ChannelDecoder(
            cfg,
            self.codec_shape,
            interleaver=self.channel_encoder.interleaver,
            shared_parity_weights=self.channel_encoder.parity_encoder.edge_log_scale,
            shared_sparse_mask=self.channel_encoder.parity_encoder.sparse_mask,
            shared_edge_log_scale=self.channel_encoder.parity_encoder.edge_log_scale,
            shared_parity_base=self.channel_encoder.parity_encoder.parity_base,
        )

        # ---- Source decoder ----
        # rate_up is owned by SourceEncoder but applied in SourceDecoder;
        # pass by reference so both stages share the same FactoredLinear.
        # token_compress_weight enables weight tying: lm_expand.weight is
        # aliased to embed.token_compress.weight when use_conv_embedding.
        self.source_decoder = SourceDecoder(
            cfg,
            rate_up=self.source_encoder.rate_up,
            token_compress_weight=self.source_encoder.embed.token_compress.weight
            if m.embeddings.use_conv_embedding
            else None,
        )
        self.tie_source_codebook: bool = self.source_decoder.tie_source_codebook
        if self.tie_source_codebook:
            # Verify the tying was applied (SourceDecoder constructor aliases
            # lm_expand.weight to token_compress.weight). Asserted here so a
            # future refactor that breaks the reference chain fails loudly.
            assert self.source_decoder.lm_expand.weight is self.source_encoder.embed.token_compress.weight, (
                "weight tying broken: lm_expand.weight is not token_compress.weight"
            )

        # V13: distill_align allocated lazily when distillation is enabled.
        if cfg.train.distill_enabled:
            teacher_hidden = cfg.train.distill_teacher_hidden_size
            self.distill_align = nn.Identity() if teacher_hidden == H else nn.Linear(teacher_hidden, H, bias=False)
        else:
            self.distill_align = None

        self.contrastive: ContrastiveAlignment | None = None
        if cfg.train.w_contrastive > 0.0:
            self.contrastive = ContrastiveAlignment(hidden_size=C, temperature=0.07)

        self.multimodal_input: MultimodalInput | None = None
        if m.multimodal.enabled:
            self.multimodal_input = MultimodalInput(cfg, text_encoder=self.source_encoder.embed)

        self._init_weights()

        # V22: set attention entropy floor on all attention blocks
        attn_floor = getattr(cfg.train, "attn_entropy_floor", 0.5)
        for mod in self.modules():
            if hasattr(mod, "set_attn_entropy_floor"):
                mod.set_attn_entropy_floor(attn_floor)

    @property
    def embed(self):
        """Compatibility shim: expose source_encoder.embed for callers
        (distillation, diagnostics) that expect the legacy attribute."""
        return self.source_encoder.embed

    @property
    def lm_head_weight(self) -> torch.Tensor:
        """Effective [V, H] output projection for compatibility callers."""
        return self.source_decoder.lm_expand.weight @ self.source_decoder.lm_compress.weight

    def _init_weights(self) -> None:
        v23_zero_init_names = {"hrm_z_h_to_hidden", "hrm_z_l_to_hidden"}
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                if any(n in name for n in v23_zero_init_names):
                    continue
                if "mut_" in name or mod is self.source_decoder.lm_compress:
                    continue
                if self.tie_source_codebook and mod is self.source_decoder.lm_expand:
                    continue
                if name.endswith("out_proj") or name.endswith("ffn.w_out") or mod is self.source_encoder.rate_up.expand:
                    nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
                    continue
                std = 1.0 / math.sqrt(max(1, mod.weight.shape[1]))
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                if mod is self.source_encoder.embed or mod is getattr(
                    self.source_encoder.embed, "token_compress", None
                ):
                    continue
                std = 1.0 / math.sqrt(max(1, mod.weight.shape[1]))
                nn.init.normal_(mod.weight, mean=0.0, std=std)

    def train(self, mode: bool = True) -> HAGIv4:
        return super().train(mode)

    def _clifford_cross_modal_mix(self, h: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        """Mix cross-modal representations using Cl(3,0,0) geometric product.

        Computes the geometric product of mean per-modality representations
        and adds a small mixed contribution back to all positions.
        """
        B, T, H = h.shape
        if H % 8 != 0:
            return h
        n_mv = H // 8
        h_mv = h.view(B, T, n_mv, 8)
        mixed = h_mv
        for mod_a in range(3):
            for mod_b in range(mod_a + 1, 3):
                mask_a = modality_ids == mod_a
                mask_b = modality_ids == mod_b
                if not (mask_a.any() and mask_b.any()):
                    continue
                mean_a = h_mv[mask_a].mean(dim=0)
                mean_b = h_mv[mask_b].mean(dim=0)
                gp = geometric_product(mean_a, mean_b)
                mixed = mixed + gp.unsqueeze(0).unsqueeze(0) * 0.1
        return mixed.view(B, T, H)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        *,
        semantic_unknown_mask: torch.Tensor,
        prediction_mask: torch.Tensor,
        valid_target_mask: torch.Tensor,
        physical_corruption_mask: torch.Tensor,
        step: int = 0,
        cached_p: torch.Tensor | None = None,
        cache: SpectralCache | None = None,
        awgn_sigma: float = 0.0,
        images: torch.Tensor | None = None,
        spectrograms: torch.Tensor | None = None,
        refinement_iterations: int | None = None,
        attention_mode: str = "bidir",
        prefix_len: torch.Tensor | int | None = None,
        soft_beta: float | None = None,
    ) -> ModelOutput:
        """V20: attention_mode selects how perception layers attend.

        - "bidir":  full bidirectional (default; masked-LM training)
        - "prefix": GLM-style prefix-LM (prefix_len bidir, rest causal)
        - "causal": pure GPT-style (AR inference)
        - "soft_causal": V22 smooth bidir→causal blending (soft_beta controls sharpness)

        The source decoder (expression stack) always runs in causal mode
        to match the LM head's left-to-right query pattern at inference.
        """
        if input_ids is None:
            raise ValueError("input_ids is required")
        if refinement_iterations is None and self.training:
            train_bp = getattr(self.cfg.train, "bp_iterations", None)
            if train_bp is None:
                train_bp = 3
        masks = SemanticMaskBatch(
            semantic_unknown_mask,
            prediction_mask,
            valid_target_mask,
            physical_corruption_mask,
        )
        masks.validate(input_ids, None)
        state = cache.to_decode_state() if cache is not None else DecodeState(kalman_p=cached_p)
        if cache is not None and state.kalman_p is None:
            state.kalman_p = cached_p

        phys_mask = physical_corruption_mask
        semantic_unknown_eff = semantic_unknown_mask
        if self.multimodal_input is not None and (images is not None or spectrograms is not None):
            h_mm, modality_ids, lengths = self.multimodal_input(input_ids, images, spectrograms)
            h_mm = self._clifford_cross_modal_mix(h_mm, modality_ids)
            B = h_mm.shape[0]
            T_text = input_ids.shape[1]
            T_total = h_mm.shape[1]
            if T_total > T_text:
                extra = torch.zeros((B, T_total - T_text), dtype=torch.bool, device=input_ids.device)
                semantic_unknown_eff = torch.cat([semantic_unknown_mask, extra], dim=1)
                phys_mask = torch.cat([physical_corruption_mask, extra], dim=1)
            encoded = self.source_encoder.forward(
                None,
                semantic_unknown_eff,
                cache,
                pre_encoded_h=h_mm,
                attention_mode=attention_mode,
                prefix_len=prefix_len,
                soft_beta=soft_beta,
            )
        else:
            encoded = self.source_encoder.forward(
                input_ids,
                semantic_unknown_mask,
                cache,
                attention_mode=attention_mode,
                prefix_len=prefix_len,
                soft_beta=soft_beta,
            )
        ch_encoded = self.channel_encoder.apply_erasure(
            self.channel_encoder.forward(encoded),
            phys_mask,
            awgn_sigma,
        )
        # V19: inform the LDPC decoder of the current AWGN sigma so its
        # Kalman validation gate uses the correct Mahalanobis threshold.
        self.channel_decoder.set_awgn_sigma(awgn_sigma if awgn_sigma > 0.0 else None)
        decoded = self.channel_decoder.forward(
            ch_encoded,
            encoded,
            state,
            self.training,
            bp_iterations=getattr(self.cfg.train, "bp_iterations", None),
            refinement_iterations=refinement_iterations,
        )
        h = self.source_decoder.forward(decoded, attention_mode="causal")

        if cache is not None:
            cache.update_decode_state(decoded.state)

        side_info = decoded.side_info
        rd_loss = (encoded.systematic.float() - decoded.latent.float()).pow(2).mean().to(h.dtype)

        h_normed = self.source_decoder.final_norm(h)
        selected = prediction_mask & valid_target_mask
        prediction_indices = selected.flatten().nonzero(as_tuple=False).squeeze(-1)
        prediction_hidden = h_normed[:, : input_ids.shape[1]]
        selected_hidden = prediction_hidden.flatten(0, 1).index_select(0, prediction_indices.to(h_normed.device))
        logits = self.source_decoder.lm_expand(
            self.source_decoder.lm_compress(selected_hidden.to(self.source_decoder.lm_compress.weight.device))
        ).to(h_normed.device)
        if targets is not None:
            if prediction_indices.numel() == 0:
                raise ValueError("prediction_mask must select at least one target during training")
            selected_targets = targets.flatten().index_select(0, prediction_indices.to(targets.device))
            ce = F.cross_entropy(logits, selected_targets.to(logits.device))
        else:
            ce = None

        aux = AuxLosses()
        if self.contrastive is not None and self.training:
            aux.contrastive = self.contrastive(encoded.systematic, semantic_unknown_eff.long())
        if targets is not None:
            if side_info.get("parity_strength") is not None:
                aux.parity = side_info["parity_strength"]
            if self.codec_shape.use_whiteness_loss and side_info.get("parity_residual") is not None:
                pilot_mask = self.source_encoder._get_pilot_mask(
                    side_info["parity_residual"].shape[1],
                    side_info["parity_residual"].device,
                )
                valid = pilot_mask.unsqueeze(0).expand(side_info["parity_residual"].shape[0], -1)
                valid = valid & ~phys_mask.to(valid.device)
                aux.whiteness = compute_whiteness_loss(side_info["parity_residual"], valid)
            aux.correction_alignment = side_info["correction_alignment"].to(h.dtype)
            aux.rate_distortion = rd_loss

            if getattr(self.cfg.train, "w_parity_diversity", 0.0) > 0.0:
                aux.parity_diversity = compute_parity_diversity_loss(self.channel_encoder.parity_encoder.masked_weights)

            if self.training and phys_mask is not None and phys_mask.any():
                z_clean = encoded.systematic
                z_erased = ch_encoded.codeword[..., : self._C]
                recovery_error = (z_clean.float() - z_erased.float()).pow(2)
                erased_error = recovery_error[phys_mask.to(recovery_error.device)]
                if erased_error.numel() > 0:
                    aux.parity_recovery = erased_error.mean().to(h.dtype)

        # V22: attention entropy penalty
        enc_pen = getattr(self.source_encoder, "_last_attn_entropy_penalty", None)
        dec_pen = getattr(self.source_decoder, "_last_attn_entropy_penalty", None)
        if enc_pen is not None:
            aux.attn_entropy = enc_pen
        if dec_pen is not None:
            aux.attn_entropy = dec_pen if aux.attn_entropy is None else aux.attn_entropy + dec_pen

        return ModelOutput(
            logits=logits,
            hidden=h,
            aux=aux,
            ce_loss=ce,
            iterations_used=side_info.get("iterations_used"),
            prediction_indices=prediction_indices,
        )
