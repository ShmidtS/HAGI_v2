"""HAGI — ternary RD-channel causal language model.

The model is a causal autoregressive LM framed as a communication channel.

Main LM path (PROVEN to learn — causal_ce 10.0->4.3 in 300 steps on tinystories):
  source-encode (CAUSAL conv, no future leak)
    -> ternary context stack (the genuine channel)
    -> causal expression stack
    -> final_norm
    -> factored LM head

Auxiliary (off the main path, never intercepts the LM signal):
  * variational InformationBottleneck: KL/distortion/perception regularizer on
    h_ctx. Inserting it into the main path deadlocked from-scratch training.
  * PredictiveDecoder: extrinsic error highway — opt-in (body.predictive.enabled).

The architecture is causal-first by design. Inserting the IB+PD into the main
path, or using a non-causal embedding conv, both produce prompt-independent
garbage at inference — see docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from hagi_v4.config import Config
from hagi_v4.model.block import AttentionConfig, HebbianFFNConfig, TransformerBlock
from hagi_v4.model.bottleneck import BottleneckConfig, InformationBottleneck
from hagi_v4.model.conv_embedding import ConvEmbedding
from hagi_v4.model.norms import RMSNorm
from hagi_v4.model.outputs import AuxLosses, ModelOutput


class HAGI(nn.Module):
    """Ternary RD-channel causal LM.

    Args:
        cfg: top-level :class:`hagi_v4.config.Config`.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        body = m.body
        H = m.hidden_size
        C = m.core_hidden_size
        self._H, self._C = H, C
        self.inference_config = type("Inf", (), {"vocab_size": m.vocab_size})()
        self._use_ternary = bool(body.ternary.use_ternary)
        self._mm_on = bool(m.multimodal.enabled)

        # ---- Source encoder (factorized, CAUSAL conv) ----
        self.embed = ConvEmbedding(
            vocab_size=m.vocab_size,
            hidden_size=H,
            factor_rank=m.embeddings.factor_rank,
            kernel_size=m.embeddings.kernel_size,
            norm_eps=m.norm_eps,
        )
        self.semantic_unknown_embed = nn.Parameter(torch.empty(H))
        nn.init.normal_(self.semantic_unknown_embed, std=1.0 / math.sqrt(H))

        # Sinusoidal pilot positional encoding (position-only, no future leak).
        self._pilot_pos_cache: dict[tuple[int, int], torch.Tensor] = {}

        attn_cfg = AttentionConfig(
            num_heads=max(1, m.attention.num_query_heads),
            head_dim=m.attention.head_dim,
            rope_theta=m.attention.rope_theta,
            attn_entropy_floor=cfg.train.attn_entropy_floor,
        )
        ffn_cfg = HebbianFFNConfig(
            expansion=body.ternary.hebbian_expansion, dropout=body.ternary.hebbian_dropout
        )

        # ---- Ternary body: context (perception) + expression stacks ----
        self.context_stack = nn.ModuleList(
            TransformerBlock(H, attn_cfg, ffn_cfg, m.norm_eps, use_ternary=self._use_ternary)
            for _ in range(body.context_layers)
        )
        self.expression = nn.ModuleList(
            TransformerBlock(H, attn_cfg, ffn_cfg, m.norm_eps, use_ternary=self._use_ternary)
            for _ in range(body.expression_layers)
        )

        # ---- Auxiliary information bottleneck (off the main path) ----
        bn_cfg = BottleneckConfig(
            dim=C,
            ib_beta=body.ib_beta,
            distortion_weight=body.distortion_weight,
            perception_weight=body.perception_weight,
            kl_free_bits=body.kl_free_bits,
            logvar_clamp=body.logvar_clamp,
            distortion_eps=body.distortion_eps,
        )
        self.bottleneck = InformationBottleneck(H, bn_cfg, m.norm_eps)

        # ---- Optional predictive decoder (off the main path) ----
        self.predictive_decoder = None
        self.rate_up = None
        self._predictive_in_path = bool(body.predictive.enabled)
        if self._predictive_in_path:
            from hagi_v4.model.predictive import PredictiveConfig, PredictiveDecoder

            pd_cfg = PredictiveConfig(
                train_iterations=body.predictive.train_iterations,
                infer_iterations=body.predictive.infer_iterations,
                convergence_threshold=body.predictive.convergence_threshold,
                update_hidden=body.predictive.update_hidden,
            )
            self.predictive_decoder = PredictiveDecoder(
                C, H, pd_cfg, m.norm_eps,
                use_kalman_blend=body.predictive.use_kalman_blend,
                use_ternary=self._use_ternary,
                hep_enabled=body.predictive.hep_enabled,
            )
            if self._use_ternary:
                from hagi_v4.model.ternary import BitLinear

                self.rate_up = BitLinear(C, H, bias=False)
            else:
                self.rate_up = nn.Linear(C, H, bias=False)
            nn.init.normal_(self.rate_up.weight, std=1.0 / math.sqrt(C))

        self.final_norm = RMSNorm(H, eps=m.norm_eps)

        # ---- Factored LM head (independent rank-r factorization) ----
        r = m.embeddings.factor_rank
        self.lm_compress = nn.Linear(H, r, bias=False)
        self.lm_expand = nn.Linear(r, m.vocab_size, bias=False)
        nn.init.normal_(self.lm_compress.weight, std=1.0 / math.sqrt(H))
        nn.init.normal_(self.lm_expand.weight, std=1.0 / math.sqrt(r))

        # ---- Optional multimodal fusion ----
        self.multimodal_input = None
        if self._mm_on:
            from hagi_v4.model.multimodal import MultimodalFusion

            self.multimodal_input = MultimodalFusion(cfg, text_encoder=self.embed)

        self._init_weights()
        for mod in self.modules():
            if hasattr(mod, "set_attn_entropy_floor"):
                mod.set_attn_entropy_floor(cfg.train.attn_entropy_floor)

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
        """Run a block stack with grad-checkpointing; sum the attn-entropy penalty."""
        entropy_pen = None
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
        images=None,
        spectrograms=None,
        attention_mode="causal",
        prefix_len=None,
        soft_beta=None,
        **_unused,
    ):
        """Forward pass.

        The main path is context-stack -> expression-stack -> LM head. The IB
        runs as an auxiliary regularizer on h_ctx; the optional predictive
        decoder runs only when ``body.predictive.enabled``. ``attention_mode``
        is normally ``"causal"`` (matches inference); training may mix in
        ``"soft_causal"`` / ``"bidir"`` for a denser representation gradient.
        """
        if input_ids is None:
            raise ValueError("input_ids is required")

        # STAGE 1 — modal source encode.
        if self._mm_on and (images is not None or spectrograms is not None):
            h, _mod_ids = self.multimodal_input(input_ids, images, spectrograms)[:2]
            h = h + (1.0 / math.sqrt(self._H)) * self._pilot_pe(h.shape[1], self._H, h.device).to(h.dtype).unsqueeze(0)
        else:
            h = self._encode_text(input_ids, semantic_unknown_mask)
        T_text = input_ids.shape[1]

        # STAGE 2 — ternary context stack (the genuine channel).
        h_ctx = self._stack_forward(h, self.context_stack, attention_mode, prefix_len, soft_beta)

        # STAGE 3 — auxiliary information bottleneck: rate/distortion/perception
        # on h_ctx as KL regularization. Does NOT intercept the LM signal.
        _, bn_info = self.bottleneck(h_ctx)

        # STAGE 4 — MAIN LM PATH: causal expression stack directly on h_ctx.
        if self._predictive_in_path:
            z = bn_info["z"]
            z_refined, pred_info = self.predictive_decoder(z, h_ctx, self.training)
            h_dec = self.rate_up(z_refined)
            h_dec = self._stack_forward(h_dec, self.expression, attention_mode="causal")
            iters_used = pred_info["iterations_used"]
        else:
            h_dec = self._stack_forward(h_ctx, self.expression, attention_mode="causal")
            iters_used = torch.zeros(h_ctx.shape[:2], dtype=torch.long, device=h_ctx.device)
        h_dec = self.final_norm(h_dec)

        # STAGE 5 — decision device: slice text positions, factored LM head.
        h_text = h_dec[:, :T_text]
        idx, logits = self._gather_logits(h_text, prediction_mask, valid_target_mask)
        ce_loss = self._ce(idx, logits, targets)

        aux = AuxLosses()
        aux.rate = bn_info["rate"]
        aux.distortion = bn_info["distortion"]
        aux.perception = bn_info["perception"]
        aux.attn_entropy = getattr(self, "_last_attn_entropy_penalty", None)

        return ModelOutput(
            logits=logits, hidden=h_text, aux=aux, ce_loss=ce_loss,
            iterations_used=iters_used, prediction_indices=idx,
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


# Backwards-compatible alias (many entry points import HAGIv4).
HAGIv4 = HAGI
