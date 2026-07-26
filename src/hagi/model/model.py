"""HAGI-2 (V28) — codec-first scalable multimodal channel LM.

Main LM path (causal, KV-cacheable):
  source-encode (CAUSAL conv, no future leak)
    -> UNIFIED ternary stack (real GQA + RoPE; water-filling MoE on every
       moe_every-th block; optional sliding-window local channel)
    -> final_norm
    -> factored LM head

Auxiliary (off the main path, never intercepts the LM signal):
  * InformationBottleneck: KL / distortion regularizer on the context hidden.
  * PredictiveRefiner (opt-in): rehabilitated HEP extrinsic refinement of a
    CLONE of the context hidden — strictly off-path (the V25 in-path placement
    deadlocked from-scratch training). EXIT-halt gates the beta-anneal.
  * GroundedInfomax (VICReg + InfoNCE): multimodal joint-embedding alignment.
  * MoE load-balance (Switch CV^2) + routing-entropy capacity maximization.
  * attn entropy floor: anti-collapse (training only).

The body is a SINGLE unified stack (V25 split it into context/expression, an
artifact of the in-path-IB era; with the IB off-path the split only blocks the
KV-cache). Multimodal input, when enabled, is encoded per-modality, compressed
to a fixed prefix via the Q-Former bridge, and prepended to the text sequence
(prefix-LM attention: prefix fully visible, text causal).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from hagi.config import Config
from hagi.model.attention import AttentionConfig
from hagi.model.block import TransformerBlock
from hagi.model.bottleneck import InformationBottleneck
from hagi.model.conv_embedding import ConvEmbedding
from hagi.model.hebbian_ffn import HebbianFFNConfig
from hagi.model.kv_cache import KVCache
from hagi.model.moe import MoESwiGLU
from hagi.model.norms import RMSNorm
from hagi.model.outputs import AuxLosses, ModelOutput


class HAGI(nn.Module):
    """Codec-first multimodal channel LM.

    Args:
        cfg: top-level :class:`hagi.config.Config`.
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
        self._moe_on = bool(body.moe.enabled)

        # ---- Source encoder (factorized, CAUSAL conv) ----
        self.embed = ConvEmbedding(
            vocab_size=m.vocab_size,
            hidden_size=H,
            factor_rank=m.embeddings.factor_rank,
            kernel_size=m.embeddings.kernel_size,
            norm_eps=m.norm_eps,
        )

        attn_cfg = AttentionConfig(
            num_heads=m.attention.num_query_heads,
            num_kv_heads=m.attention.num_kv_heads,
            head_dim=m.attention.head_dim,
            rope_theta=m.attention.rope_theta,
            attn_entropy_floor=cfg.train.attn_entropy_floor,
        )
        ffn_cfg = HebbianFFNConfig(expansion=body.ternary.hebbian_expansion)
        inter = body.moe.intermediate_size or (body.ternary.hebbian_expansion * H)

        # ---- UNIFIED ternary stack (the genuine channel) ----
        from hagi.config import layer_sliding_windows

        per_layer_window = layer_sliding_windows(m)
        moe_every = max(1, body.moe.moe_every) if self._moe_on else 0
        self.blocks = nn.ModuleList()
        for li in range(body.num_layers):
            use_moe_here = self._moe_on and moe_every > 0 and (li % moe_every == (moe_every - 1))
            mixer = None
            if use_moe_here:
                mixer = MoESwiGLU(H, inter, body.moe, m.norm_eps, use_ternary=self._use_ternary)
            block = TransformerBlock(H, attn_cfg, ffn_cfg, m.norm_eps, use_ternary=self._use_ternary, mixer=mixer)
            # Per-layer sliding window (0 = full attention; >0 = local channel).
            block.attn.sliding_window = per_layer_window[li] if per_layer_window else 0
            self.blocks.append(block)

        self.final_norm = RMSNorm(H, eps=m.norm_eps)

        # ---- Auxiliary information bottleneck (off the main path) ----
        self.bottleneck = InformationBottleneck(H, body.bottleneck, m.norm_eps)

        # ---- Factored LM head (independent rank-r factorization) ----
        r = m.embeddings.factor_rank
        self.lm_compress = nn.Linear(H, r, bias=False)
        self.lm_expand = nn.Linear(r, m.vocab_size, bias=False)
        nn.init.normal_(self.lm_compress.weight, std=1.0 / math.sqrt(H))
        nn.init.normal_(self.lm_expand.weight, std=1.0 / math.sqrt(r))

        # ---- Optional multimodal fusion (per-modality + Q-Former + grounded) ----
        self.multimodal_input = None
        if self._mm_on:
            from hagi.model.multimodal import MultimodalFusion

            self.multimodal_input = MultimodalFusion(cfg, text_encoder=self.embed)

        # ---- Optional off-path HEP predictive refinement (opt-in) ----
        self.refinement_head = None
        if m.refinement.enabled:
            from hagi.model.refinement import RefinementHead

            self.refinement_head = RefinementHead(
                H, m.vocab_size, m.refinement, m.norm_eps, use_ternary=self._use_ternary
            )

        self._init_weights()
        for mod in self.modules():
            if hasattr(mod, "set_attn_entropy_floor"):
                mod.set_attn_entropy_floor(cfg.train.attn_entropy_floor)

    def _init_weights(self) -> None:
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                if "lm_compress" in name or "lm_expand" in name:
                    continue
                if name.endswith("out_proj") or name.endswith("W") or name.endswith("down"):
                    continue
                std = 1.0 / math.sqrt(max(1, mod.weight.shape[1]))
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    @property
    def lm_head_weight(self) -> torch.Tensor:
        return self.lm_expand.weight @ self.lm_compress.weight

    def allocate_for_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> list[KVCache]:
        """Allocate a per-layer KV-cache for incremental decoding.

        Returns the list of caches (also attached to each attention layer).
        """
        m = self.cfg.model
        caches: list[KVCache] = []
        for blk in self.blocks:
            cache = KVCache(m.attention.max_seq_len, m.attention.num_kv_heads, m.attention.head_dim, dtype, device)
            blk.attn.attach_cache(cache)
            caches.append(cache)
        return caches

    def reset_cache(self) -> None:
        """Detach (and drop) the per-layer KV-cache and the conv-history cache."""
        for blk in self.blocks:
            blk.attn.detach_cache()
        self.embed.reset_conv_cache()

    def _stack_forward(self, h, attention_mode, prefix_len, soft_beta, positions):
        """Run the unified block stack with grad-checkpointing; sum attn-entropy + route-entropy."""
        entropy_pen = None
        route_entropy_acc = None
        checkpointing = self.training and len(self.blocks) > 1
        moe_lb_acc = h.new_zeros(()) if any(blk.is_moe for blk in self.blocks) else None
        for blk in self.blocks:
            if checkpointing:
                def run(b_in, *, b=blk, am=attention_mode, pl=prefix_len, sb=soft_beta, pos=positions):
                    return b(b_in, attention_mode=am, prefix_len=pl, soft_beta=sb, positions=pos)
                h = ckpt.checkpoint(run, h, use_reentrant=False)
            else:
                h = blk(h, attention_mode=attention_mode, prefix_len=prefix_len, soft_beta=soft_beta, positions=positions)
            pen = getattr(blk, "_last_attn_entropy_penalty", None)
            if pen is not None:
                entropy_pen = pen if entropy_pen is None else entropy_pen + pen
            if blk.is_moe and self.training:
                lb = blk.moe.last_load_balance if blk.moe is not None else None
                if lb is not None and moe_lb_acc is not None:
                    moe_lb_acc = moe_lb_acc + lb
                re = blk.moe.last_routing_entropy if blk.moe is not None else None
                if re is not None:
                    route_entropy_acc = re if route_entropy_acc is None else route_entropy_acc + re
        self._last_attn_entropy_penalty = entropy_pen
        self._last_moe_lb = moe_lb_acc
        self._last_route_entropy = route_entropy_acc
        return h

    def forward(
        self,
        input_ids=None,
        targets=None,
        *,
        prediction_mask=None,
        valid_target_mask=None,
        images=None,
        spectrograms=None,
        attention_mode="causal",
        prefix_len=None,
        soft_beta=None,
        positions=None,
        **_unused,
    ):
        """Forward pass.

        Args:
            input_ids: ``[B, T_text]`` token IDs.
            targets: ``[B, T_text]`` next-token targets (optional).
            prediction_mask: ``[B, T_text]`` positions to score.
            valid_target_mask: ``[B, T_text]`` positions with a valid target.
            images, spectrograms: optional modality inputs (multimodal only).
            attention_mode: causal (default / inference) | bidir | prefix | soft_causal.
            prefix_len: prefix length for prefix mode (multimodal bridge).
            soft_beta: soft-causal decay.
            positions: absolute positions for RoPE (None -> arange/cache offset).

        Returns:
            :class:`ModelOutput` with logits, hidden, aux, ce_loss.
        """
        if input_ids is None:
            raise ValueError("input_ids is required")

        # STAGE 1 — modal source encode.
        if self._mm_on and (images is not None or spectrograms is not None):
            h, modality_ids, mm_info = self.multimodal_input(input_ids, images, spectrograms)
            if prefix_len is None and mm_info["prefix_len"] > 0:
                prefix_len = mm_info["prefix_len"]
                attention_mode = "prefix"
        else:
            h = self.embed(input_ids)
            modality_ids = None
            mm_info = {"prefix_len": 0}

        # STAGE 2 — UNIFIED ternary stack (the genuine channel).
        h_ctx = self._stack_forward(h, attention_mode, prefix_len, soft_beta, positions)

        # STAGE 3 — auxiliary information bottleneck (off-path).
        bn_info = self.bottleneck(h_ctx)

        # STAGE 4 — MAIN LM PATH: final norm + factored head on text positions.
        h_dec = self.final_norm(h_ctx)
        t_text = input_ids.shape[1]
        if self._mm_on and mm_info["prefix_len"] > 0:
            h_text = h_dec[:, mm_info["prefix_len"]:]
        else:
            h_text = h_dec[:, :t_text]

        idx, logits = self._gather_logits(h_text, prediction_mask, valid_target_mask)
        ce_loss = self._ce(idx, logits, targets)

        aux = AuxLosses()
        aux.rate = bn_info["rate"]
        aux.distortion = bn_info["distortion"]
        aux.moe_lb = getattr(self, "_last_moe_lb", None)
        aux.route_entropy = getattr(self, "_last_route_entropy", None)
        aux.attn_entropy = getattr(self, "_last_attn_entropy_penalty", None)

        # STAGE 4b — off-path HEP predictive refinement (opt-in). Runs on a CLONE
        # of h_ctx; the main logits come from the UN-refined h_ctx (V25 lesson:
        # in-path refinement deadlocks from-scratch training). The only gradient
        # into the refinement branch is the auxiliary refinement loss.
        if self.refinement_head is not None and self.training:
            ref_loss, _h_refined = self.refinement_head(
                h_ctx,
                main_logits=logits if idx is not None else None,
                targets=targets if targets is not None else None,
                prediction_indices=idx,
                refine_weight=self.cfg.train.w_refine,
            )
            aux.refinement = ref_loss
            aux.exit_novelty = self.refinement_head.refiner.novelty()
        elif self.refinement_head is not None:
            # eval: still produce the diagnostic without building the loss graph.
            self.refinement_head.refiner(h_ctx)
            aux.exit_novelty = self.refinement_head.refiner.novelty()

        # STAGE 5 — grounded infomax (multimodal, off-path). Computed on h_ctx
        # over the full sequence (prefix + text) so the modality-pooled
        # embeddings reflect the channel output, not the source encoder alone.
        if self._mm_on and modality_ids is not None:
            vicreg, infonce = self.multimodal_input.grounded(h_ctx, modality_ids)
            aux.vicreg = vicreg
            aux.infonce = infonce

        return ModelOutput(
            logits=logits, hidden=h_text, aux=aux, ce_loss=ce_loss, prediction_indices=idx,
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
