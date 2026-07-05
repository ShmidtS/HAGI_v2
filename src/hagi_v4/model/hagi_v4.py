"""HAGI V4 — top-level model. Pure orchestration.

Pipeline: mask -> embed -> perception -> GP2D -> refinement (4 iters)
-> expression -> output (full plane prediction).

Bidirectional attention throughout (no causal mask). Masked CE training
(predict masked positions, not next-token). Iterative geometric refinement
with gradient checkpointing (no h.detach).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hagi_v4.config import HAGIv4Config
from hagi_v4.model.cast import CoherenceHead
from hagi_v4.model.gdr import GradeDecomposedRecurrence
from hagi_v4.model.gp2d import GeometricProduct2D
from hagi_v4.model.hrm import RefinementCore
from hagi_v4.model.msa import MSAModule
from hagi_v4.model.norms import RMSNorm, build_rope_cache
from hagi_v4.model.outputs import InferenceOutput, TrainOutput
from hagi_v4.model.transformer_block import TransformerBlock
from hagi_v4.train.losses import information_bottleneck_loss


class HAGIv4(nn.Module):
    def __init__(self, cfg: HAGIv4Config) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        self.embed = nn.Embedding(m.vocab_size, m.hidden_size)
        self.mask_embed = nn.Parameter(torch.zeros(m.hidden_size))
        self.perception = nn.ModuleList(TransformerBlock(m) for _ in range(m.perception_layers))
        self.gp2d = GeometricProduct2D(m.gp2d, m.hidden_size)
        self.reasoning = nn.ModuleList(TransformerBlock(m) for _ in range(m.reasoning_layers))
        self.gdr = GradeDecomposedRecurrence(m.gdr, m.hidden_size)
        self.hrm = RefinementCore(m.hrm, m.refinement, m.hidden_size)
        self.hrm.set_max_steps(cfg.train.max_steps)
        self.msa = MSAModule(m.msa, m.hidden_size)
        self.coherence = CoherenceHead(m.cast, m.hidden_size)
        self.expression = nn.ModuleList(TransformerBlock(m) for _ in range(m.expression_layers))
        self.final_norm = RMSNorm(m.hidden_size, eps=m.norm_eps)
        self.lm_head = nn.Linear(m.hidden_size, m.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.mask_embed.fill_(0.0)

    def _rope(self, T: int, device, dtype):
        a = self.cfg.model.attention
        return build_rope_cache(T, a.head_dim, a.rope_theta, device, dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        step: int = 0,
    ) -> TrainOutput | InferenceOutput:
        B, T = input_ids.shape
        h = self.embed(input_ids)
        if mask is not None:
            h = torch.where(
                mask.unsqueeze(-1),
                self.mask_embed.expand(B, T, -1),
                h,
            )
        cos, sin = self._rope(T, h.device, h.dtype)

        for blk in self.perception:
            h, _ = blk(h, cos, sin)

        h = self.gp2d(h)

        self.msa.clear()
        h, losses, iterations_used = self.hrm(
            h,
            self.reasoning,
            self.gdr,
            self.gp2d,
            self.msa,
            cos,
            sin,
            targets,
            self.lm_head.weight,
            self.final_norm,
            training=self.training,
            mask=mask,
            step=step,
        )

        for blk in self.expression:
            h, _ = blk(h, cos, sin)

        h_normed = self.final_norm(h)

        if targets is not None:
            if mask is not None and mask.any():
                h_masked = h_normed[mask]
                t_masked = targets[mask]
                ce_loss = F.cross_entropy(
                    F.linear(h_masked, self.lm_head.weight),
                    t_masked,
                )
                del h_masked, t_masked
            else:
                flat_h = h_normed.reshape(-1, h_normed.size(-1))
                flat_t = targets.reshape(-1)
                chunk = 4096
                total_ce = h_normed.new_zeros(())
                for i in range(0, flat_h.size(0), chunk):
                    end = min(i + chunk, flat_h.size(0))
                    logits_c = F.linear(flat_h[i:end], self.lm_head.weight)
                    total_ce = total_ce + F.cross_entropy(logits_c, flat_t[i:end], reduction="sum")
                ce_loss = total_ce / flat_t.size(0)
                del flat_h, flat_t

            unmask_loss = h.new_zeros(())

            coh_loss = self.coherence.coherence_loss(h)

            whiteness_loss = self.gp2d.whiteness_loss() if self.cfg.model.gp2d.use_whiteness_loss else h.new_zeros(())

            ib_loss = h.new_zeros(())
            if self.cfg.train.w_ib > 0:
                ib_loss = information_bottleneck_loss(
                    h,
                    targets,
                    self.lm_head.weight,
                    beta=self.cfg.train.ib_beta,
                )

            grade_spec_loss = h.new_zeros(())
            if self.cfg.model.moe.use_grade_specialization and self.gdr._last_gate is not None:
                gate_flat = self.gdr._last_gate.reshape(-1, 4)
                for blk in self.reasoning:
                    gs = blk.moe.grade_specialization_loss(gate_flat)
                    grade_spec_loss = grade_spec_loss + gs

            total = (
                ce_loss
                + unmask_loss
                + self.cfg.train.w_moe_aux * losses["moe_aux"]
                + self.cfg.train.w_gdr_router * losses["gdr_router"]
                + 0.01 * losses["msa_lb"]
                + losses["deep_supervision"]
                + self.cfg.train.w_coherence * coh_loss
                + self.cfg.train.w_whiteness * whiteness_loss
                + self.cfg.train.w_ib * ib_loss
                + self.cfg.train.w_grade_specialization * grade_spec_loss
            )
            return TrainOutput(
                loss=total,
                moe_aux_loss=losses["moe_aux"],
                gdr_router_loss=losses["gdr_router"],
                coherence_loss=coh_loss,
                deep_supervision_loss=losses["deep_supervision"],
                hidden=h,
                mask=mask if mask is not None else torch.zeros_like(input_ids, dtype=torch.bool),
            )

        logits = F.linear(h_normed, self.lm_head.weight)
        return InferenceOutput(
            logits=logits,
            hidden=h,
            iterations_used=iterations_used,
        )
