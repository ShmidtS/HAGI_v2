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
from hagi_v4.model.outputs import AuxLosses, ModelOutput, compute_grade_spec_loss, compute_whiteness_loss
from hagi_v4.model.transformer_block import TransformerBlock


def _information_bottleneck_loss(
    h: torch.Tensor,
    targets: torch.Tensor,
    lm_head_weight: torch.Tensor,
    beta: float = 1.0,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Information Bottleneck regularizer: I(X;Z) - beta * I(Y;Z).

    I(X;Z) proxy: hidden state variance (complexity).
    I(Y;Z) proxy: negative cross-entropy (predictive information).
    """
    complexity = h.float().var(dim=(0, 1)).sum()

    flat_h = h.reshape(-1, h.size(-1))
    flat_t = targets.reshape(-1)
    total_ce = h.new_zeros(())
    for i in range(0, flat_h.size(0), chunk_size):
        end = min(i + chunk_size, flat_h.size(0))
        logits_c = F.linear(flat_h[i:end], lm_head_weight)
        total_ce = total_ce + F.cross_entropy(logits_c, flat_t[i:end], reduction="sum")
    ce = total_ce / flat_t.size(0)
    del flat_h, flat_t

    predictive_info = -ce
    return complexity - beta * predictive_info


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
        self.expression = nn.ModuleList(TransformerBlock(m) for _ in range(m.expression_layers))
        self.final_norm = RMSNorm(m.hidden_size, eps=m.norm_eps)
        self.lm_head = nn.Linear(m.hidden_size, m.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.coherence = CoherenceHead(m.cast, m.hidden_size, lm_head=self.lm_head, final_norm=self.final_norm)
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

    def _rope(self, T: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.cfg.model.attention
        return build_rope_cache(T, a.head_dim, a.rope_theta, device, dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        step: int = 0,
    ) -> ModelOutput:
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
            h, _, _ = blk(h, cos, sin)

        h, _ = self.gp2d(h)

        self.msa.clear()
        h, side_info = self.hrm(
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
            h, _, _ = blk(h, cos, sin)

        logits = self.coherence(h)

        aux = AuxLosses()
        if targets is not None:
            aux.deep_supervision = side_info.deep_supervision_loss
            aux.moe_lb = side_info.moe_lb
            aux.msa_lb = side_info.msa_lb
            aux.gdr_router = side_info.gdr_router_loss

            aux.coherence = self.coherence.coherence_loss(h)

            if self.cfg.model.gp2d.use_whiteness_loss and side_info.gp2d_residual is not None:
                aux.whiteness = compute_whiteness_loss(side_info.gp2d_residual)

            if self.cfg.train.w_ib > 0:
                aux.ib = _information_bottleneck_loss(
                    h,
                    targets,
                    self.lm_head.weight,
                    beta=self.cfg.train.ib_beta,
                )

            if (
                self.cfg.model.moe.use_grade_specialization
                and side_info.gdr_gate_probs is not None
                and side_info.moe_router_probs is not None
            ):
                grade_spec = h.new_zeros(())
                for rp in side_info.moe_router_probs:
                    gs = compute_grade_spec_loss(
                        side_info.gdr_gate_probs,
                        rp,
                        num_experts=self.cfg.model.moe.num_experts,
                    )
                    grade_spec = grade_spec + self.cfg.model.moe.grade_specialization_weight * gs
                aux.grade_spec = grade_spec

        return ModelOutput(
            logits=logits,
            hidden=h,
            aux=aux,
            iterations_used=side_info.iterations_used,
        )
