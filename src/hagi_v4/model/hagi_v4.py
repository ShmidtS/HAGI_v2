"""HAGI V5 — Codec Language Model.

Architecture as communication channel (Shannon separation theorem):

  Source Encoder: embed → perception (H) → bottleneck_down (H→H/2)
  Channel Encoder: GP2D parity in compressed space (H/2)
  Channel: adaptive erasure (masking)
  Channel Decoder: HRM iterative belief propagation in compressed space (H/2)
  Source Decoder: bottleneck_up (H/2→H) → expression (H) → lm_head (H→V)

Key insight: the refinement loop (7 layers × 4 iterations = 28 layer-apps,
the most expensive part) operates in compressed space H/2=288.
Only perception (2 layers) and expression (2 layers) run at full H=576.
This gives 4x compute reduction and 2x memory reduction in the core.

The bottleneck IS the information bottleneck — no separate IB loss needed.
Deep supervision uses a lightweight core_lm_head (H/2→V) projected directly
from compressed space, avoiding full H materialization at intermediate stages.
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


class HAGIv4(nn.Module):
    def __init__(self, cfg: HAGIv4Config) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        H = m.hidden_size
        C = m.core_hidden_size

        self.embed = nn.Embedding(m.vocab_size, H)
        self.mask_embed = nn.Parameter(torch.randn(H) * 0.02)

        self.perception = nn.ModuleList(TransformerBlock(m, hidden_size=H) for _ in range(m.perception_layers))

        self.bottleneck_down = nn.Linear(H, C, bias=False)
        self.bottleneck_norm = RMSNorm(C, eps=m.norm_eps)
        self.bottleneck_up = nn.Linear(C, H, bias=False)

        self.core_mask_embed = nn.Parameter(torch.randn(C) * 0.02)
        self.gp2d = GeometricProduct2D(m.gp2d, C)
        self.reasoning = nn.ModuleList(TransformerBlock(m, hidden_size=C) for _ in range(m.reasoning_layers))
        self.gdr = GradeDecomposedRecurrence(m.gdr, C)
        self.hrm = RefinementCore(m.hrm, m.refinement, C)
        self.hrm.set_max_steps(cfg.train.max_steps)
        self.msa = MSAModule(m.msa, C)

        self.core_lm_head = nn.Linear(C, m.vocab_size, bias=False)
        nn.init.normal_(self.core_lm_head.weight, mean=0.0, std=0.02)

        self.expression = nn.ModuleList(TransformerBlock(m, hidden_size=H) for _ in range(m.expression_layers))
        self.final_norm = RMSNorm(H, eps=m.norm_eps)
        self.lm_head = nn.Linear(H, m.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.coherence = CoherenceHead(m.cast, H, lm_head=self.lm_head, final_norm=self.final_norm)
        self._init_weights()

    def _init_weights(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)

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
        H = self.cfg.model.hidden_size
        C = self.cfg.model.core_hidden_size

        h = self.embed(input_ids)
        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), self.mask_embed.expand(B, T, -1), h)

        cos, sin = self._rope(T, h.device, h.dtype)

        for blk in self.perception:
            h, _, _ = blk(h, cos, sin)

        z = self.bottleneck_down(h)
        z = self.bottleneck_norm(z)

        if mask is not None:
            z = torch.where(mask.unsqueeze(-1), self.core_mask_embed.expand(B, T, -1), z)

        self.msa.clear()
        z, side_info = self.hrm(
            z,
            self.reasoning,
            self.gdr,
            self.gp2d,
            self.msa,
            cos,
            sin,
            targets,
            self.core_lm_head.weight,
            training=self.training,
            mask=mask,
            step=step,
            extrinsic_alpha=self.cfg.model.refinement.extrinsic_alpha,
            convergence_threshold=self.cfg.model.refinement.convergence_threshold,
            use_convergence_halt=self.cfg.model.refinement.use_convergence_halt,
        )

        h = self.bottleneck_up(z)

        for blk in self.expression:
            h, _, _ = blk(h, cos, sin)

        h_normed = self.final_norm(h)
        if targets is not None and mask is not None and mask.any():
            h_masked = h_normed[mask]
            t_masked = targets[mask]
            logits_masked = F.linear(h_masked, self.lm_head.weight)
            ce = F.cross_entropy(logits_masked, t_masked)
            logits = None
        elif targets is not None:
            logits = F.linear(h_normed, self.lm_head.weight)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        else:
            logits = F.linear(h_normed, self.lm_head.weight)
            ce = None

        aux = AuxLosses()
        if targets is not None:
            aux.deep_supervision = side_info.deep_supervision_loss
            aux.moe_lb = side_info.moe_lb
            aux.msa_lb = side_info.msa_lb
            aux.gdr_router = side_info.gdr_router_loss
            aux.coherence = self.coherence.coherence_loss(h)

            if self.cfg.model.gp2d.use_whiteness_loss and side_info.gp2d_residual is not None:
                aux.whiteness = compute_whiteness_loss(side_info.gp2d_residual)

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

            if side_info.parity_strength is not None:
                aux.parity = side_info.parity_strength

            if side_info.extrinsic_norms and len(side_info.extrinsic_norms) > 1:
                aux.extrinsic_info = h.new_tensor(sum(side_info.extrinsic_norms) / len(side_info.extrinsic_norms))

            if side_info.iterations_used is not None:
                aux.efficiency = side_info.iterations_used.float().mean()

        return ModelOutput(
            logits=logits,
            hidden=h,
            aux=aux,
            ce_loss=ce,
            iterations_used=side_info.iterations_used,
        )
