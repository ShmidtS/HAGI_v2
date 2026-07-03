"""Grade-Decomposed Recurrence (GDR) — HAGI V4 core novel mechanism.

The hidden state is split into Clifford grades with distinct update dynamics:

    scalar    (64)  : confidence/resolution  — slow   (momentum ~0.8)
    vector    (96)  : entities/concepts      — medium (momentum ~0.5)
    bivector  (96)  : relations              — fast   (full update)
    trivector (64)  : higher-order structure — fast   (full update)
    residual (256)  : unconstrained channel  — pass-through

Cross-grade mixing via Cl(3,0,0) geometric self-product on vector grade:
    vector x vector -> scalar + bivector

Applied once per recurrence iteration inside the reasoning core.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from hagi_v4.algebra.clifford import BLADE_COUNT, geometric_product_self_g02
from hagi_v4.config import GDRConfig


class GradeRouter(nn.Module):
    """Learnable capacity gate over 4 Clifford grades (MoE-style).

    Projects graded context to 4 logits, softmaxes them, and returns a
    per-token gate [B, T, 4] that scales each grade's update magnitude.
    A Shazeer/Switch load-balance aux loss prevents collapse to one grade.
    """

    def __init__(
        self,
        ctx_size: int,
        num_grades: int = 4,
        alpha: float = 0.01,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_grades = num_grades
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.gate_proj = nn.Linear(ctx_size, num_grades, bias=False)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.01)

    def forward(self, graded_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.gate_proj(graded_ctx)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        if self.training:
            noise = torch.randn_like(logits) * 0.01
            logits = logits + noise.detach()
        probs = torch.softmax(logits, dim=-1)
        aux = None
        if self.training:
            flat = probs.reshape(-1, self.num_grades)
            top_idx = flat.argmax(dim=-1)
            one_hot = torch.zeros_like(flat)
            one_hot.scatter_(1, top_idx.unsqueeze(-1), 1.0)
            fraction = one_hot.mean(dim=0).detach()
            mean_prob = flat.mean(dim=0)
            aux = self.alpha * float(self.num_grades) * (fraction * mean_prob).sum()
        return probs, aux


class GradeDecomposedRecurrence(nn.Module):
    """One iteration of grade-decomposed update + geometric interaction.

    Uses a shared trunk (Linear + SiLU) and single fused head (Linear)
    instead of four separate per-grade MLPs: 2 matmuls instead of 8.
    """

    def __init__(self, cfg: GDRConfig, hidden_size: int = 576):
        super().__init__()
        scale = hidden_size / 576
        self.d_scalar = max(8, int(64 * scale))
        self.d_vector = max(8, int(96 * scale))
        self.d_bivector = max(8, int(96 * scale))
        self.d_trivector = max(8, int(64 * scale))
        self.d_vector = (self.d_vector // 8) * 8
        self.d_bivector = (self.d_bivector // 8) * 8
        if self.d_vector < 8:
            self.d_vector = 8
        if self.d_bivector < 8:
            self.d_bivector = 8
        self.d_residual = hidden_size - (self.d_scalar + self.d_vector + self.d_bivector + self.d_trivector)
        if self.d_residual < 1:
            total_grade = self.d_scalar + self.d_vector + self.d_bivector + self.d_trivector
            self.d_residual = max(1, hidden_size - total_grade)
            if self.d_residual < 1:
                self.d_scalar = max(8, hidden_size // 8)
                self.d_vector = max(8, (hidden_size - self.d_scalar) // 3 // 8 * 8)
                self.d_bivector = max(8, (hidden_size - self.d_scalar - self.d_vector) // 2 // 8 * 8)
                self.d_trivector = max(
                    8, hidden_size - self.d_scalar - self.d_vector - self.d_bivector - max(1, hidden_size // 4)
                )
                self.d_residual = max(
                    1, hidden_size - self.d_scalar - self.d_vector - self.d_bivector - self.d_trivector
                )
        ctx = self.d_scalar + self.d_vector + self.d_bivector + self.d_trivector
        self.ctx_size = ctx
        self._split_sizes = [self.d_scalar, self.d_vector, self.d_bivector, self.d_trivector]
        self._bounds = [
            0,
            self.d_scalar,
            self.d_scalar + self.d_vector,
            self.d_scalar + self.d_vector + self.d_bivector,
            ctx,
        ]

        self.grade_trunk = nn.Sequential(nn.Linear(ctx, ctx), nn.SiLU())
        self.grade_head = nn.Linear(ctx, ctx)

        def _mom_logit(m: float) -> float:
            m = min(max(m, 1e-4), 1 - 1e-4)
            return math.log(m / (1 - m))

        self.scalar_mom_logit = nn.Parameter(torch.tensor(_mom_logit(cfg.scalar_momentum)))
        self.vector_mom_logit = nn.Parameter(torch.tensor(_mom_logit(cfg.vector_momentum)))

        assert self.d_vector % BLADE_COUNT == 0, "vector grade must be divisible by 8"
        self.n_mv = self.d_vector // BLADE_COUNT

        self.geo_to_scalar = nn.Linear(self.d_vector, self.d_scalar, bias=False)
        self.geo_to_bivector = nn.Linear(self.d_vector, self.d_bivector, bias=False)
        self.gate_scalar = nn.Parameter(torch.zeros(1))
        self.gate_bivector = nn.Parameter(torch.zeros(1))

        self.grade_router: GradeRouter | None = None
        if cfg.use_grade_router:
            self.grade_router = GradeRouter(
                ctx,
                num_grades=4,
                alpha=cfg.grade_router_alpha,
            )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        b = self._bounds
        scalar = h[..., b[0] : b[1]]
        vector = h[..., b[1] : b[2]]
        h[..., b[2] : b[3]]
        h[..., b[3] : b[4]]
        residual = h[..., b[4] :]

        graded_ctx = h[..., : self.ctx_size]

        graded = self.grade_head(self.grade_trunk(graded_ctx))
        s_upd, v_upd, b_upd, t_upd = torch.split(graded, self._split_sizes, dim=-1)

        router_aux: torch.Tensor | None = None
        if self.grade_router is not None:
            gate, router_aux = self.grade_router(graded_ctx)
            s_upd = s_upd * gate[..., 0:1]
            v_upd = v_upd * gate[..., 1:2]
            b_upd = b_upd * gate[..., 2:3]
            t_upd = t_upd * gate[..., 3:4]

        sm = torch.sigmoid(self.scalar_mom_logit)
        vm = torch.sigmoid(self.vector_mom_logit)
        scalar_new = sm * scalar + (1 - sm) * s_upd
        vector_new = vm * vector + (1 - vm) * v_upd
        bivector_new = b_upd
        trivector_new = t_upd

        *lead, _ = vector_new.shape
        mv = vector_new.reshape(*lead, self.n_mv, BLADE_COUNT)
        g0, g2 = geometric_product_self_g02(mv)
        g0_flat = g0.reshape(*lead, self.d_vector)
        g2_flat = g2.reshape(*lead, self.d_vector)
        geo_scalar = torch.sigmoid(self.gate_scalar) * self.geo_to_scalar(g0_flat)
        geo_bivector = torch.sigmoid(self.gate_bivector) * self.geo_to_bivector(g2_flat)
        scalar_new = scalar_new + geo_scalar
        bivector_new = bivector_new + geo_bivector

        out = torch.cat([scalar_new, vector_new, bivector_new, trivector_new, residual], dim=-1)
        return out, router_aux
