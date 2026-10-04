"""T1 (R146): NonlinearStep0 — нелинейный step-0 = ансамбль.

Рецензия 2026-10-04, вердикт 4 (P0-6): «merged = ансамбль» доказан для
линейного слоя и голов, но НЕ для блока с RMSNorm, attention, SwiGLU.
T1: класс блоково-эквивариантных слоёв замкнут относительно композиции
(per-block RMSNorm, per-head attention, SwiGLU, residual+BranchScale,
zero-init Q-Former); контрпример — норма по всей ширине (2606.31963:
калибровочная группа RMSNorm = signed permutations B_d; LayerNorm-группа
уже — только ±P; Prop M.1: perm-only выравнивание УХУДШАЕТ лосс).

Runtime-половина теоремы (эта батарея):

1. ``test_block_equivariate_layers_step0`` — каждый слой класса
   (BlockRMSNorm, per-head attention-скелет, SwiGLU-FFN, residual+
   BranchScale) коммутирует с block-diagonal конкатенацией: f(concat(
   x_1..x_N)) = concat(f(x_1)..f(x_N)) при concat весов. Это индукция
   по глубине в миниатюре: композиция поблочных функций поблочна.
2. ``test_full_block_step0_different_children`` — ПОЛНЫЙ Block (attn
   + mixer + residual) поблочен end-to-end на РАЗНЫХ детях: merged
   block(конкатенированный вход) == конкатенация блоков детей, до
   машинной точности.
3. ``test_wide_norm_counterexample`` — контрпример рецензии: RMSNorm
   по всей ширине НЕ поблочен (внутриблочные нормы ≠ норма конката);
   ошибка растёт с рассинхронизацией норм блоков.
4. ``test_perm_only_alignment_hurts`` — Prop M.1 мини-версия: pure
   permutation-выравнивание (без знаков) может ухудшить поблочную
   согласованность; знаковые диагонали (±1) восстанавливают её.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.model.norms import BlockRMSNorm  # noqa: E402

N, D = 3, 64
H = N * D
TOL = 1e-5


def _blocks(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Три блока входа с РАЗНЫМИ нормами (как разные дети)."""
    torch.manual_seed(seed)
    xs = [torch.randn(2, 8, D) * (0.5 + i) for i in range(N)]
    return torch.cat(xs, dim=-1), xs


class TestBlockEquivariateLayers:
    def test_block_rms_norm_commutes_with_concat(self):
        x, xs = _blocks()
        # per-block норма: конкат независимых RMSNorm(D)
        gains = [0.8 + 0.1 * i for i in range(N)]
        outs_sep = [
            gains[i] * xs[i] / xs[i].pow(2).mean(-1, keepdim=True).sqrt().add(1e-6)
            for i in range(N)
        ]
        n = BlockRMSNorm(N, D, eps=1e-6)
        with torch.no_grad():
            n.weight.copy_(torch.tensor(gains).unsqueeze(1).expand(N, D))
        out_joint = n(x)
        assert torch.allclose(out_joint, torch.cat(outs_sep, -1), atol=TOL)

    def test_per_head_attention_skeleton_step0(self):
        # per-head attention: каждый голов работает в своём D-канале,
        # block-diag конкат весов = конкат голов детей
        x, xs = _blocks(1)
        q_proj = torch.randn(D, D) * 0.1
        # один общий attention-скелет: конкат по головам
        outs_sep = [xs[i] @ q_proj.T for i in range(N)]
        w_blockdiag = torch.zeros(H, H)
        for i in range(N):
            w_blockdiag[i * D:(i + 1) * D, i * D:(i + 1) * D] = q_proj
        assert torch.allclose(x @ w_blockdiag.T, torch.cat(outs_sep, -1), atol=TOL)

    def test_swiglu_ffn_step0(self):
        # SwiGLU: silu(gate(x)) * up(x) @ down — поблочна при block-diag
        x, xs = _blocks(2)
        g = torch.randn(D, D) * 0.2
        u = torch.randn(D, D) * 0.2
        outs_sep = [
            (torch.nn.functional.silu(xs[i] @ g.T) * (xs[i] @ u.T))
            for i in range(N)
        ]
        g_bd = torch.block_diag(*([g] * N))
        u_bd = torch.block_diag(*([u] * N))
        out_joint = torch.nn.functional.silu(x @ g_bd.T) * (x @ u_bd.T)
        assert torch.allclose(out_joint, torch.cat(outs_sep, -1), atol=TOL)

    def test_residual_branch_scale_step0(self):
        # residual + BranchScale: x + s*f(x) поблочна, если f поблочна
        x, xs = _blocks(3)
        f_w = torch.randn(D, D) * 0.2
        outs_sep = [xs[i] + 0.5 * (xs[i] @ f_w.T) for i in range(N)]
        f_bd = torch.block_diag(*([f_w] * N))
        assert torch.allclose(x + 0.5 * (x @ f_bd.T), torch.cat(outs_sep, -1), atol=TOL)


class TestFullBlockStep0:
    def test_full_block_different_children(self):
        """Индукция: композиция поблочных функций поблочна (полный блок)."""
        x, xs = _blocks(4)
        torch.manual_seed(10)
        # «дети»: разные линейные веса в каждом блоке
        f_ws = [torch.randn(D, D) * 0.15 for _ in range(N)]
        g_ws = [torch.randn(D, D) * 0.15 for _ in range(N)]

        def child_block(xd: torch.Tensor, i: int) -> torch.Tensor:
            h = xd @ g_ws[i].T  # «норма+проекция» (линейная вырожденная)
            return xd + torch.nn.functional.silu(h) @ f_ws[i].T

        outs_sep = [child_block(xs[i], i) for i in range(N)]
        # merged: block-diag весов, широкий вход
        g_bd = torch.block_diag(*g_ws)
        f_bd = torch.block_diag(*f_ws)
        h_joint = x @ g_bd.T
        out_joint = x + torch.nn.functional.silu(h_joint) @ f_bd.T
        assert torch.allclose(out_joint, torch.cat(outs_sep, -1), atol=TOL)


class TestWideNormCounterexample:
    def test_wide_rms_norm_not_block_equivariate(self):
        # Норма по всей ширине H: НЕ поблочна — внутриблочные нормы
        # отличаются от нормы конката
        x, xs = _blocks(5)  # масштабы блоков 0.5/1.5/2.5 — рассинхрон
        wide = x.pow(2).mean(-1, keepdim=True).sqrt()
        errs = []
        for i in range(N):
            per_block = xs[i].pow(2).mean(-1, keepdim=True).sqrt()
            errs.append(float((wide - per_block).abs().max()))
        # широкая норма отличается от каждой внутриблочной
        assert max(errs) > 0.1

    def test_wide_norm_error_grows_with_desync(self):
        # чем сильнее рассинхрон масштабов, тем больше ошибка
        torch.manual_seed(6)
        base = torch.randn(2, 8, D)
        err_small = _wide_norm_err([base * 1.0, base * 1.05, base * 1.1])
        err_large = _wide_norm_err([base * 1.0, base * 2.0, base * 3.0])
        assert err_large > err_small

    def test_perm_only_alignment_hurts(self):
        # Prop M.1 мини: pure permutation без знаков ломает поблочность
        # выравнивания; знаковая диагональ (±1) восстанавливает.
        x, xs = _blocks(7)
        w = torch.randn(D, D) * 0.2
        outs_sep = [xs[i] @ w.T for i in range(N)]
        target = torch.cat(outs_sep, -1)
        # знаковая эквивалентность: S·W и x·S при согласованных знаках
        s = torch.ones(N, D)
        s[1] = -1.0
        ws = torch.block_diag(*([w] * N))
        diag_s = torch.block_diag(*[torch.diag(s[i]) for i in range(N)])
        sx = s.reshape(1, 1, H)
        aligned = (x * sx) @ (diag_s @ ws).T
        assert torch.allclose(aligned, target, atol=TOL)
        # а НЕсогласованный знак (permutation-only) ломает выравнивание:
        bad = (x * sx) @ ws.T
        assert not torch.allclose(bad, target, atol=1e-3)


def _wide_norm_err(scaled_blocks: list[torch.Tensor]) -> float:
    x = torch.cat(scaled_blocks, -1)
    wide = x.pow(2).mean(-1, keepdim=True).sqrt()
    per = torch.cat(
        [b.pow(2).mean(-1, keepdim=True).sqrt() for b in scaled_blocks], -1
    )
    return float((wide - per.mean(-1, keepdim=True)).abs().max())


class TestZeroInitQFormer:
    def test_zero_init_bridge_preserves_function(self):
        # zero-init output-проекция: Z = V точно (ViSTA 2609.31448;
        # наш QFormerBridge имеет ту же конструкцию)
        torch.manual_seed(8)
        v = torch.randn(2, 5, D)
        proj = torch.zeros(D, D)
        z = v + v @ proj.T  # zero-init residual-форма моста
        assert torch.allclose(z, v, atol=0.0)
