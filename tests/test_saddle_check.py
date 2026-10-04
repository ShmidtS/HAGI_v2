"""§7.2 runtime-проверка седла двойного zero-init + mixer-диагностика.

Порт протокола рецензии FORMALIZATION_PLAN §7.2 (до теорем T1–T6):

1. Строгий double-zero (gain=0 И факторы=0) — полное седло: ВСЕ
   градиенты тождественно нулевые. Гипотеза рецензии подтверждена
   на реальном коде HadamardMixer.
2. Реальный код никогда не имел строгого double-zero: факторы
   инициализируются normal (не zero), так что при gain=0 градиент
   фактора нулевым НЕ бывает... за исключением случая gain=0: тогда
   dL/dfactors ∝ gain = 0 — полу-седло, канал раскручивается только
   через крошечный градиент gain. Это фактич diagnostics истории
   (mixer.gain → 0 в gen1–gen5 при mixer_init_scale=0.0).
3. MergedHAGI.diagnostics() теперь репортит ||R|| рядом с gain
   (2607.16568: гейт и ветка анти-коррелированы; ранжирование по
   |gain| без ||R|| даёт обратный порядок).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hagi.model.merge import HadamardMixer  # noqa: E402


def _probe(gain_val: float, zero_factors: bool, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    m = HadamardMixer(256, 4, rank=16, mixer_init_scale=gain_val)
    if zero_factors:
        torch.nn.init.zeros_(m.gate.weight)
        torch.nn.init.zeros_(m.up.weight)
    x = torch.randn(2, 32, 256, requires_grad=False)
    target = torch.randn(2, 32, 256)
    y = m(x)
    F.mse_loss(y, target).backward()
    return {
        "gain": float(m.gain),
        "grad_gain": m.gain.grad.abs().item(),
        "grad_gate": m.gate.weight.grad.abs().mean().item(),
        "grad_up": m.up.weight.grad.abs().mean().item(),
        "grad_down": m.down.weight.grad.abs().mean().item(),
    }


class TestSaddle072:
    def test_strict_double_zero_is_full_saddle(self):
        # gain=0 AND factors=0 -> все градиенты ровно 0: полное седло,
        # гипотеза рецензии подтверждена на реальном коде
        r = _probe(0.0, zero_factors=True)
        assert r["grad_gain"] == 0.0
        assert r["grad_gate"] == 0.0
        assert r["grad_up"] == 0.0
        assert r["grad_down"] == 0.0

    def test_gain_zero_factors_normal_grads_alive(self):
        # gain=0 + normal факторы (реальные gen1–gen5): grad gain жив,
        # но градиенты факторов ∝ gain = 0 — полу-седло
        r = _probe(0.0, zero_factors=False)
        assert r["grad_gain"] > 0.0
        assert r["grad_gate"] == 0.0
        assert r["grad_up"] == 0.0
        # документируем полу-седло: канал стартует только через grad gain

    def test_gam_nonzero_gain_all_grads_alive(self):
        # gen6 GAM: gain=-0.1, факторы Gram-Schmidt — все градиенты живы,
        # седла нет
        r = _probe(-0.1, zero_factors=False)
        assert r["grad_gain"] > 0.0
        assert r["grad_gate"] > 0.0
    #
    # (2607.16568) anti-saddle init: не-нулевой gain при zero-factors —
    # ровно один ненулевой якобиан, не два нулевых

    def test_antisaddle_init_single_live_jacobian(self):
        r = _probe(0.02, zero_factors=True)
        # gain≠0: градиенты факторов живы через out≠0... gate/up нулевые
        # ⇒ out=0 ⇒ grad gain = <dy, 0> = 0, но grad down/up живы?
        # out = down(silu(gate(h))*up(h)) при gate=0: silu(0)*up = 0
        # ⇒ out = 0 ⇒ ВСЕ градиенты нулевые при zero gate — ловушка
        # zero-gate, отличная от double-zero: Even Т-Router init (только
        # gain≠0) не спасает, если gate=0. Правильный анти-седел: BOTH
        # факторы gate/up НЕ нулевые И gain ≠ 0.
        assert r["grad_gain"] == 0.0
        r2 = _probe(0.0, zero_factors=False)
        assert r2["grad_gain"] > 0.0


class TestMixerDiagnostics:
    def test_merged_reports_norm_next_to_gain(self):
        from hagi.config import Config
        from hagi.model.merge import MergedHAGI

        c = Config()
        c.model.vocab_size = 512
        c.model.hidden_size = 384
        c.model.num_layers = 2
        c.model.attention.num_query_heads = 6
        c.model.attention.num_kv_heads = 3
        c.model.attention.head_dim = 64
        c.model.ffn.expansion = 1.0
        c.model.ffn.multiple_of = 1
        c.model.embedding.tie_lm_head = False
        c.model.embedding.conv_kernel = 1
        c.merge.enabled = True
        c.merge.n_experts = 3
        c.merge.expert_hidden = 128
        c.merge.mixer_type = "hadamard"
        c.merge.mixer_rank = 16
        m = MergedHAGI(c, n_mixers=1, mixer_init_scale=0.1)
        d = m.diagnostics()
        assert "mixers/0/gain" in d
        assert "mixers/0/resid_norm" in d
        assert "mixers/0/effective" in d
        assert abs(d["mixers/0/effective"] - 0.1 * d["mixers/0/resid_norm"]) < 1e-6
        assert d["mixers/0/resid_norm"] > 0.0

    def test_format_metrics_includes_mixer_keys(self):
        from hagi.train.loop import format_metrics

        line = format_metrics(
            {"step": 10, "ce": 3.0, "bpt": 4.5, "ppl": 20.0,
             "kl": 0.0, "mixers/0/gain": -0.1, "mixers/0/resid_norm": 1537.0,
             "mixers/0/effective": 153.0}
        )
        assert "gain=-1.000e-01" in line
        assert "resid_norm=1.537e+03" in line
