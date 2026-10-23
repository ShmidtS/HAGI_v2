"""Тесты рекуррентного бэкбона (gated linear recurrence)."""

from __future__ import annotations

import pytest
import torch

from hagi.model.recurrent import Recurrent, RecurrentConfig, build_recurrent_block

H = 32
T = 64
B = 3


def _make(**kw) -> Recurrent:
    torch.manual_seed(0)
    return build_recurrent_block(H, RecurrentConfig(**kw), use_ternary=False)


def test_output_shape():
    m = _make()
    x = torch.randn(B, T, H)
    out = m(x)
    assert out.shape == (B, T, H)


def test_causal_no_future_leak():
    m = _make()
    with torch.no_grad():
        m.out_proj.weight.normal_(std=0.05)  # иначе zero-init даёт нули всюду
    m.eval()
    x = torch.randn(2, T, H)
    with torch.no_grad():
        out1 = m(x)
        # сдвигаем вход на 1 позицию в будущее: прошлое то же, будущее другое
        x2 = torch.cat([x[:, : T // 2], torch.randn(2, T - T // 2, H)], dim=1)
        out2 = m(x2)
    # общий префикс => выходы на префиксе совпадают
    torch.testing.assert_close(out1[:, : T // 2], out2[:, : T // 2], rtol=1e-5, atol=1e-5)
    # и хвосты различаются (санити: блок не вырожден в константу)
    assert not torch.allclose(out1[:, T // 2 :], out2[:, T // 2 :])


def test_parallel_vs_stepwise_consistency():
    m = _make()
    m.eval()
    x = torch.randn(B, T, H)
    with torch.no_grad():
        par = m(x)
        h = m.state(x)
        steps = []
        for t in range(T):
            o, h = m.step(x[:, t], h)
            steps.append(o)
        stepwise = torch.stack(steps, dim=1)
    torch.testing.assert_close(par, stepwise, rtol=1e-5, atol=1e-5)


def test_zero_init_out_proj_is_identity_addition():
    m = _make()
    x = torch.randn(B, T, H)
    out = m(x)
    assert torch.all(out == 0), "zero-init out_proj должен давать нулевую добавку"
    assert torch.all(m.out_proj.weight == 0)


def test_state_dict_names_stable():
    m = _make()
    keys = set(m.state_dict().keys())
    expected = {
        "rec_norm.weight",
        "in_proj.weight",
        "nu",
        "out_norm.weight",
        "out_proj.weight",
        "branch_scale.scale",
    }
    assert keys == expected


def test_gradients_flow():
    m = _make()
    # выходим из zero-init, иначе out_proj не получит градиент по весу через 0
    with torch.no_grad():
        m.out_proj.weight.normal_(std=0.05)
    x = torch.randn(B, T, H)
    loss = m(x).square().mean()
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_state_dim_option_and_factory():
    m = build_recurrent_block(H, RecurrentConfig(state_dim=16), use_ternary=False)
    assert m.state_dim == 16
    out = m(torch.randn(1, 8, H))
    assert out.shape == (1, 8, H)


def test_alpha_bounded():
    m = _make()
    r = torch.randn(4, 8, m.state_dim) * 10
    a = m._alpha(r)
    assert (a > 0).all() and (a < 1).all()


def test_step_rejects_multi_token():
    m = _make()
    with pytest.raises(ValueError):
        m.step(torch.randn(1, 4, H), None)
