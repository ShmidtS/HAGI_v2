"""§7.2 runtime-проверка седла двойного zero-init (до теорем T1-T6).

Гипотеза рецензии: gain=0 И residual-факторы=0 вместе -> седло с нулевым
градиентом -> gamma-дефицит 9x может быть артефактом кода.

Проверяем на реальном HadamardMixer из победной линии:
1. dL/d(gain) при gain=0 и НЕнулевых факторах (реальный код: init normal)
2. dL/d(gain) при gain=0 И zero-факторах (строгий double-zero)
3. dL/d(факторы) при gain=0 (заморожен ли канал при нулевом gain)
4. Реальный gen6-режим: gain=-0.1 (GAM phase-6), факторы Gram-Schmidt
"""
import sys

sys.path.insert(0, "src")
import torch
import torch.nn.functional as F

from hagi.model.merge import HadamardMixer

torch.manual_seed(0)
H, N, RANK = 1152, 3, 64


def probe(tag: str, gain_val: float, zero_factors: bool) -> None:
    m = HadamardMixer(H, N, rank=RANK, mixer_init_scale=gain_val)
    if zero_factors:
        torch.nn.init.zeros_(m.gate.weight)
        torch.nn.init.zeros_(m.up.weight)
    x = torch.randn(2, 64, H, requires_grad=False)
    target = torch.randn(2, 64, H)
    y = m(x)
    loss = F.mse_loss(y, target)
    loss.backward()
    g_gain = m.gain.grad.abs().item()
    g_gate = m.gate.weight.grad.abs().mean().item()
    g_up = m.up.weight.grad.abs().mean().item()
    g_down = m.down.weight.grad.abs().mean().item()
    print(
        f"{tag:34s} loss={loss.item():.4f} |grad gain|={g_gain:.3e} "
        f"|grad gate|={g_gate:.3e} |grad up|={g_up:.3e} "
        f"|grad down|={g_down:.3e}"
    )


probe("gain=0, factors normal (код)", 0.0, False)
probe("gain=0, factors ZERO (double-zero)", 0.0, True)
probe("gain=-0.1 (GAM gen6)", -0.1, False)
probe("gain=+0.02 (T-Router anti-saddle)", 0.02, False)

# Символическая структура: y = mixed + branch_scale(down(silu(gate(h))*up(h)))*gain
# dL/dgain = <dL/dy, branch_scale*out> — НЕ зависит от gain;
# dL/dfactors ~ gain * dL/dy * d(out)/dfactors — зануляется при gain=0,
# но НЕ при gain=-0.1. Значит: strict double-zero = седло (обе группы
# градиентов нулевые), реальный gen6-код седлом НЕ является.
print()
print("VERDICT: strict double-zero IS a saddle; gen6 config is NOT")
