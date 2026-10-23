"""Gated linear recurrence block — attention-free mixer branch (RNN эксперимент).

Reference-first: адаптация RG-LRU из Griffin/Hawk (De et al., "Griffin: Mixing
Gated Linear Recurrences with Local Attention for Efficient Language Models",
arXiv:2402.19427, §2.2) и LRU (Orvieto et al., arXiv:2303.06349); PyTorch-паттерн
пер-канальной диагональной рекурренты — Gothos/LRU-pytorch (github.com/Gothos/LRU-pytorch).

Recurrence (per channel, real-valued):

    a_t = sigmoid(r_t) * A,            A = exp(-exp(nu)) ∈ (0,1)   (LRU-стабилизация)
    b_t = sqrt(1 - a_t^2) * sigmoid(i_t) * x_t
    h_t = a_t ⊙ h_{t-1} + b_t

Оба гейта зависят только от текущего входа (не от h), что сохраняет
параллелизм. Тренировочный проход — параллельный chunked scan в
лог-пространстве (см. ``_parallel_scan``): наивный глобальный
cumsum-вариант переполняется уже при T=64 и a≈e^-6, поэтому проход разбит
на чанки с log-sum-exp нормировкой внутри чанка и явным переносом.

Интерфейс drop-in к Attention: forward(x) -> [B,T,H], pre-norm на входе,
out_proj zero-init + BranchScale — function-preserving вставка (блок
стартует как нулевая добавка в резидуальный поток). Имена тензоров
(in_proj/out_proj/rec_norm) — поименованные 2D-веса, пригодные для
per-tensor latent merge, как qkv_proj/out_proj у Attention.

FFN-путь не дублируется: Block уже содержит mixer. Интеграция в model.py —
отдельный шаг, здесь только модуль.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from hagi.model.ffn import BranchScale, linear
from hagi.model.norms import RMSNorm

# Граница лог-пространственной аккумуляции: log a зажат снизу, чтобы размах
# внутри чанка скана не уводил экспоненты за пределы float64.
_LOG_A_MIN = -20.0
# Размах логарифмов внутри чанка, безопасный для float32 (e^88 ≈ f32 max,
# запас ×2 на аккумуляцию): определяет динамический размер чанка.
_FP32_LOG_RANGE = 40.0


@dataclass
class RecurrentConfig:
    """Per-layer recurrence geometry (аналог AttentionConfig)."""

    state_dim: int = 0  # 0 = hidden_size
    gate_bias: float = 0.0  # начальный сдвиг сигмоид гейтов (0 = гейт 0.5)


def build_recurrent_block(
    hidden_size: int,
    cfg: RecurrentConfig | None = None,
    norm_eps: float = 1e-5,
    use_ternary: bool = True,
    residual_scale: float = 1.0,
    clamp_ratio: float = 2.0,
) -> Recurrent:
    """Фабрика по образцу сборки Attention в Block."""
    return Recurrent(
        hidden_size,
        cfg or RecurrentConfig(),
        norm_eps=norm_eps,
        use_ternary=use_ternary,
        residual_scale=residual_scale,
        clamp_ratio=clamp_ratio,
    )


class Recurrent(nn.Module):
    """Gated diagonal linear recurrence — drop-in mixer-альтернатива attention.

    Args:
        hidden_size: H ширина резидуального потока.
        cfg: геометрия рекурренты.
        norm_eps: RMSNorm epsilon.
        use_ternary: квантовать 2D-проекции.
        residual_scale: init-масштаб ветки (BranchScale).
        clamp_ratio: clamp BranchScale.
    """

    def __init__(
        self,
        hidden_size: int,
        cfg: RecurrentConfig | None = None,
        norm_eps: float = 1e-5,
        use_ternary: bool = True,
        residual_scale: float = 1.0,
        clamp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        cfg = cfg or RecurrentConfig()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        self.hidden_size = hidden_size
        self.state_dim = int(cfg.state_dim) or hidden_size
        self.gate_bias = float(cfg.gate_bias)

        self.rec_norm = RMSNorm(hidden_size, eps=norm_eps)
        # Fused in_proj: [x_state, r_gate, i_gate] — один вызов linear на слой
        # (тот же приём, что qkv_proj у Attention).
        self.in_proj = linear(hidden_size, 3 * self.state_dim, use_ternary)
        # LRU-параметризация A: nu (log-log) гарантирует A ∈ (0,1); init 0 ⇒ A=e^{-1}.
        self.nu = nn.Parameter(torch.zeros(self.state_dim))
        self.out_norm = RMSNorm(self.state_dim, eps=norm_eps)
        # Zero-init: блок стартует как тождественная (нулевая) добавка —
        # function-preserving depth-grow. Биаса у linear() нет (bias-free).
        self.out_proj = linear(self.state_dim, hidden_size, use_ternary)
        nn.init.zeros_(self.out_proj.weight)
        self.branch_scale = BranchScale(residual_scale, clamp_ratio=clamp_ratio)
        # Incremental decode state (аналог KVCache у Attention): включается
        # attach_state() из model.allocate_cache; None → полный проход с h0=0
        # (тренировка/eval без кэша — поведение исходное).
        self._state: torch.Tensor | None = None
        self._state_enabled = False
        self._state_len = 0  # токенов поглощено (позиции/cache_len)

    # -- core ---------------------------------------------------------------

    def _gates(self, h_normed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Проекции в состояние, гейт рекурренты и гейт входа. Всё [..., D]."""
        z = self.in_proj(h_normed)
        xz, rz, iz = z.split(self.state_dim, dim=-1)
        return xz, rz + self.gate_bias, iz + self.gate_bias

    def _alpha(self, r: torch.Tensor) -> torch.Tensor:
        """a = sigmoid(r) * A ∈ (0,1), с clamp лог-пространства, fp32."""
        a_log = F.logsigmoid(r.float()) - torch.exp(self.nu.float())
        return a_log.clamp(min=_LOG_A_MIN).exp()

    @staticmethod
    def _parallel_scan(
        a: torch.Tensor,
        b: torch.Tensor,
        chunk: int = 1024,
        h0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """h_t = Σ_{s≤t} (Π_{k=s+1..t} a_k) b_s — векторизованный чанковый скан.

        Наивный глобальный вариант h_t = exp(L_t)·cumsum(exp(-L_s)b_s)
        переполняется: при a ≈ e^-6 размах L растёт как T·|log a|. Здесь
        проход разбит на чанки размера ≤ _safe_chunk: внутри чанка L
        нормируется на максимум чанка (log-sum-exp стабилизация).

        Ключевой факт: чанк выбирается по самому быстрому каналу, поэтому
        медленные каналы переносят память через границы чанков — перенос
        НЕ отбрасывается. Схема — стандартный двухуровневый скан:
        (1) векторизованный внутричанковый проход (без python-цикла,
        главный источник copy_/slice в профиле устранён);
        (2) цикл переноса только по границам чанков на тензорах [B,D]
        (≤ T/chunk ≈ 85 итераций малых ядер) + одно векторизованное
        применение переноса ко всем позициям.

        a, b: [B, T, D]. Возвращает h: [B, T, D] (float32).
        """
        a = a.float()
        b = b.float()
        t_sz = a.shape[1]
        log_a = a.clamp(min=math.exp(_LOG_A_MIN)).log()
        # Безопасный размер чанка: кумулятивный log a внутри чанка не
        # дальше _FP32_LOG_RANGE от максимума, иначе exp(m-lt) переполняет
        # f32. При обученных a≈0.9 весь проход — один чанк; при a≈0.03
        # чанк ≈ 12 — но чанки считаются параллельно, итераций нет.
        max_rate = max((-log_a).max().item(), 1e-6)
        chunk = max(1, min(chunk, int(_FP32_LOG_RANGE / max_rate)))
        n_c = (t_sz + chunk - 1) // chunk
        pad = n_c * chunk - t_sz
        if pad:
            a = F.pad(a, (0, 0, 0, pad))
            b = F.pad(b, (0, 0, 0, pad))
            log_a = F.pad(log_a, (0, 0, 0, pad))
        # [B, nC, C, D]: кумулятивные логарифмы от начала каждого чанка
        b4 = b.view(a.shape[0], n_c, chunk, a.shape[2])
        lt = torch.cumsum(log_a.view(a.shape[0], n_c, chunk, a.shape[2]), dim=2)
        m = lt.max(dim=2, keepdim=True).values.detach()  # якорь масштаба ≤ 0
        scaled = torch.cumsum(torch.exp(m - lt) * b4, dim=2)
        out = torch.exp(lt - m) * scaled
        # (2) перенос между чанками: h_k = A_k·h_{k-1} + last_k на [B,D]
        a_chunk = torch.exp(lt[:, :, -1, :])  # полное затухание чанка
        last = out[:, :, -1, :]
        h = h0.float() if h0 is not None else a_chunk.new_zeros(a_chunk.shape[0], a_chunk.shape[2])
        carries = []
        for k in range(n_c):
            carries.append(h)
            h = a_chunk[:, k] * h + last[:, k]
        carries = torch.stack(carries, dim=1)  # [B, nC, D]
        out = out + carries.unsqueeze(2) * torch.exp(lt)  # lt ≤ 0 → безопасно
        return out.reshape(a.shape[0], -1, a.shape[2])[:, :t_sz]

    # -- API ----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ветка рекурренты для [B,T,H] (резидуал добавляет вызывающий).

        Causality — структурная: h_t зависит только от x_{≤t}, поэтому
        ``mask`` игнорируется (принимается для интерфейсной совместимости с
        Attention), ``positions`` — тоже (позиционная информация не нужна).

        С прикреплённым state (attach_state) проход продолжает рекурренту с
        сохранённого h и обновляет его последним шагом — инкрементальный
        декод без KV-кэша.

        Returns:
            ``[B, T, H]``.
        """
        xz, rz, iz = self._gates(self.rec_norm(x))
        a = self._alpha(rz)
        b = torch.sqrt(torch.clamp(1.0 - a * a, min=0.0)) * torch.sigmoid(iz.float()) * xz.float()
        h = self._parallel_scan(a, b, h0=self._state if self._state_enabled else None)
        if self._state_enabled:
            self._state = h[:, -1]
            self._state_len += x.shape[1]
        return self.branch_scale(self.out_proj(self.out_norm(h.to(x.dtype))))

    # -- incremental decoding state -------------------------------------------

    def attach_state(self) -> None:
        """Включить инкрементальное состояние (лениво: h создаётся на первом
        forward из батча входа — как KVCache.attach_cache до prefill)."""
        self._state_enabled = True
        self._state = None
        self._state_len = 0

    def detach_state(self) -> None:
        """Выключить и сбросить состояние (аналог detach_cache)."""
        self._state_enabled = False
        self._state = None
        self._state_len = 0

    @property
    def length(self) -> int:
        """Токенов поглощено состоянием (аналог KVCache.length)."""
        return self._state_len if self._state_enabled else 0

    # -- stepwise decoding ----------------------------------------------------

    def state(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Начальное (нулевое) состояние h: [B, D] на устройстве/в dtype входа."""
        if x is None:
            return torch.zeros(1, self.state_dim)
        return x.new_zeros(x.shape[0], self.state_dim)

    def step(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Один шаг генерации: x [B, H] (или [B, 1, H]) -> (выход [B, H], новое h)."""
        squeeze = x.dim() == 3
        if squeeze:
            if x.shape[1] != 1:
                raise ValueError(f"step() принимает один токен, got T={x.shape[1]}")
            x = x[:, 0]
        if h is None:
            h = self.state(x)
        xz, rz, iz = self._gates(self.rec_norm(x))
        a = self._alpha(rz)
        b = torch.sqrt(torch.clamp(1.0 - a * a, min=0.0)) * torch.sigmoid(iz.float()) * xz.float()
        h_new = a.double() * h.double() + b.double()
        out = self.branch_scale(self.out_proj(self.out_norm(h_new.to(x.dtype))))
        return (out[:, 0] if squeeze else out), h_new.to(h.dtype)
