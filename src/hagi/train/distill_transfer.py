"""T3 (замена R130): DistillTransfer — сертификат дистилляции в натах.

Рецензия 2026-10-04 вердикт 2: R130 переформулирует γ-дефицит, а не
закрывает его (GainOp — посылка D'≥ρD+ηE_dev−ξ, не построенный
оператор; единицы не сходятся; η без метрики Σ бессмысленна). T3 —
реальный измеримый оператор переноса:

**Граница переноса (2609.38666 + 2607.15467 + 2609.39436):**

    CE_q(θ) − CE_q(E) ≤ KL(p_E‖p_θ) + M·‖q − p_E‖₁   при |log(p_E/p_θ)| ≤ M

— сколько недобирает студент (θ) до ансамбля-учителя (E) на данных q,
через (а) KL-расхождение распределений и (б) L₁-зазор данных до
учителя. Оба члена измеримы на реальных логитах.

**КПД канала (γ-оператор, в чистых натах):**

    η = (CE_leafmean − CE_student) / twoGap

где twoGap = CE_leafmean − CE_ensemble — полный разрыв «усреднение
экспертов» (GapLaw), числитель — фактически реализованная часть.
η ∈ [0,1]: 1 = студент догнал ансамбль, 0 = дистилляция не дала ничего.
Это ТА γ, дефицит которой (0.002 vs 0.018) — узкое место системы,
теперь в честных единицах и без метрического смешения.

**Выбор направления KL (2609.38666 closed-form):** forward KL →
взвешенное арифметическое среднее учителей (сохраняет вклад каждого
эксперта, универсальность); reverse KL → нормализованное геометрическое
среднее (подавляет minority-учителя, модность). Для универсальной
модели forward — правильный дефолт, reverse — для скорости на моде.

**Cold-start (2607.16955):** свежий студент под reverse-KL даёт
нулевую массу там, где учителя согласны → ваншинг-градиент; merged-
студент разделяет поддержку с листьями, но on-policy фаза нужна.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "transfer_bound",
    "channel_efficiency",
    "forward_kl_teacher",
    "reverse_kl_teacher",
    "distill_floor_gap",
]


def transfer_bound(
    logit_student: torch.Tensor,
    logit_teacher: torch.Tensor,
    logit_data: torch.Tensor | None = None,
) -> dict[str, float]:
    """Граница переноса на одном батче: KL(p_E‖p_θ) + M·‖q−p_E‖₁.

    ``logit_student``/``logit_teacher`` — [*, V] логиты; ``logit_data``
    — если передан, эмпирическое q (one-hot/multihot/сглаженное) для
    L₁-члена, иначе член 0 (дистилляция на teacher-распределении,
    q = p_E ⟹ ‖q−p_E‖₁ = 0).
    """
    if logit_student.shape != logit_teacher.shape:
        raise ValueError(
            f"student {tuple(logit_student.shape)} != teacher "
            f"{tuple(logit_teacher.shape)}"
        )
    ls = logit_student.to(torch.float64)
    lt = logit_teacher.to(torch.float64)
    log_p = torch.log_softmax(ls, dim=-1)
    log_t = torch.log_softmax(lt, dim=-1)
    p = log_p.exp()
    t = log_t.exp()
    # KL(p_E || p_theta) по позициям, среднее
    kl = float((t * (log_t - log_p)).sum(dim=-1).mean())
    m_bound = float((log_t - log_p).abs().max())
    l1 = 0.0
    if logit_data is not None:
        if logit_data.shape != logit_teacher.shape:
            raise ValueError("logit_data shape mismatch")
        q = torch.softmax(logit_data.to(torch.float64), dim=-1)
        l1 = float((q - t).abs().sum(dim=-1).mean())
    return {
        "kl": kl,
        "m_bound": m_bound,
        "l1_data_teacher": l1,
        "bound": kl + m_bound * l1,
    }


def channel_efficiency(
    ce_leaf_mean: float,
    ce_student: float,
    ce_ensemble: float,
) -> float:
    """η = (CE_leafmean − CE_student) / twoGap — КПД дистилл-канала.

    twoGap = CE_leafmean − CE_ensemble (GapLaw, > 0 при разногласии).
    η ∈ [0,1] при 0 ≤ student-gain ≤ gap; > 1 = студент ОБГОНИЛ
    ансамбль (возможно при регуляризации), < 0 = деградация.
    """
    gap = ce_leaf_mean - ce_ensemble
    if gap <= 0.0:
        raise ValueError(
            f"twoGap must be > 0 (experts agree, nothing to distill): {gap}"
        )
    return (ce_leaf_mean - ce_student) / gap


def forward_kl_teacher(
    logits: list[torch.Tensor], weights: list[float] | None = None
) -> torch.Tensor:
    """Forward-KL цель: взвешенное АРИФМЕТИЧЕСКОЕ среднее учителей.

    Аргмин forward KL(p̄‖p_θ) по p_θ = p̄ = Σ w_i p_i (2609.38666):
    сохраняет вклад каждого эксперта — выбор универсальности.
    """
    if not 1 <= len(logits) <= 8:
        raise ValueError("1..8 teachers expected")
    w = weights if weights is not None else [1.0 / len(logits)] * len(logits)
    if len(w) != len(logits) or abs(sum(w) - 1.0) > 1e-9 or min(w) < 0.0:
        raise ValueError("weights must match teachers, sum to 1, be >= 0")
    probs = [
        torch.softmax(lg.to(torch.float64), dim=-1) for lg in logits
    ]
    mix = sum(wi * pi for wi, pi in zip(w, probs))
    # логит смеси через log (последний слой дистилляции работает в
    # логит-пространстве; log — точный обратный к softmax)
    return torch.log(mix.clamp_min(1e-30))


def reverse_kl_teacher(
    logits: list[torch.Tensor], weights: list[float] | None = None
) -> torch.Tensor:
    """Reverse-KL цель: нормализованное ГЕОМЕТРИЧЕСКОЕ среднее.

    Аргмин reverse KL(p_θ‖p̄) = Π p_i^{w_i} / Z (2609.38666): подавляет
    minority-учителей (misleading teacher не может завалить верный
    консенсус), выбор модности. Cold-start осторожно (2607.16955).
    """
    if not 1 <= len(logits) <= 8:
        raise ValueError("1..8 teachers expected")
    w = weights if weights is not None else [1.0 / len(logits)] * len(logits)
    if len(w) != len(logits) or abs(sum(w) - 1.0) > 1e-9 or min(w) < 0.0:
        raise ValueError("weights must match teachers, sum to 1, be >= 0")
    log_p = torch.log_softmax(logits[0].to(torch.float64), dim=-1) * w[0]
    for lg, wi in zip(logits[1:], w[1:]):
        log_p = log_p + torch.log_softmax(lg.to(torch.float64), dim=-1) * wi
    # нормализация геометрической смеси (÷Z): logsumexp возвращает
    # нормализованные лог-вероятности — это и есть логит-форма
    return log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)



def distill_floor_gap(
    ce_argmax_teacher: float, ce_full_teacher: float
) -> float:
    """Потолок канала (2607.15467): argmax передаёт минимум.

    Разница CE(argmax-учитель) − CE(full-учитель) — сколько «dark
    knowledge» (форма распределения сверх моды) ещё доступно каналу.
    Если student CE ≈ argmax CE — канал исчерпан, дальнейшая
    дистилляция без новой информации от учителя невозможна.
    """
    return ce_argmax_teacher - ce_full_teacher
