"""Шаг 4 плана формализации: Bregman-slice дистилляция расхождений.

Тождество Брегмана (§1.4 плана): разница между суммой экспертов и
объединённой моделью равна среднему расхождению D_Φ(W_k, W̄). Чтобы
превратить разногласие в прирост γ, обучение должно идти НЕ на
консенсусных токенах (где D_Φ ≈ 0, градиент ≈ 0 — канал mixer.gain
оптимизатор выключает, R104), а на токенах с высоким информационным
расхождением учителей (high-disagreement tokens, < 5% потока).

Селекция: mean pairwise Jensen-Shannon между распределениями учителей
на токен; порог — КВАНТИЛЬ батча (self-calibrating: фиксированный τ
ломается при смене домена, квантиль 0.95 держит долю потока по плану).

Цель: forward-KL смесь учителей (forward_kl_teacher, T3/R134 — выбор
универсальности) на выбранных токенах:

    L = (1/|M|) Σ_{(b,t) ∈ M} KL(p̄ ‖ p_θ),  M = {meanJS > q_0.95}

Границы честно: JS — симметричный суррогат D_Φ (Брегман для CE — это
KL; mean pairwise JS ограничен и устойчив к нулю массы учителей);
селекция по квантилю батча — измеряемая, но батч-локальная.
"""
from __future__ import annotations

import torch

from hagi.train.distill_transfer import forward_kl_teacher

__all__ = [
    "mean_pairwise_js",
    "disagreement_mask",
    "disagreement_slice_loss",
]


def mean_pairwise_js(probs: list[torch.Tensor]) -> torch.Tensor:
    """Средняя попарная JS-дивергенция учителей на строку.

    ``probs`` — список [N][rows, V] вероятностных тензоров (одинаковые
    формы); возвращает [rows] — среднее по N(N−1)/2 парам
    0.5·KL(p‖m) + 0.5·KL(q‖m), m = (p+q)/2. JS ∈ [0, ln 2].
    Считает в float64: JS — малая разность двух почти равных
    перекрёстных энтропий, fp32-каннеляция искажает ближний к нулю
    хвост (консенсус), который и есть порог маски.
    """
    if len(probs) < 2:
        raise ValueError("need >= 2 teachers to measure disagreement")
    probs = [p.to(torch.float64) for p in probs]
    rows = probs[0].shape[0]
    total = torch.zeros(rows, dtype=torch.float64, device=probs[0].device)
    n_pairs = 0
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            m = 0.5 * (probs[i] + probs[j])
            kl_pm = (probs[i] * (probs[i].clamp_min(1e-300).log()
                                 - m.clamp_min(1e-300).log())).sum(-1)
            kl_qm = (probs[j] * (probs[j].clamp_min(1e-300).log()
                                 - m.clamp_min(1e-300).log())).sum(-1)
            total = total + 0.5 * (kl_pm + kl_qm)
            n_pairs += 1
    return total / n_pairs


def disagreement_mask(
    teacher_logits: list[torch.Tensor],
    quantile: float = 0.95,
    chunk: int = 256,
) -> torch.Tensor:
    """Маска токенов расхождения: meanJS > квантиля батча.

    ``teacher_logits`` — список [N] тензоров [*, V] (одинаковые формы,
    любые ведущие размерности). Возвращает bool-маску той же ведущей
    формы; отобрано ≈ 1−quantile токенов. Считается под no_grad, в
    float32, чанками по ``chunk`` строк — V=32768 не лезет в память
    целиком при N учителях.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    lead = teacher_logits[0].shape[:-1]
    v = teacher_logits[0].shape[-1]
    flat = [lg.reshape(-1, v).detach().to(torch.float32) for lg in teacher_logits]
    js_parts: list[torch.Tensor] = []
    for s in range(0, flat[0].shape[0], chunk):
        probs = [torch.softmax(f[s:s + chunk], dim=-1) for f in flat]
        js_parts.append(mean_pairwise_js(probs))
    js = torch.cat(js_parts)
    # строгое неравенство: при полном консенсусе (JS≡0) порог = 0 и
    # маска пуста — каналу нечему учиться, см. тест test_empty
    thr = torch.quantile(js, quantile)
    return (js > thr).reshape(lead)


def disagreement_slice_loss(
    student_logits: torch.Tensor,
    teacher_logits: list[torch.Tensor],
    quantile: float = 0.95,
    chunk: int = 256,
) -> torch.Tensor:
    """KL(p̄ ‖ p_θ) на токенах расхождения (forward-KL, T3/R134).

    Студент учится только там, где учителя расходятся: ёмкость модели
    тратится на разрешение противоречий, консенсусный градиент не
    глушит канал. Учителя не получают градиента. Пустая маска
    (полный консенсус) возвращает 0, связанный с графом студента.
    """
    mask = disagreement_mask(teacher_logits, quantile=quantile, chunk=chunk)
    v = student_logits.shape[-1]
    rows_s = student_logits.reshape(-1, v)
    rows_t = [lg.reshape(-1, v) for lg in teacher_logits]
    sel = mask.reshape(-1)
    if not bool(sel.any()):
        return rows_s.sum() * 0.0
    target = forward_kl_teacher([t[sel] for t in rows_t])
    log_p_theta = torch.log_softmax(rows_s[sel].to(torch.float64), dim=-1)
    # KL(p̄ ‖ p_θ) = Σ p̄ (log p̄ − log p_θ); target уже log p̄
    kl = (target.exp() * (target - log_p_theta)).sum(-1)
    return kl.mean()
