# Раунд 18: growth_gate v4 — точная logit-алгебра вместо merge (приоритеты A+B)

## Что изменилось (синтез-обзор §4–§6)

Убран самый дорогой шаг петли роста: merged-модель строить НЕ нужно.
Ансамблевый CE = LSE(m_N) − m_N[t], m_N = S/N (running сумма
логитов листьев); кандидата решает точный ΔCE = CE_{N+1} − CE_N из
ОДНОГО forward кандидата (ensemble_delta_ce). Gap стримится
(jensen_gap_accum: S + A, O(one-leaf) памяти вместо [N,B,T,V]).

Стоимость gate-цикла: минуты merge+wide-forward → **~8 секунд**.

## Верификация

- Тождества (gap-accum ≡ gap-lse; CE-logits ≡ brute-force softmax;
  ΔCE ≡ прямая новая средняя) — бит-в-бит на синтетике (atol 1e-9).
- Живой прогон flat3 + кандидаты (leafdat_2xcorp, leafdat_b2,
  финальные step-1600):
  - 2xcorp: certifiedGain +0.092 → **точный ΔCE −0.188**, 1−ρ 0.332
  - b2:     certifiedGain +0.089 → точный ΔCE −0.176, 1−ρ 0.340
  - **Ранжирование 1−ρ ≠ ранжирование точного ΔCE** — живая
    демонстрация §3 синтеза (корреляция выбрасывает масштаб):
    1−ρ предпочитает b2, точная алгебра — 2xcorp. Prefilter ≠
    decision, иерархия выдержана.
- Точность: flat3 gap 0.298 (v4, float64) vs 0.399 (v3, bf16 CE) —
  формульный путь чище; CE 6.12 на финальных шагах vs 7.02 на
  полутренированных (v3 брал step-800) — референс раунда 16 был на
  полутренированных листьях; v4 берёт финальный step (TODO раунда
  14 закрыт: _find_checkpoint = последний step-*.pt).

## KKT-фикс kv_waterfill_bits (§16)

Старый код делал clamp_min(0) ПОСЛЕ замкнутой формы — нарушал
бюджет при отрицательных b_i. Исправлено активным множеством:
drop b_i<0 → пересчёт λ на активном наборе → повтор до стабилизации
(настоящий KKT waterfilling с ограничением b_i ≥ 0).

## Честный понижение статуса (§7)

G(N) ≈ G∞(1−1/N) — модель (docstring в GapLaw.lean), НЕ теорема;
N* = √(G∞/ε) выведен из модели. В v4 поле названо n_star_model,
в docstring явно: «model-based prediction, not a theorem». Вердикты
сатурации по-прежнему опираются на измеренную траекторию gap между
соседними уровнями. Lean-запрос (§7): доказать
0 ≤ ΔG_N ≤ C/(N(N+1)) при exchangeable deviations — тогда N*
станет theorem-backed.

## Backlog из синтеза (не перенесено, приоритеты C–G)

- C: R^T R = I для production RealF3 6×6 + индукция по дереву.
- D: параметрический root/contrast mixer Q(θ) = P_root + R_θ·P_contrast.
- E: ternary absmean → LS scale → q refresh A/B.
- F: 2D precision b_{l,m} (layer × tree-mode).
- G: shared base memory + low-rank expert deltas (96% параметров
  листа — embedding+head; BTX-усреднение провалилось 10.10→10.36).
- Lean-задача: G∞-закон как теорема (bound на ΔG_N).
- 60°/90° именование Q(pi/2) в комментариях (§9 — инвариантность
  не ломает, provenance поправить).

Тесты: 13 passed.
