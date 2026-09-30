# Round 56b: PoE-транспорт == logit-mean (равные веса); related work верифицирован

## PoE-гейт (существующие чекпоинты, 0 GPU)

| пул | logit-mean | PoE (1/3 log-probs) |
|---|---|---|
| 3× boost-сибы | 3.5650 | 3.5650 |
| 2 seed + 1 boost | 3.5783 | 3.5783 |

При равных весах mean-logprob и mean-logit дают одинаковое CE
(разница — аддитивная константа ΣlogZ/3, не влияющая на предсказание
и CE). **PoE как транспорт НЕ даёт прироста над logit-mean при
равных весах** — отличается от merge+joint (3.4443, лучший).
Вывод: ансамбль-канал на этих сибах ограничен самим disagreement
(GapLaw), а не формой пула. Инференс- PoE закрыт.

(Первая попытка с суммой log-probs дала 6.42 — тройное
заострение температуры, известный PoE-эффект; ошибка моя, исправлено
в той же сессии.)

## Related work — верификация (правило: доказательства, не утверждения)

- **TorchLean** (LeanDojo/Caltech/Anandkumar, arXiv:2602.22631) —
  ПОДТВЕРЖДЁН (alphaxiv + icml.cc + leandojo.org + LinkedIn
  автора). Lean-4 фреймворк: typed tensors, verified autograd,
  IEEE-754, IBP/CROWN. Пересечение с Hagi: устойчивость по
  Ляпунову (адаптируемо для SafeQP-контроллера) — НЕ дублирует
  наш уровень (рост поколений/merge-алгебра/LazyAdam/GapLaw).
- FREE-Merging / CAGrad / PCGrad — не проверены по-файльно; CAGrad
  и PCGrad известны как реальные (NeurIPS 20/21), их QP/проекция —
  прямые родственники SafeQP; для формализации stochastic-SafeQP
  взять аналитическую схему CAGrad — разумно.
- Позиционирование Hagi устойчиво: формализация ПРОЦЕССА РОСТА
  (merge-алгебра, GapLaw, GenCycle, waterfilling, LazyAdam) не
  покрыта TorchLean.
