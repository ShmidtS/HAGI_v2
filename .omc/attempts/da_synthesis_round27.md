# Раунд 27: DA-синтез — fused local+sinks, O(T(W+S)) вместо T²-маски

## Перенос из DA-анализа (arXiv:2609.02737)

Скопинг-факт: текущие gen-циклы без window-слоёв ([0,0,0]) —
dense-mask-на-sink НЕ ест текущие прогоны. Фикс сделан для будущих
window/long-context экспериментов (снимает блокер).

## Реализовано

`local_window_attention(..., sink_len=S)`: fused chunked local+sinks.
Ключевая тонкость, пойманная бит-точной верификацией: sink-ключи и
window-ключи НЕ должны дублироваться в softmax (дубликат сдвигает
массу внимания); window-диапазон начинается с конца sink-зоны
(wk0 = max(k0, sl)).

Верификация: fused == dense-mask бит-точно (atol 1e-9, фактически
3.9e-16); window-only путь не изменён; тест добавлен (19 green).

## Замеры скорости (bf16, W=256, S=4)

| T | fused | dense mask | ускорение |
|---|---|---|---|
| 1024 | 1.82 ms | 5.03 ms | 2.76× |
| 4096 | 8.24 ms | 78.06 ms | **9.47×** |

Ускорение растёт с T (T²/T(W+S)) — ровно предсказание синтеза §10.
Learnable sink_bias оставлен на mask-пути (ему нужен градиент через
маску; chunked путь = для bias-free конфигов).

## Не перенесено из DA-анализа (честный скопинг)

- A-Cortex (cortex как attention router) — условие запуска:
  long-context нагрузка (T≥32k); сейчас T=512-1024, выигрыш
  block-KV-routing не материализуется.
- 4-уровневая память (local/focus/index/global) — то же условие.
- Latent controller c_t = softmax(W_c h_t) — спроектирован в
  анализе, ждёт long-context фазу.
- Objective CE + λ_kv·A/A_full + λ_step·D — записан как цель
  inference-контроллера, не реализован (YAGNI до инференс-фазы).

## Приоритет по синтезу подтверждён

sampled NCE > A-Cortex для НЕМЕДЛЕННОГО ускорения (training-bound);
A-Cortex выше по архитектурной перспективе для future long-context
HAGI. Оба в очереди с явными условиями запуска.
