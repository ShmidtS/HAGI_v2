# Раунд 8: zero-init residual projections (speedrun-приём) — принят

## Источник

Полный разбор annotated NanoGPT speedrun (nilmamano.com, июль 2026,
скрипт Prime Intellect): zero-init всех output-проекций — один из
эмпирически провалидированных трюков рекордов. Проверил наш арсенал
против speedrun-списка:

| speedrun-приём | у нас |
|---|---|
| QK-norm | есть (qk_norm: true) |
| Muon | есть (раунд 6) |
| torch.compile | есть (раунд 5) |
| WSD-decay | есть (warmup 40 + decay 20%) |
| relu² FFN | нет — проверено: FFN = 1.5 мс/шаг, копейки |
| value embeds | нет — oracle-тест: −0.008 ната, не окупается |
| **zero-init proj** | **НЕ было — реализован и принят** |

## Реализация

`train.zero_init_proj: true` (config.py + hook в train.py:6): после
построения модели все `attn.out_proj.weight` и `mixer.mixer.down.weight`
обнуляются — residual stream стартует как identity, блоки «вплывают»
из нуля. Совместим с env-переключателем HAGI_ZERO_INIT_PROJ.

## Замер (800 шагов, held-out, sweep-scale)

| вариант | seed 3001 | seed 7777 |
|---|---|---|
| Muon (ref) | 4.8972 | 4.9062 |
| **Muon + zero-init** | **4.8810** | **4.8834** |

Стабильно −0.02 ната на обоих seed. Микро, но согласованное и
бесплатное (0 мс). Принято в канонический конфиг.

## Канонический элемент (финал)

`configs/leaf_h128_v2.yaml`: H=128, L3, ld1, lr 1e-2 (adamw), warmup 40,
decay 20%, wd 0.3, bs32, accum1, **Muon**, fused_ce, compile,
**zero_init_proj**.

**CE 4.88, 278 секунд.** За сессию: 10.05 → 4.88 (−5.17 ната,
ppl 23150 → 131), время −41%.

## Проверено и отклонено в этом раунде

- relu² FFN: телу 4.5 мс/шаг — не рычаг.
- value embeds: oracle-alpha −0.008 ната на готовом листе.
- softcap: логиты ±0.5 на init, tanh ничего не делает.

## Ось первичного элемента ЗАКРЫТА

Все оси выжаты с двойным seed-подтверждением: lr, warmup, wd, batch,
ld, CE-ядро, compile, Muon, zero-init. Дальше — только применение:
лестница flat9/depth2 на Muon+zi-листьях, head-multiplicity триггер.
