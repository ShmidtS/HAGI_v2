# Исследовательские срезы HAGI: decision plane, data artifacts, embeddings

Документ фиксирует три **независимых opt-in среза**. Они не смешаны в
одном неатрибутируемом изменении и не меняют поведение baseline.

## 1. DecisionPlane — структурированные решения

`model.decision.enabled=true` добавляет `DecisionHead(AdaptiveComponent)` над
последней текстовой позицией после `out_norm` и `_apply_mixers`. Head
наблюдательный: его logits не возвращаются в LM path. При включении он
инициализируется точным нулём и не сдвигает RNG baseline.

Контракт objective:

```text
lm_loss       = mean CE по scored LM tokens
decision_loss = mean CE по выбранным sequence labels
loss          = lm_loss + decision.loss_weight * decision_loss
```

В `Trainer` microbatch-ы взвешиваются независимо по количеству LM tokens
и decision rows. Mixed objective использует обе компоненты. Отдельный
pre-registered experimental lane `decision_only` намеренно передаёт
decision labels без LM targets; метрика тогда явно помечается
`receiver="decision_only"`, а отсутствие любых scored rows отклоняется.

Границы контракта:

- default `model.decision.enabled=false`: нет модуля, параметров и
  checkpoint keys; `count_params`, `describe` и generation не меняются;
- `freeze_base=true` оставляет trainable только `DecisionHead`;
- `self_improve` с decision plane отклоняется до появления trajectory
  labels;
- expert decision state при merge не складывается: merged level получает
  fresh zero-init head;
- full forward и prefill+incremental decode дают одинаковые final-position
  decision logits.

Runner: `scripts/decision_plane_ab.py`, четыре lanes
(`majority_or_uniform`, `frozen_probe`, `decision_only`, `end_to_end`).
Synthetic 3-seed run подтвердил mechanism, но не quality. Затем
pre-registered real Banking77 gate (`scripts/decision_plane_banking77.py`)
выполнил 10 003 train / 3 080 official test rows на seeds 1234/2243/3252,
3 epochs, exact-length batches, train-only compact vocabulary и
zero-padding/no-truncation протокол.

Canonical schema-v2 report
`reports/decision_plane_banking77_v2.json` (SHA-256
`da7184d6d60bd7ab76f35b4af95f2f71b1902c702fdbe332369bd7b0ec344174`)
закрывает mechanism/evidence gates:

- NLL ниже majority и frozen probe: **3/3 seeds**;
- accuracy без регрессии: **3/3 seeds**;
- все 1 413 optimizer-step records finite и hash-bound;
- source snapshot/provenance стабильны во время run;
- ECE ≤ 0.15: только **2/3 seeds**; seed 3252 = **0.154493**.

Поэтому gate корректно fail-closed: `quality_supported=false`,
`promotion_status=research-only`. Schema-v1 report superseded: он формально
прошёл ECE с запасом 0.000162, но не содержал source snapshot и per-step
ledgers. V1/v2 shared the declared data/config/seed/budget, but were not
byte-identical source-tree executions and did not enable deterministic
Torch algorithms; small body-lane drift is recorded as a reproducibility
caveat, not a reason to rerun until the threshold passes. Независимый audit
canonical v2: 0 High / 0 Medium / Low.

## 2. Versioned data artifacts и bounded acquisition

Слой `src/hagi/data/artifacts.py` не меняет production loader. Он публикует
опциональный каталог только после проверки schema, полного file map,
SHA-256, byte/token counts, uint32 range и manifest-last atomic rename.

`scripts/prepare_training_data.py` предоставляет:

```text
acquire -> validate/filter/deduplicate -> tokenize -> EOS-pack -> publish
```

- только strict UTF-8 NDJSON;
- exact normalized-text SHA-256 dedup;
- duplicate/empty/malformed records попадают в quarantine ledger;
- EOS-delimited little-endian `uint32` shards;
- bounded local/HTTP acquisition с обязательным SHA-256 для remote;
- запрет credentials в URL, redirects, malformed/negative
  `Content-Length`, oversized body и silent temp leftovers;
- symlink/junction/root escape fail-closed;
- deterministic repeat-publication до одинаковых bytes;
- проверенный на установленном Gigatoken 0.10 native seam:
  `Tokenizer(name).encode_batch_list(batch)`.

Команды:

```bash
python scripts/prepare_training_data.py acquire SOURCE OUTPUT \
  --max-bytes N --expected-sha256 HASH
python scripts/prepare_training_data.py prepare INPUT OUTPUT_DIR \
  --tokenizer NAME --vocab-size V
python scripts/prepare_training_data.py validate OUTPUT_DIR
```

Статус: **promoted как opt-in infrastructure**: 34 generic artifact/acquisition
tests и независимый security review проходят. `scripts/prepare_banking77.py`
добавляет 11 offline tests и публикует pinned Banking77 real corpus:

- upstream commit `57ec275d8078af65b7731c2a98be812d844a6d6b`;
- 10 003 train / 3 080 test строк, 77 категорий, 0 rejected/overlap;
- exact text, source-row-aligned labels и upstream provenance CSV;
- три little-endian EOS-packed `uint32` shards, 192 682 токена;
- `manifest.json` SHA-256
  `44d50edd994e7c32f30066a26052e15f902c74b79a7498ba60f475c76b453f3b`;
- held-out `text/test.txt` SHA-256
  `b4148261b6025a32f2c8cd3318ae48bc6225f74603ae022346c9ca371f29a06c`;
- отдельный replay installed Gigatoken 0.10.0 воспроизвёл shard bytes;
- независимый recheck: 0 High / 0 Medium / 0 Minor.

Это **integrity/reproducibility evidence**, а не model quality и не
cryptographic authenticity. SHA-256 не заменяет signed manifests.
Старые `data/*.compact.bin` по-прежнему не имеют immutable acquisition
manifests и не получают quality-supported status.

## 3. `tie_lm_head`: минимальный embedding/head A/B

Production уже поддерживает `model.embedding.tie_lm_head`. Новый runner
изолированно сравнивает tied и untied head:

- одинаковое тело, batches, seeds, optimizer и update budget;
- untied projection инициализируется **точным состоянием** codebook;
- step-zero hidden/logit hashes и exact CE совпадают;
- измеряются exact native-token CE по RU/EN/math-code/instruction slices,
  parameter + optimizer bytes и step time.

3-seed real packed-stream run: untied выиграл mean CE на 2/3 seeds
(средняя разница `-0.02257` nats), но training-state memory выросла в
`1.934x`, выше заранее объявленного лимита `1.75x`. Поэтому untied head
**не promoted**; tied остаётся default.

Native-token CE нельзя использовать для сравнения разных tokenizers.
Opt-in evaluator `scripts/common_reference_eval.py` реализует strict
UTF-8 round-trip, retained-prefix rolling windows и total NLL / UTF-8 BPB через
`head.exact_loss`; 20 evaluator tests, 34 generic data tests и 11 publisher
tests проходят (65 focused total). Это разблокирует саму методику
измерения. Затем canonical matched **from-scratch tiny HAGI** gate на
pinned Banking77 завершён: exact `3×2×471`, 2 826 finite/hash-bound
optimizer-step records, execution-valid и systems-supported. Candidate
`tokenizer-0997f410` выиграл только seed 2243; парные BPB improvements
(baseline минус candidate) равны:

- seed 1234: `−0.028601`;
- seed 2243: `+0.006763`;
- seed 3252: `−0.062658`.

Все три дельты не положительны, median `−0.028601 < 0.01`, mean
`−0.028165 < 0.005`. Поэтому quality gate fail-closed:
`quality_supported=false`, `systems_supported=true`, tokenizer replacement
**rejected**, production baseline остаётся неизменным. Report SHA-256:
`b3a6a371ea9c3f0ef4c4bb54a03e989e4fac3a6277943d56073634063e0f3ef6`.
Независимые evidence/science audits не нашли residual High/Medium
findings. Наличие hash plan/runner/tests стабильно во время run не заменяет
внешнюю подпись/timestamp preregistration; это ограничение temporal
provenance, а не повод повторять run.

## Carry-forward

Комбинированный promoted architecture сейчас не существует. Следующий
валидный порядок:

1. baseline tokenizer оставить неизменным; не повторять canonical run ради
   прохождения уже зафиксированного BPB gate и не ослаблять его пороги;
2. отдельно pre-register source-specific DEPT/TRIM/SPEC embedding slice с
   явным optimizer/aggregation contract и fixed immutable evaluation;
3. не повторять DecisionPlane gate ради ECE; combined experiment строить
   только из independently promoted components.

Ни один компонент не называется «идеальным»: результаты остаются
bounded evidence на объявленных данных, seed budget и objective.
