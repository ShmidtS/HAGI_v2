# HIP unspecified launch failure (F3 leaf rerun, 2026-09-26 16:34)

Симптом: `scripts/train.py` на `configs/f3_leaf_s1001.yaml` падает на первом
же шаге в `src/hagi/model/head.py:135` -> `logits.max(dim=-1)` ->
`torch.AcceleratorError: HIP error: unspecified launch failure`.
Стек указывает на `logits.max`, но HIP-ошибки асинхронны, так что это лишь
первое место, где ошибка всплыла, а не место причины.

Контекст: чекпойнты f3_leaf_s100{1,2,3}/step-0003000.pt на месте и полны
(166 MB каждый), то есть предыдущие прогоны этих листьев были успешны.
Прогон 16:34 — это повтор, вероятно с `--resume`.

Окружение прогона: `mm: off | ternary_cache: True | ternary_fp32_master: False |
ce_keep: 1/bernoulli | sampled_k: full | decision: off`
В рабочих прогонах N=6/N=7 было `sampled_k: 2048` и `ce_keep: 0.25/stride`.

- [1] H1 (ПОДТВЕРЖДЁН, но не через память): падение было вызвано ДАННЫМИ, а не
  OOM. Корпус на диске содержал id старого 262144-пространства (gemma-4-E2B-it),
  а модель имеет vocab_size 32768 (hagi_32k BPE). Первый батч содержал
  id=238288 -> `nn.Embedding(32768, 384)` бросает IndexError внутри
  AOTriton-обёртки, который на ROCm выходит как "unspecified launch failure".
  Проверено прямо: `SourceEncoder(32768, 384)(ids)` -> IndexError.
- [2] Гипотеза "сломан чекпойнт/конфиг" ОТПАЛА: те же конфиг и GPU дали
  exit 0 после починки данных (`python scripts/train.py --steps 3` -> EXIT=0,
  step 0 ce=10.3860).

## Итог: обе задачи закрыты одной причиной
Сломанные данные объясняли И падение HIP, И тихую деградацию микса
(3 источника вместо 6). Починка: `count_unigram.py` (738.1M токенов,
170570 живых id) -> `compact_vocab.py --target-vocab 32768 --drop`
(сохранено 99.1255% массы) -> loader сам подхватил `.compact.bin`
(`dataset.py:62-70`), ids.max() стал 32665, embedding работает.
