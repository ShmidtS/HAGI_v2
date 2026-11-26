# Окружение рабочего цикла обучения (scripts/neuro_symbolic_cycle.py)

Запуск (PowerShell, из корня проекта), лог — UTF-16 LE:
```
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_MAIN_FREE='1'
$env:HAGI_DEVICE='cuda'; $env:HAGI_ARCH='optimal'; $env:HAGI_TEXT_ONLY='1'; $env:HAGI_SYMBOLIC='1'; $env:PYTORCH_ROCM_ARCH='gfx1151'; $env:GC_PYTHON_THRESHOLD='200'
Start-Process python -ArgumentList '-X','utf8','scripts/neuro_symbolic_cycle.py' -WorkingDirectory C:\HAGI_v2 -RedirectStandardOutput checkpoints\cycle.log -RedirectStandardError checkpoints\cycle_err.log -WindowStyle Hidden
```
Перед перезапуском — копия `checkpoints/unified.pt` в `checkpoints/backup_<дата>/`.
Режим по умолчанию — лучший по замерам (scripts/arch_bench.py, 4 текстовых домена, равный бюджет):
- `optimal`: 21 866 параметров, CE домена 2.25 (hnet 105 088 параметров — 3.28, rope — 3.26);
- `HAGI_TEXT_ONLY=1`: только печатные файлы без дублей (прежний цикл: 1295/1420 доменов — куски диска);
- знания домена — в волокнах CortexFiber с Hedge-смесью (забывание ствола ≈ 0);
- рост — сертифицированными шагами (dirLogitRange), приём по admissionMargin.
Саморазвитие в цикле (включено по умолчанию):
- `evolve` (фаза 5, раз в `HAGI_EVOLVE_EVERY=8` чанков): эволюция алгоритма обучения — популяция
  геномов {lr_muon, lr_adam, batch, momentum, wd, warmup_frac, min_scale, beta2}, непрерывное
  пространство, адаптивная мутация, BLX-скрещивание; гейт — парный по контрольной CE на последних
  чанках; состояние `checkpoints/evolve_state.json`, принятый геном читает `train_optimal`.
- `arch-evolve` (раз в `HAGI_ARCH_EVERY=24`): точное расширение ширины ствола, приём при
  выигрыше CE больше цены параметров (`HAGI_ARCH_LAMBDA`), предел `HAGI_ARCH_MAX`.
- `programs` (символическая половина): изобретённые операторы -> файл программ -> исполнение
  через белый список операций (`hagi/data/safe_exec.py`) -> сертификат равенства.
Монитор: `python -X utf8 scripts/monitor_cycle.py`.
