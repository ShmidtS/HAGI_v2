# Окружение рабочего цикла обучения (scripts/neuro_symbolic_cycle.py)

Запуск (PowerShell, из корня проекта), лог — UTF-16 LE:
```
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_MAIN_FREE='1'
$env:HAGI_DEVICE='cuda'; $env:HAGI_ARCH='hnet'; $env:HAGI_SYMBOLIC='1'; $env:PYTORCH_ROCM_ARCH='gfx1151'; $env:GC_PYTHON_THRESHOLD='200'
Start-Process python -ArgumentList '-X','utf8','scripts/neuro_symbolic_cycle.py' -WorkingDirectory C:\HAGI_v2 -RedirectStandardOutput checkpoints\cycle.log -RedirectStandardError checkpoints\cycle_err.log -WindowStyle Hidden
```
Перед перезапуском — копия `checkpoints/unified.pt` в `checkpoints/backup_<дата>/`.
Монитор: `python -X utf8 scripts/monitor_cycle.py`.
