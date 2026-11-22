Set-Location C:\HAGI_v2
$env:OPENBLAS_NUM_THREADS="1"; $env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"
$env:OPENBLAS_MAIN_FREE="1"
$env:HAGI_DEVICE="cuda"
# WAVE 218/219: semantic distill on (gemma -> ternary LM)
$env:HAGI_SEMANTIC="1"
# WAVE 205: ROCm allocator hygiene — expandable segment reduces
# fragmentation over multi-hour runs (measured: chunk 700s->980s,
# HIP failure after ~1h on iGPU)
$env:PYTORCH_ROCM_ARCH="gfx1151"
$env:PYTORCH_NO_CUDA_MEMORY_CACHING="0"
$env:GC_PYTHON_THRESHOLD="200"
python -X utf8 scripts/restore_domains.py *> C:\HAGI_v2\checkpoints\restore.log
