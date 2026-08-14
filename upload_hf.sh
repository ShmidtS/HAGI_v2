#!/bin/bash
set -e
REPO="Svyatoblood/HAGI-DeepSeek-V4-Flash-0731-2M"
export PYTHONIOENCODING=utf-8

echo "[1/4] README" && hf upload "$REPO" README_HF.md README.md
echo "[2/4] config.json" && hf upload "$REPO" dsv4_shared_only/config.json config.json
echo "[3/4] skeleton model.safetensors (16GB)" && hf upload "$REPO" dsv4_shared_only/model.safetensors model.safetensors
echo "[4/4] reduced experts (30GB, 43 layers)" && hf upload "$REPO" dsv4_reduced reduced/
echo "[5/5] scripts" && hf upload "$REPO" scripts/ scripts/
echo "ALL UPLOADED"
