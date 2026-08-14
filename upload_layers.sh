#!/bin/bash
REPO="Svyatoblood/HAGI-DeepSeek-V4-Flash-0731-2M"
export PYTHONIOENCODING=utf-8
for L in $(seq 0 42); do
  d="dsv4_reduced/layer_${L}"
  if [ -d "$d" ]; then
    echo "=== layer ${L} ==="
    for attempt in 1 2 3; do
      if hf upload "$REPO" "$d" "reduced/layer_${L}/" >> hf_upload_layers.log 2>&1; then
        echo "layer ${L} OK" >> hf_upload_layers.log
        break
      else
        echo "layer ${L} attempt ${attempt} FAILED, retrying" >> hf_upload_layers.log
        sleep 10
      fi
    done
  fi
done
echo "ALL LAYERS DONE" >> hf_upload_layers.log
