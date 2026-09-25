#!/usr/bin/env bash
# Train the six N=6 domain experts sequentially.
#
# Sequential by standing project policy: never run two trainings at once.
# Each expert trains on a different corpus mix, so the merged model at N=6
# gets six genuinely distinct initialisations rather than six slices of the
# same distribution.
set -u
cd "$(dirname "$0")/.."

EXPERTS="e1_wiki_ru e2_web_en e3_math e4_wiki_en e5_tinystory e6_ru_mix"

for name in $EXPERTS; do
  cfg="configs/m2_e6_${name}.yaml"
  log="logs/m2_e6_${name}.log"
  if [ -f "checkpoints/m2_e6_${name}/step-0003000.pt" ]; then
    echo "== ${name}: already trained, skipping"
    continue
  fi
  echo "== ${name}: training -> ${log}"
  python scripts/train.py --config "${cfg}" --device cuda > "${log}" 2>&1
  code=$?
  if [ ${code} -ne 0 ]; then
    echo "!! ${name} failed with exit ${code}; stopping the chain" >&2
    tail -5 "${log}" >&2
    exit ${code}
  fi
  echo "== ${name}: done"
done

echo "all six experts trained"
