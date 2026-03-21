#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-42 1337 2024}"

for seed in ${SEEDS}; do
  RUN_ID="frontier_int5_top1_seed_${seed}_$(date +%Y%m%d_%H%M%S)" \
  SEED="${seed}" \
  bash "$(dirname "$0")/frontier_int5_top1_seed_8xh100.sh"
done

