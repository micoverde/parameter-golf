#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export RUN_ID="${RUN_ID:-pr569_single_h100}"
export SEED="${SEED:-1337}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export BACKOUT_ENABLED="${BACKOUT_ENABLED:-0}"

if command -v torchrun >/dev/null 2>&1; then
  TORCH_LAUNCHER=(torchrun)
else
  TORCH_LAUNCHER=(python3 -m torch.distributed.run)
fi

"${TORCH_LAUNCHER[@]}" --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-03-23_11L_VRL_FullGPTQ_LeakyReLU2_1.1175/train_gpt.py
