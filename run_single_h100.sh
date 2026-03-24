#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export RUN_ID="${RUN_ID:-pr609_single_h100}"
export SEED="${SEED:-1337}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export TARGET_MB="${TARGET_MB:-15.9}"

python3 - <<'PY'
from flash_attn_interface import flash_attn_func  # noqa: F401
print("FA3 OK")
PY

if command -v torchrun >/dev/null 2>&1; then
  TORCH_LAUNCHER=(torchrun)
else
  TORCH_LAUNCHER=(python3 -m torch.distributed.run)
fi

"${TORCH_LAUNCHER[@]}" --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-03-24_XSA-all_FullGPTQ_ParallelMuon_1.1155/train_gpt.py
