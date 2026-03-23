#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-pr414_single_h100_schedule_matched}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-1800}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3000}"
export LATE_QAT_THRESHOLD="${LATE_QAT_THRESHOLD:-0.10}"

exec bash "$SCRIPT_DIR/run_single_h100.sh"
