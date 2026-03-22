#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUN_ID="${RUN_ID:-pr374_single_h100_late_qat_restart}"
CHECKPOINT_PATH="${LATE_QAT_RESUME_PATH:-$SCRIPT_DIR/.checkpoints/${BASE_RUN_ID}_late_qat_resume.pt}"

rm -f "$CHECKPOINT_PATH"

RUN_ID="${BASE_RUN_ID}_phase1" \
QAT_ENABLED=0 \
LATE_QAT_SPLIT=1 \
LATE_QAT_RESUME_PATH="$CHECKPOINT_PATH" \
RESUME_PATH="" \
"$SCRIPT_DIR/run_single_h100.sh"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "late_qat_restart: expected checkpoint not found at $CHECKPOINT_PATH" >&2
  exit 1
fi

RUN_ID="${BASE_RUN_ID}_phase2" \
QAT_ENABLED=1 \
LATE_QAT_SPLIT=0 \
LATE_QAT_THRESHOLD=0 \
LATE_QAT_RESUME_PATH="$CHECKPOINT_PATH" \
RESUME_PATH="$CHECKPOINT_PATH" \
WARMUP_STEPS=0 \
"$SCRIPT_DIR/run_single_h100.sh"
