#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-pr569_single_h100_no_qat_align}"
export QAT_CLIP_PCT="${QAT_CLIP_PCT:-1.0}"

exec bash "$SCRIPT_DIR/run_single_h100_engineering.sh"
