#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-pr569_single_h100_no_full_gptq}"
export FULL_GPTQ_ENABLED="${FULL_GPTQ_ENABLED:-0}"

exec bash "$SCRIPT_DIR/run_single_h100_engineering.sh"
