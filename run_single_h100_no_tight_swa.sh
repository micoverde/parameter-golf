#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-pr414_single_h100_no_tight_swa}"
export SWA_ENABLED=0

exec bash "$SCRIPT_DIR/run_single_h100.sh"
