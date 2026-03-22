#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-pr315_single_h100_late_qat}"
export LATE_QAT=1

exec "$SCRIPT_DIR/run_single_h100.sh"
