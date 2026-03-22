#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-pr315_single_h100_control}"
export LATE_QAT=0

exec "$SCRIPT_DIR/run_single_h100.sh"
