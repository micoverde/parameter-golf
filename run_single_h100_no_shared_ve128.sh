#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-pr374_single_h100_no_shared_ve128}"
export VE_ENABLED=0

exec "$SCRIPT_DIR/run_single_h100.sh"
