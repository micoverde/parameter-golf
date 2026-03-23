#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RUN_ID="${RUN_ID:-pr569_single_h100_no_prune}"
export PRUNE_PCT="${PRUNE_PCT:-0}"

exec bash "$SCRIPT_DIR/run_single_h100_engineering.sh"
