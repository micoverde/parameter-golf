#!/usr/bin/env bash
set -euo pipefail

# Step-matched reproduction of the accepted March 19 WDQ record.
# The original accepted log reached ~7199 steps in ~600s on slower 8xH100 hardware.
# Our first RunPod reproduction hit ~7401 steps in 600s and regressed badly, so this
# launcher trims training wallclock to keep the optimizer schedule closer to the
# accepted run while staying within the contest's 10-minute training cap.

export RUN_ID="${RUN_ID:-accepted_wdq_control_8xh100_stepmatch}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-584}"

exec "$(dirname "$0")/accepted_wdq_control_8xh100.sh"
