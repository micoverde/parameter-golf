# WDQ Int6 STE-QAT

This experiment starts from the accepted March 19 WDQ backbone and adds one new idea:

- late-phase straight-through-estimator fake quantization on `CastedLinear` weights

The goal is to make the exported int6 matrices more robust without changing the core
architecture or switching record families.

Key choices:

- base backbone: `records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py`
- export target stays int6
- QAT defaults to activating only in the late tail via `QAT_START_SCALE`
- final-2 `c_k.weight` layers remain exempt, matching the record's late-K fp16 passthrough idea

Relevant env vars:

- `QAT_ENABLE`
- `QAT_BITS`
- `QAT_START_SCALE`

Launchers:

- `runpod/wdq_int6_ste_qat_8xh100.sh`
- `runpod/wdq_int6_ste_qat_single_h100.sh`
