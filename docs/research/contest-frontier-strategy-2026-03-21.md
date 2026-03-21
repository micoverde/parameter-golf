# Contest Frontier Strategy

Date: 2026-03-21

This branch uses the current public Parameter Golf frontier as the strategic anchor.

Milestones:

1. Reproduce the accepted March 19 WDQ control on contest-grade 8x H100 hardware.
2. Use the public March 20 frontier stack as the mainline path into the low 1.14s.
3. Keep TTT as an optional eval-time bonus lane, not the centerpiece.

Public anchors:

- `records/track_10min_16mb/2026-03-19_WarmdownQuantization`
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`

Lane policy:

- `contest_8xh100`: promotion-relevant only
- `proxy_t4`: proxy-only, used for directionality, runtime, export robustness, and artifact-size screening

Current practical use:

- use Azure T4 to screen proxy deltas and catch broken paths
- use 8x H100 only for matched serious runs or record-attempts

Immediate branch goals:

- maintain exact 8x H100 launchers for the public frontier stack
- maintain cheap T4 proxy launchers for the same script family
- keep the WDQ STE-QAT experiment isolated as a candidate treatment
