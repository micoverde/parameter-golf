# Parameter Golf Wild Frontier

Date: March 20, 2026

## Objective

Move from the fast smoke lane, which is currently hovering near `3.46` bpb, to
a serious lane that can realistically hit `<= 2.0` and then explore wild ideas
against a strong control.

## Current Ground Truth

- Fast ARENA champion:
  - [experiments/champions/current_champion.json](/home/warrenjo/src/parameter-golf/experiments/champions/current_champion.json)
  - useful for cheap direction-finding, not for score targets like `2.0`
- Strongest checked-in record in this repo:
  - [records/track_10min_16mb/2026-03-19_WarmdownQuantization/submission.json](/home/warrenjo/src/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/submission.json)
  - `val_bpb = 1.15744040`
- Current detached smoke battle:
  - `b11a` and `b11b`
  - treatment = `train1408_maxbatch`

## 10-Lens Research Synthesis

Archived specialist passes:

- Long-context:
  - `train1408_maxbatch` is the cleanest same-family follow-up to `train1408`
- Quantization:
  - sensitivity-aware mixed precision remains attractive, especially around
    output-path tensors
- Schedule:
  - `sched400` is the safe schedule-only follow-up, not `sched20`
- Loss/data:
  - masking `BOS` targets is a clean orthogonal loss-side bet

Live researcher passes:

- Architecture:
  - best bounded wild branch is `5x2` shared-depth with tiny loop-specific LoRA
- Eval/test-time compute:
  - best high-upside wild branch is porting LoRA TTT eval onto the stronger
    current backbone
- Systems:
  - best correctness-preserving speed lever is exact cached sliding-window eval
- Orchestration:
  - detached background battle execution is the stable default path
- Compression:
  - layout-adaptive export is a clean next codec-side idea
- Symmetry:
  - mirrored or phase-aware tying is still a viable structural branch

## Ranked Frontier

1. Serious control lane: `WarmdownQuantization`
   - [records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py](/home/warrenjo/src/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py)
   - reason: repo-proven path to far below `2.0`

2. Wild score branch: `WarmdownQuantization + LoRA TTT eval`
   - base:
     [records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py](/home/warrenjo/src/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py)
   - evaluator to port:
     [records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py](/home/warrenjo/src/parameter-golf/records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py)
   - reason: biggest plausible score jump from repo-native ideas

3. Wild architecture branch: `5x2 loop-lora maxbatch`
   - base:
     [records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py](/home/warrenjo/src/parameter-golf/records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py)
   - recurrence ingredients:
     [records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py](/home/warrenjo/src/parameter-golf/records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py)

4. Eval systems branch: exact cached sliding-window eval
   - reason: not a score trick by itself, but it makes wild eval-heavy ideas
     cheaper and cleaner to test

5. Codec branch: layout-adaptive export on quantized tensors
   - reason: bounded, challenge-legal, and synergistic with near-cap models

## Immediate Sequence

1. Keep the detached smoke ARENA loop running for cheap attribution.
2. Promote a serious control lane around `WarmdownQuantization`.
3. Build the first truly wild branch against that serious lane:
   `WarmdownQuantization + LoRA TTT eval`.
4. Keep `BOS` masking, `sched400`, and layout-adaptive export as orthogonal
   side branches, not mixed into the same treatment.

## New Files In This Branch

- [experiments/wdq_serious_control/train_gpt.py](/home/warrenjo/src/parameter-golf/experiments/wdq_serious_control/train_gpt.py)
- [experiments/wdq_serious_control/README.md](/home/warrenjo/src/parameter-golf/experiments/wdq_serious_control/README.md)
- [runpod/serious_wdq_control_single_h100.sh](/home/warrenjo/src/parameter-golf/runpod/serious_wdq_control_single_h100.sh)
