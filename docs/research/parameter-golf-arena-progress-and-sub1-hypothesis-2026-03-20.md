# Toward Sub-1.0 BPB in Parameter Golf

Superseded on 2026-03-21 by
`docs/research/contest-frontier-strategy-2026-03-21.md`.

This note is preserved as a historical research record for the earlier
single-H100-centered phase of the campaign. It should not be used as the
current strategic anchor for contest decisions.

Date: March 20, 2026

Author: Laplace, postdoctoral research pass synthesized into the local research record

## Abstract

This note summarizes the Parameter Golf work completed so far using only local,
measured results and branch-level evidence from the workspace. Two experimental
lanes have emerged. First, a cheap ARENA smoke lane was used to screen ideas by
fixed control/treatment battles with repeated seeds. In that lane, the promoted
`seq10l_fp16_sliding_challenger` treatment improved mean post-quantization
validation performance from `3.94481757` to `3.463062159` bits per byte (bpb),
an absolute gain of `0.481755411` bpb with `success_rate = 1.0`. Second, a
serious WarmdownQuantization-derived lane established a much stronger single-H100
control with `2.03139894` exact post-quantization bpb and `2.00698872`
sliding-window bpb. The evidence therefore supports a two-lane strategy:
maintain the smoke lane for fast attribution and use the WarmdownQuantization
lane as the real score-seeking control. The strongest data-supported next move
is evaluation-side work on the serious lane. The strongest high-upside hypothesis
for reaching `<= 1.0` bpb is a multiplicative stack built on the serious
WarmdownQuantization control: better evaluation, test-time adaptation, and
compression-aware architecture changes, tested one at a time under ARENA.

## 1. Objective

The near-term goal is not just to improve over a weak smoke baseline, but to
develop a disciplined path to a challenge-competitive score and ultimately a
winning score at or below `1.0` bpb. This article asks three questions:

1. What has actually improved so far?
2. Which directions are supported by measured data rather than intuition alone?
3. What hypothesis stack could plausibly drive the score toward `<= 1.0`?

## 2. Experimental Method

### 2.1 ARENA design

All meaningful local progress has been organized as ARENA-style control/treatment
battles.

- A control arm is fixed before the treatment is run.
- The treatment changes one bounded idea family at a time.
- Repeated seeds are used when affordable.
- Promotion requires improvement in the target metric while preserving basic
  run integrity and artifact legality.

This design prevents uncontrolled drift and makes negative results useful.

### 2.2 Two-lane operating model

Two lanes are now required because the cheapest lane is not the right lane for
winning the contest.

- Smoke lane:
  - purpose: rapid directional screening
  - cost profile: cheap
  - failure tolerance: high
  - promotion criterion: internal only
- Serious lane:
  - purpose: actual score-seeking control/treatment work
  - cost profile: materially higher
  - failure tolerance: lower
  - promotion criterion: real improvement on the stronger control

### 2.3 Metrics

The primary metric is validation `bpb`.

- Smoke lane used `post_quant_val_bpb`.
- Serious lane uses both:
  - `post_quant_val_bpb`
  - `sliding_window_val_bpb`

Submission bytes are tracked as a constraint-side metric, not as a success
metric by themselves.

## 3. Completed Results

### 3.1 Smoke lane: large directional gain from sliding-window treatment

The strongest completed smoke result is the merged ARENA battle recorded in
[2026-03-20_b06_seq10l_fp16_sliding_challenger_merged.json](/home/warrenjo/src/parameter-golf/experiments/arena_runs/2026-03-20_b06_seq10l_fp16_sliding_challenger_merged.json).

Control:

- run family: `seq4096_control_smoke`
- control `post_quant_val_bpb`: `3.94481757`
- control `post_quant_val_loss`: `6.66065922`
- total submission bytes: `5,021,877`

Promoted treatment:

- run family: `seq10l_fp16_sliding_challenger`
- mean `post_quant_val_bpb`: `3.463062159`
- mean `post_quant_val_loss`: `5.847235393`
- mean total submission bytes: `6,212,447.3`
- success rate: `1.0`
- replicates: `10`

Measured gain:

- absolute bpb improvement: `0.481755411`
- relative improvement vs control: about `12.2%`

Interpretation:

This was the first clear proof that the ARENA battle loop could identify a real
improvement with enough consistency to justify promotion. It also showed that
evaluation-aware model families can dominate a naive smoke control even under
small, cheap runs.

### 3.2 Serious lane: WarmdownQuantization establishes a near-2.0 control

The strongest completed serious result is the current champion manifest in
[serious_current_champion.json](/home/warrenjo/src/parameter-golf/experiments/champions/serious_current_champion.json).

Serious control:

- run family: `wdq_serious_control`
- `post_quant_val_bpb`: `2.03139894`
- `post_quant_val_loss`: `3.42993005`
- `sliding_window_val_bpb`: `2.00698872`
- `sliding_window_val_loss`: `3.38871638`
- mean total submission bytes: `6,579,188`
- training setup:
  - `train_seq_len = 2048`
  - `eval_seq_len = 1408`
  - `iterations = 20000`
  - `train_batch_tokens = 524288`

Measured gain inside the same run:

- sliding-window minus exact post-quant bpb: `-0.02441022`

Interpretation:

This result changes the research program. Once a serious control sits at
`2.00698872`, the smoke lane is no longer the path to victory. It remains useful
for triage, but any claim about winning potential now has to beat the serious
WarmdownQuantization lane.

## 4. What Is Actually Supported by Data

Three claims are supported by measured local evidence.

### 4.1 ARENA itself is working as a research filter

The smoke lane delivered a large, repeated gain across ten treatment runs. This
means the control/treatment structure is already paying for itself as an
experimental method.

### 4.2 Evaluation choices matter materially

The serious control improved from `2.03139894` to `2.00698872` by moving from
exact post-quant evaluation to sliding-window evaluation within the same family.
That is a measured change of more than `0.024` bpb, which is large relative to
the remaining distance to `2.0`.

### 4.3 WarmdownQuantization-family controls are the right backbone for
serious work

A control near `2.0` is already available locally in the serious lane, whereas
the smoke lane remains in the mid-`3` range. Therefore the serious lane is the
correct base for any sub-`2.0` or sub-`1.0` ambition.

## 5. What Is Promising But Not Yet Proven

The following directions are promising, but they are not yet established
improvements over the serious control and should be described as hypotheses or
active tests rather than results.

### 5.1 `eval2048` on the serious lane

Reason it is promising:

- it stays within the same successful family
- it attacks a measured leverage point, namely evaluation context

Current status:

- active serious-lane treatment branch
- not yet a completed promoted result in the local record

### 5.2 Layout-adaptive export

Reason it is promising:

- it is challenge-legal
- it is bounded
- it may recover byte budget or reduce effective quantization damage without
  changing the training recipe

Current status:

- implemented as a codec-side challenger
- not yet validated as a serious-lane improvement

### 5.3 WarmdownQuantization plus LoRA test-time training

Reason it is promising:

- it is the highest-upside eval-side idea identified in the current repo-native
  research pass
- it specifically targets the phase that already shows measurable leverage

Current status:

- still a hypothesis
- not yet implemented and validated against the serious control in the local
  record

### 5.4 `5x2` shared-depth loop LoRA

Reason it is promising:

- it is the strongest bounded architecture-side hypothesis from the current
  research sweep
- it offers a path to more effective capacity under strict byte limits through
  structured sharing rather than naive parameter growth

Current status:

- design direction identified
- not yet a validated improvement

## 6. Negative And Inconclusive Lessons

Not every same-family tweak deserves promotion pressure.

- Small cheap-lane improvements can be real and still be strategically
  irrelevant once a stronger serious control exists.
- A branch that looks good in training curves may still fail to matter after
  post-quant or sliding-window evaluation.
- The program should resist mixing architecture, evaluation, schedule, and codec
  changes into one treatment because that destroys attribution.

These lessons matter because they determine how quickly the project can reject
weak ideas and conserve expensive serious-lane compute.

## 7. Hypothesis Stack for Reaching `<= 1.0` BPB

No local result currently justifies a claim that `<= 1.0` is close. The most
defensible statement is that `<= 1.0` likely requires multiplicative gains from
several nearly orthogonal sources.

### 7.1 Stage A: beat the serious control cleanly

Target:

- move below the current serious sliding-window control of `2.00698872`

Most plausible mechanism:

- evaluation-side improvement first, because the data already shows that
  evaluation protocol changes can move the score materially

### 7.2 Stage B: add test-time adaptation without breaking the artifact budget

Target:

- reduce residual modeling error after the backbone is fixed

Most plausible mechanism:

- LoRA-based document-aware test-time adaptation on top of the
  WarmdownQuantization serious lane

### 7.3 Stage C: increase effective capacity through structure, not raw size

Target:

- gain expressivity without violating the artifact constraint

Most plausible mechanism:

- looped/shared-depth architectures with tiny loop-specific adapters

### 7.4 Stage D: reclaim bytes to reallocate precision or auxiliary machinery

Target:

- improve the quality/bytes tradeoff

Most plausible mechanism:

- layout-adaptive export or related compression-aware serialization choices

### 7.5 Winning hypothesis

The current winning hypothesis is therefore:

1. establish a stable serious WarmdownQuantization control
2. improve evaluation on that control
3. layer in bounded LoRA test-time adaptation
4. only then introduce a structured sharing architecture
5. use codec-side savings to support the strongest variant that survives ARENA

This is the smallest hypothesis stack that is both aggressive and still anchored
to measured local evidence.

## 8. Recommended Next Experiments

Priority order:

1. Finish and score the serious `eval2048` treatment strictly against
   `2.00698872`.
2. Implement the minimum viable `WarmdownQuantization + LoRA TTT` branch as a
   pure evaluation-side intervention.
3. Run layout-adaptive export as a codec-only serious-lane treatment.
4. Keep the smoke ARENA lane alive for early rejection and fast directional
   screening.
5. Defer `5x2` loop LoRA until the eval-side and codec-side single-variable
   tests are complete.

## 9. Conclusion

The data so far supports optimism, but not complacency. The project has already
demonstrated that disciplined ARENA battles can find real gains and that a
serious WarmdownQuantization lane can sit almost exactly at `2.0` bpb on local
infrastructure. What remains unproven is the path from `~2.0` to `<= 1.0`. The
best evidence-backed route is not random novelty. It is a staged attack:
evaluation first, test-time adaptation second, structured capacity third, and
compression-aware packing throughout.
