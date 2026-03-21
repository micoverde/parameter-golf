# Contest Frontier Strategy

Date: 2026-03-21

This document is now the authoritative strategy note for the local
`parameter-golf` campaign. It supersedes the older single-H100-centered
research framing and aligns the repo with the current public leaderboard.

## Strategic anchor

The correct ordering is:

1. reproduce a contest-grade `8x H100` baseline
2. port and validate the public leaderboard stack
3. keep TTT as an optional bonus lane, not the centerpiece

The main correction is simple: the raw gap from the old single-H100 proxy
control to the leaderboard was mostly a compute-regime mismatch, not proof
that the underlying research program was wrong.

## Public anchors

Current public anchors in the repo:

- `records/track_10min_16mb/2026-03-19_WarmdownQuantization`
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`

Current public score landmarks:

- accepted WDQ anchor: about `1.2154`
- current public leader: about `1.1428`
- practical gap from WDQ anchor to current leader: about `0.0726` bpb

This means the serious contest objective is no longer "beat `2.0`." It is:

- first, reproduce the accepted WDQ family cleanly on contest hardware
- second, move from the low `1.21s` into the mid `1.14s`
- third, only then reserve a moonshot lane for `<= 1.0`

## Lane policy

Two lanes remain necessary, but their status is explicit:

- `contest_8xh100`
  - promotion-relevant only
  - the only lane that can be compared directly to public scores
  - requires fixed control/treatment runs and leaderboard-like evaluation
- `proxy_t4`
  - proxy-only
  - used for directionality, export robustness, runtime sanity, artifact-size
    checks, and broken-path debugging
  - must not be used for leaderboard claims or serious promotions

Required policy implication:

- T4 results can justify "safe to escalate" or "do not escalate"
- T4 results cannot justify "this is better than the public board"

## Mainline stack

The public board is no longer signaling "TTT first." It is signaling
capacity-per-byte first.

The current public stack to import and validate is:

- mixed `int5` MLP / `int6` attention-style quantization
- `10` or `11` layers
- `MLP3x`
- `BigramHash`
- `SmearGate`
- `Muon` with nontrivial weight decay
- `SWA`
- orthogonal init
- compression-conscious export such as `zstd-22`
- sliding or stride-aware eval where it is actually positive on the trained
  regime

The public evidence so far suggests:

- `MLP3x` is a major contributor
- mixed precision is buying real extra capacity
- `BigramHash` and `SmearGate` are part of the winning recipe
- TTT is real but currently not the dominant leaderboard line

## Artifact budget thesis

The strongest structural criticism of the older local plan is correct: the old
serious proxy left too much of the `16,000,000` byte cap unused. Frontier runs
are using almost all of the budget, while the old local serious proxy sat near
`6.58M` total bytes.

Therefore:

- spare bytes are not "comfort"
- spare bytes are lost capacity
- the default objective should be to use most of the artifact budget
  deliberately, not to leave large idle headroom

## Contest-grade milestones

### Milestone 1: clean contest baseline

Reproduce the accepted March 19 WDQ family on `8x H100`.

Success condition:

- land near the accepted contest-grade score band
- confirm that export degradation is normal rather than catastrophic

Failure implication:

- if reproduction is still materially wrong, fix schedule/export robustness
  before trying to stack new ideas

### Milestone 2: public frontier stack

Run the imported March 20 frontier stack as the serious mainline.

Success condition:

- stable training and export
- artifact bytes close to the cap
- score movement into the public frontier band

### Milestone 3: optional bonus lanes

Only after the mainline is stable should bonus lanes compete for serious
compute, including:

- WDQ STE-QAT as an export-robustness treatment
- TTT as an eval-time option on top of a strong base
- later moonshots for sub-`1.0`

## Azure T4 operating plan

Azure T4 is useful right now because it can eliminate broken paths before the
next paid `8x H100` run.

What T4 is good for:

- does the script run end to end
- does export roundtrip work
- does `QAT` reduce the proxy quantization gap
- do `BigramHash` and `SmearGate` integrate cleanly
- does the full stack stay under the artifact cap

What T4 is not good for:

- wallclock schedule matching
- final score claims
- leaderboard comparisons

The T4 proxy lane should prefer short, bounded runs that answer:

- does this treatment break training
- does it bloat the artifact
- does it help or hurt roundtrip robustness

## Immediate repo goals

- maintain exact `8x H100` launchers for contest-grade reproduction and frontier
  runs
- maintain cheap T4 proxy launchers for the same script families
- keep `wdq_int6_ste_qat` isolated as a serious candidate treatment
- treat TTT as a flaggable side lane, not the core branch architecture

## Immediate run order

1. finish Azure T4 environment setup
2. run a proxy check on the imported public frontier stack
3. run the proxy WDQ STE-QAT treatment
4. use those results to decide the next paid `8x H100` run
