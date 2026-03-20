# wdq_serious_control

Serious-control wrapper around the strongest checked-in record family in this
repo: `2026-03-19_WarmdownQuantization`.

Default intent:

- train on the stronger `train@2048` / `eval@1408` family
- use the warmdown-for-compression schedule from the record
- preserve the record's `int6 + fp16 tied embedding + late-K passthrough`
  export path

This is not a new record submission. It exists to establish a realistic
sub-`2.0` control lane on available RunPod H100s while the fast ARENA smoke lane
continues exploring smaller ideas.
