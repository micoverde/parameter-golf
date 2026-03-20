# seq10l fp16 sliding eval1408

This experiment wrapper reuses the record-derived 10-layer fp16 tied-embedding
sliding-window recipe and overrides evaluation defaults for a moderate-context
`1408`-token sliding-window canary.

Default behavior:
- `EVAL_SEQ_LEN=1408`
- `EVAL_STRIDE=64`
- `EVAL_BATCH_SEQS=64`

The actual trainer lives in:
- `records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py`
