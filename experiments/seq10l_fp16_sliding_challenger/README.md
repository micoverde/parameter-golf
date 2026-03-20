# seq10l fp16 sliding challenger

This experiment wrapper delegates to the record-derived 10-layer fp16 tied-embedding
sliding-window recipe and sets canary-friendly defaults for ARENA-style battles.

Default behavior:
- `EVAL_STRIDE=64`
- `EVAL_BATCH_SEQS=64`

The actual trainer lives in:
- `records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py`
