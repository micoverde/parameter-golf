# seq10l fp16 train1408

This experiment wrapper reuses the record-derived 10-layer fp16 tied-embedding
sliding-window recipe and changes the learned context to `1408` tokens while
keeping sliding-window evaluation enabled.

Default behavior:
- `TRAIN_SEQ_LEN=1408`
- `TRAIN_BATCH_TOKENS=518144`
- `EVAL_SEQ_LEN=1408`
- `EVAL_STRIDE=64`
- `EVAL_BATCH_SEQS=128`

The actual trainer lives in:
- `records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py`
