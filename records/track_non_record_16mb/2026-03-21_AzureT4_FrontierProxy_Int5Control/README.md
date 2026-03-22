This record captures our strongest completed proxy-control run for the current frontier family.

This is a `non-record` submission and is not intended for leaderboard acceptance. It is a development/proxy run used to screen frontier ideas before spending scarce `8xH100` budget. The main differences from leaderboard-valid runs are:

- hardware is `1x NVIDIA Tesla T4 16GB`, not `8xH100`
- validation is capped with `VAL_MAX_SEQS=512`, so the metric is directional only
- this run is meant to demonstrate reproducible artifact sizing and post-quant roundtrip behavior, not a contest-grade score

Why submit this anyway:

- the artifact is real and under the `16,000,000` byte cap
- the run cleanly exercises the current frontier family end-to-end
- it gives a reproducible proxy baseline for causal ARENA-style treatment battles
- it exposed a measurable post-quant degradation (`+0.0488 bpb`) that motivated our next QAT treatment

Configuration:

- Track: `non-record`, proxy-development, still under the `16,000,000` byte artifact cap
- Base family: March 20 frontier `10L Int5-MLP + BigramHash(10240) + SWA + WD=0.04`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- Bigram path: `BIGRAM_VOCAB_SIZE=10240 BIGRAM_DIM=128`
- Batching: `TRAIN_BATCH_TOKENS=32768 TRAIN_SEQ_LEN=2048`
- Validation: first `512` sequences from `fineweb_val_*`
- Export: mixed quantized artifact printed as `int6+zstd` in the log, total counted submission bytes `15,536,094`

Command:

```bash
RUN_ID=proxy_t4_frontier_int5_top1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SEED=42 \
ITERATIONS=100 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=32768 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=25 \
EVAL_STRIDE=0 \
EVAL_BATCH_SEQS=8 \
PROXY_T4_COMPAT=1 \
VAL_MAX_SEQS=512 \
torchrun --standalone --nproc_per_node=1 \
records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py
```

Key metrics from `train.log`:

- pre-quant eval at stop: `val_loss:5.2741`, `val_bpb:3.1599`
- post-quant roundtrip eval: `val_loss:5.35550585`, `val_bpb:3.20873382`
- post-quant gap: `+0.04883382 bpb`
- train time: `507950ms`
- eval time: `66629ms`
- peak memory: `5289 MiB allocated`, `7154 MiB reserved`
- serialized model: `15482160 bytes`
- code size: `53934 bytes`
- total submission size: `15536094 bytes`

Included files:

- `train_gpt.py`: exact code snapshot used for the run
- `train.log`: exact Azure T4 training log
- `submission.json`: metadata for this non-record proxy submission

Notes:

- This run is also logged in MLflow experiment `MLFLOW ARENA 2026-03-21 Golf Frontier Proxy`.
- We used this proxy control as the pinned control for subsequent STE-QAT treatment battles.
