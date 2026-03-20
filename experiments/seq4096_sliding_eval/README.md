# Seq4096 + Sliding Eval Candidate

This tracked experiment is a thin wrapper around the accepted in-repo `SlidingWindowEval` trainer with the `TrainingOptSeq4096` defaults layered in through environment variables.

Purpose:

- keep implementation small and reviewable
- reuse a known-good sliding-window evaluation path
- reuse the cleaner `seq_len=4096` training regime without copying a large trainer snapshot yet

Current default recipe:

- `TRAIN_SEQ_LEN=4096`
- `TRAIN_BATCH_TOKENS=393216`
- `TIED_EMBED_LR=0.030`
- `MATRIX_LR=0.020`
- `SCALAR_LR=0.020`
- `MUON_MOMENTUM=0.99`
- `MUON_MOMENTUM_WARMUP_START=0.92`
- `MUON_MOMENTUM_WARMUP_STEPS=1500`
- `WARMDOWN_ITERS=3000`
- `EVAL_STRIDE=64`
- `EVAL_BATCH_SEQS=256`
- `QAT=0`
- `NUM_LOOPS=1`
- `LORA_RANK=0`

Run locally or on RunPod with:

```bash
RUN_ID=seq4096_sliding_eval_v0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 \
  experiments/seq4096_sliding_eval/train_gpt.py
```

This is a research scaffold, not a contest submission folder yet.
