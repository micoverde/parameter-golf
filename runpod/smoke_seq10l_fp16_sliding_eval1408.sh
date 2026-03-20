#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="${RUN_ID:-smoke_seq10l_fp16_sliding_eval1408}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export ITERATIONS="${ITERATIONS:-20}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export EVAL_BATCH_SEQS="${EVAL_BATCH_SEQS:-64}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-1408}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  experiments/seq10l_fp16_sliding_eval1408/train_gpt.py
