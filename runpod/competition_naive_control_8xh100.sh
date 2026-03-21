#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="${RUN_ID:-competition_naive_control_8xh100}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py
