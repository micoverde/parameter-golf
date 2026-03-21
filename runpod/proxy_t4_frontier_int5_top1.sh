#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="${RUN_ID:-proxy_t4_frontier_int5_top1}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export SEED="${SEED:-42}"
export ITERATIONS="${ITERATIONS:-1000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export EVAL_STRIDE="${EVAL_STRIDE:-0}"
export EVAL_BATCH_SEQS="${EVAL_BATCH_SEQS:-8}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py

