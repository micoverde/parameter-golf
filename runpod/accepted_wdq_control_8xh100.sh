#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="${RUN_ID:-accepted_wdq_control_8xh100}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-5000}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export EVAL_STRIDE="${EVAL_STRIDE:-256}"
export ROPE_TRAIN_SEQ_LEN="${ROPE_TRAIN_SEQ_LEN:-1024}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-20000}"
export MATRIX_LR="${MATRIX_LR:-0.02}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.03}"
export SCALAR_LR="${SCALAR_LR:-0.02}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"
export MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
export MLP_HIDDEN="${MLP_HIDDEN:-1536}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py
