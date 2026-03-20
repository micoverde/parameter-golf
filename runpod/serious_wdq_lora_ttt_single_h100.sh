#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="${RUN_ID:-serious_wdq_lora_ttt_single_h100}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export ITERATIONS="${ITERATIONS:-20000}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-1800}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export EVAL_STRIDE="${EVAL_STRIDE:-256}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-20000}"
export MATRIX_LR="${MATRIX_LR:-0.02}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.03}"
export SCALAR_LR="${SCALAR_LR:-0.02}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"
export MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
export MLP_HIDDEN="${MLP_HIDDEN:-1536}"
export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-2048}"
export TTT_CHUNK_SIZE="${TTT_CHUNK_SIZE:-256}"
export TTT_BATCH_SIZE="${TTT_BATCH_SIZE:-16}"
export TTT_LORA_RANK="${TTT_LORA_RANK:-8}"
export TTT_LORA_LR="${TTT_LORA_LR:-0.01}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py
