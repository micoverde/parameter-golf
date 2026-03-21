#!/usr/bin/env bash
set -euo pipefail

export RUN_ID="${RUN_ID:-frontier_int5_top1_8xh100}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export SEED="${SEED:-42}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py

