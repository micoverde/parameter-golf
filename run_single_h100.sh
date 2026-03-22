#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export RUN_ID="${RUN_ID:-pr315_single_h100}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export SEED="${SEED:-2025}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-1800}"
export ITERATIONS="${ITERATIONS:-9000}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3000}"
export NUM_LAYERS="${NUM_LAYERS:-11}"
export MLP_MULT="${MLP_MULT:-3.0}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}"
export BIGRAM_DIM="${BIGRAM_DIM:-128}"
export XSA_LAST_N="${XSA_LAST_N:-4}"
export EMA_ENABLED="${EMA_ENABLED:-1}"
export EMA_DECAY="${EMA_DECAY:-0.997}"
export SWA_ENABLED="${SWA_ENABLED:-0}"
export ROPE_DIMS="${ROPE_DIMS:-16}"
export LN_SCALE="${LN_SCALE:-1}"
export LATE_QAT="${LATE_QAT:-0}"
export QAT_THRESHOLD="${QAT_THRESHOLD:-0.1}"
export MUON_WD="${MUON_WD:-0.04}"
export ADAM_WD="${ADAM_WD:-0.04}"
export MATRIX_LR="${MATRIX_LR:-0.025}"
export SCALAR_LR="${SCALAR_LR:-0.025}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"

if command -v torchrun >/dev/null 2>&1; then
  TORCH_LAUNCHER=(torchrun)
else
  TORCH_LAUNCHER=(python3 -m torch.distributed.run)
fi

"${TORCH_LAUNCHER[@]}" --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/train_gpt.py
