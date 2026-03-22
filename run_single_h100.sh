#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export RUN_ID="${RUN_ID:-pr374_single_h100}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export SEED="${SEED:-1337}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3000}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export NUM_LAYERS="${NUM_LAYERS:-11}"
export MLP_MULT="${MLP_MULT:-3.0}"
export XSA_LAST_N="${XSA_LAST_N:-4}"
export ROPE_DIMS="${ROPE_DIMS:-16}"
export LN_SCALE="${LN_SCALE:-1}"
export SWA_ENABLED="${SWA_ENABLED:-1}"
export SWA_EVERY="${SWA_EVERY:-50}"
export QAT_ENABLED="${QAT_ENABLED:-0}"
export LATE_QAT_THRESHOLD="${LATE_QAT_THRESHOLD:-0.1}"
export LATE_QAT_SPLIT="${LATE_QAT_SPLIT:-0}"
export LATE_QAT_RESUME_PATH="${LATE_QAT_RESUME_PATH:-$SCRIPT_DIR/.checkpoints/${RUN_ID}_late_qat_resume.pt}"
export RESUME_PATH="${RESUME_PATH:-}"
export VE_ENABLED="${VE_ENABLED:-1}"
export VE_DIM="${VE_DIM:-128}"
export VE_LAYERS="${VE_LAYERS:-9,10}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}"
export BIGRAM_DIM="${BIGRAM_DIM:-128}"
export ADAM_WD="${ADAM_WD:-0.04}"
export MUON_WD="${MUON_WD:-0.04}"
export MATRIX_LR="${MATRIX_LR:-0.025}"
export SCALAR_LR="${SCALAR_LR:-0.025}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}"

if command -v torchrun >/dev/null 2>&1; then
  TORCH_LAUNCHER=(torchrun)
else
  TORCH_LAUNCHER=(python3 -m torch.distributed.run)
fi

"${TORCH_LAUNCHER[@]}" --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-03-21_v38_TightSWA/train_gpt.py
