#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
from pathlib import Path

DEFAULTS = {
    "RUN_ID": "seq4096_sliding_eval_v0",
    "TRAIN_SEQ_LEN": "4096",
    "TRAIN_BATCH_TOKENS": "393216",
    "TIED_EMBED_LR": "0.030",
    "MATRIX_LR": "0.020",
    "SCALAR_LR": "0.020",
    "MUON_MOMENTUM": "0.99",
    "MUON_MOMENTUM_WARMUP_START": "0.92",
    "MUON_MOMENTUM_WARMUP_STEPS": "1500",
    "WARMDOWN_ITERS": "3000",
    "EVAL_STRIDE": "64",
    "EVAL_BATCH_SEQS": "256",
    "QAT": "0",
    "NUM_LOOPS": "1",
    "LORA_RANK": "0",
}


def main() -> None:
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)

    target = (
        Path(__file__).resolve().parents[2]
        / "records"
        / "track_10min_16mb"
        / "2026-03-19_SlidingWindowEval"
        / "train_gpt.py"
    )
    if not target.exists():
        raise SystemExit(f"missing delegated trainer: {target}")

    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
