#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
from pathlib import Path

DEFAULTS = {
    "RUN_ID": "seq10l_fp16_sliding_eval1408_v0",
    "EVAL_STRIDE": "64",
    "EVAL_BATCH_SEQS": "64",
    "EVAL_SEQ_LEN": "1408",
}


def main() -> None:
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)

    target = (
        Path(__file__).resolve().parents[2]
        / "records"
        / "track_10min_16mb"
        / "2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit"
        / "train_gpt.py"
    )
    if not target.exists():
        raise SystemExit(f"missing delegated trainer: {target}")

    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
