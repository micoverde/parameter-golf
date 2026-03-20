#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
from pathlib import Path

DEFAULTS = {
    "RUN_ID": "wdq_serious_control_v0",
    "TRAIN_SEQ_LEN": "2048",
    "EVAL_SEQ_LEN": "1408",
    "EVAL_STRIDE": "64",
    "WARMDOWN_ITERS": "20000",
    "MATRIX_LR": "0.06",
    "TIED_EMBED_LR": "0.07",
    "SCALAR_LR": "0.06",
    "GRAD_CLIP_NORM": "1.0",
    "MUON_BACKEND_STEPS": "5",
    "MLP_HIDDEN": "992",
}


def main() -> None:
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)

    target = (
        Path(__file__).resolve().parents[2]
        / "records"
        / "track_10min_16mb"
        / "2026-03-19_WarmdownQuantization"
        / "train_gpt.py"
    )
    if not target.exists():
        raise SystemExit(f"missing delegated trainer: {target}")

    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
