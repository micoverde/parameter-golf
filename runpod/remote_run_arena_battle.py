#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path

WORKSPACE = Path("/workspace")
REPO_DIR = WORKSPACE / "parameter-golf"
REPO_URL = os.environ.get("REPO_URL", "https://github.com/micoverde/parameter-golf.git")
BRANCH = os.environ.get("BRANCH", "feature/arena-battle-loop")
TRAIN_SHARDS = os.environ.get("TRAIN_SHARDS", "1")
PASSTHROUGH_ENV_KEYS = [
    "CHAMPION_PATH",
    "LANE_NAME",
    "DATASET_VARIANT",
    "TOKENIZER_PATH",
    "TRAIN_SEQ_LEN",
    "FIXTURE_TRAIN_SEQ_LEN",
    "EVAL_SEQ_LEN",
    "FIXTURE_EVAL_SEQ_LEN",
    "TRAIN_BATCH_TOKENS",
    "MATRIX_LR",
    "SCALAR_LR",
    "TIED_EMBED_LR",
    "MUON_MOMENTUM",
]


def run(cmd: list[str], cwd: Path | None = None, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    prefix = f"[cwd={cwd}] " if cwd else ""
    print(f"$ {prefix}{' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_repo() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not REPO_DIR.exists():
        run(["git", "clone", REPO_URL, str(REPO_DIR)])
    else:
        run(["git", "remote", "set-url", "origin", REPO_URL], cwd=REPO_DIR)
    run(["git", "fetch", "origin"], cwd=REPO_DIR)
    run(["git", "checkout", BRANCH], cwd=REPO_DIR)
    run(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=REPO_DIR)


def ensure_data() -> None:
    run(
        [
            "python3",
            "data/cached_challenge_fineweb.py",
            "--variant",
            "sp1024",
            "--train-shards",
            TRAIN_SHARDS,
        ],
        cwd=REPO_DIR,
    )


def run_battle() -> None:
    battle_env = {
        "POD_ID": os.environ.get("POD_ID", ""),
        "GPU_TYPE": os.environ.get("GPU_TYPE", ""),
        "BRANCH": BRANCH,
        "BATTLE_ID": os.environ.get("BATTLE_ID", "b03"),
        "ITERATIONS": os.environ.get("ITERATIONS", "20"),
        "TRAIN_SHARDS": TRAIN_SHARDS,
        "EVAL_BATCH_SEQS": os.environ.get("EVAL_BATCH_SEQS", "64"),
        "EVAL_STRIDE": os.environ.get("EVAL_STRIDE", "64"),
        "TREATMENT_SCRIPT": os.environ.get(
            "TREATMENT_SCRIPT",
            "runpod/smoke_seq10l_fp16_sliding_challenger.sh",
        ),
        "TREATMENT_NAME": os.environ.get(
            "TREATMENT_NAME",
            "seq10l_fp16_sliding_challenger",
        ),
        "TREATMENT_TARGET": os.environ.get(
            "TREATMENT_TARGET",
            "experiments/seq10l_fp16_sliding_challenger/train_gpt.py",
        ),
        "SEEDS": os.environ.get(
            "SEEDS",
            "1337,1338,1339,1340,1341,1342,1343,1344,1345,1346",
        ),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        ),
    }
    for key in PASSTHROUGH_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None:
            battle_env[key] = value
    run(
        ["python3", "runpod/run_arena_treatment_battle.py"],
        cwd=REPO_DIR,
        extra_env=battle_env,
    )


def main() -> None:
    ensure_repo()
    ensure_data()
    run_battle()


if __name__ == "__main__":
    main()
