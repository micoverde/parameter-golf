#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_URL = os.environ.get("REPO_URL", "https://github.com/micoverde/parameter-golf.git")
BRANCH = os.environ.get("BRANCH", "feature/contest-preflight-ci")
TRAIN_SHARDS = os.environ.get("TRAIN_SHARDS", "1")
RUN_SMOKE = os.environ.get("RUN_SMOKE", "1") == "1"
ARM = os.environ.get("ARM", "both")
WORKSPACE = Path("/workspace")
REPO_DIR = WORKSPACE / "parameter-golf"


def run(cmd: list[str], cwd: Path | None = None, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    printable = " ".join(cmd)
    prefix = f"[cwd={cwd}] " if cwd else ""
    print(f"$ {prefix}{printable}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_repo() -> None:
    if not REPO_DIR.exists():
        run(["git", "clone", REPO_URL, str(REPO_DIR)])
    else:
        run(["git", "remote", "set-url", "origin", REPO_URL], cwd=REPO_DIR)

    run(["git", "fetch", "origin"], cwd=REPO_DIR)
    run(["git", "checkout", BRANCH], cwd=REPO_DIR)
    run(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=REPO_DIR)


def ensure_data() -> None:
    tokenizer = REPO_DIR / "data" / "tokenizers" / "fineweb_1024_bpe.model"
    if tokenizer.exists():
        print("tokenizer already present; refreshing requested train shard range", flush=True)
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


def run_smoke(script: str, run_id: str) -> None:
    smoke_env = {
        "RUN_ID": run_id,
        "ITERATIONS": os.environ.get("ITERATIONS", "20"),
        "VAL_LOSS_EVERY": os.environ.get("VAL_LOSS_EVERY", "0"),
        "WARMUP_STEPS": os.environ.get("WARMUP_STEPS", "0"),
        "MAX_WALLCLOCK_SECONDS": os.environ.get("MAX_WALLCLOCK_SECONDS", "0"),
        "NPROC_PER_NODE": os.environ.get("NPROC_PER_NODE", "1"),
    }
    run(["bash", script], cwd=REPO_DIR, extra_env=smoke_env)


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    ensure_repo()
    ensure_data()
    if RUN_SMOKE:
        if ARM in {"control", "both"}:
            run_smoke("runpod/smoke_seq4096_control.sh", os.environ.get("CONTROL_RUN_ID", "smoke_seq4096_control"))
        if ARM in {"treatment", "both"}:
            run_smoke(
                "runpod/smoke_seq4096_sliding_eval.sh",
                os.environ.get("TREATMENT_RUN_ID", "smoke_seq4096_sliding_eval"),
            )


if __name__ == "__main__":
    main()
