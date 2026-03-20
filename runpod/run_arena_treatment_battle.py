#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from tools.parse_train_log import parse_train_log

CHAMPION_PATH = REPO_DIR / "experiments" / "champions" / "current_champion.json"


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def champion_metric(champion: dict[str, Any], base_key: str) -> float:
    metrics = champion["arm"]["metrics"]
    if base_key in metrics:
        return float(metrics[base_key])
    mean_key = f"mean_{base_key}"
    if mean_key in metrics:
        return float(metrics[mean_key])
    raise KeyError(f"champion metric not found: {base_key}")


def treatment_script() -> str:
    return os.environ.get("TREATMENT_SCRIPT", "runpod/smoke_seq4096_sliding_eval.sh")


def treatment_name() -> str:
    return os.environ.get("TREATMENT_NAME", "seq4096_sliding_eval")


def treatment_target() -> str:
    return os.environ.get("TREATMENT_TARGET", "experiments/seq4096_sliding_eval/train_gpt.py")


def run_one(seed: int, battle_id: str, log_dir: Path) -> dict[str, Any]:
    log_path = log_dir / f"treatment_seed{seed}.log"
    env = os.environ.copy()
    env.update(
        {
            "SEED": str(seed),
            "RUN_ID": f"arena_{battle_id}_{treatment_name()}_seed{seed}",
            "ITERATIONS": os.environ.get("ITERATIONS", "20"),
            "VAL_LOSS_EVERY": os.environ.get("VAL_LOSS_EVERY", "0"),
            "WARMUP_STEPS": os.environ.get("WARMUP_STEPS", "0"),
            "MAX_WALLCLOCK_SECONDS": os.environ.get("MAX_WALLCLOCK_SECONDS", "0"),
            "EVAL_BATCH_SEQS": os.environ.get("EVAL_BATCH_SEQS", "64"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            ),
        }
    )
    with log_path.open("w", encoding="utf-8") as fh:
        completed = subprocess.run(
            ["bash", treatment_script()],
            cwd=REPO_DIR,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    parsed = parse_train_log(log_path)
    parsed["run_name"] = env["RUN_ID"]
    parsed["returncode"] = completed.returncode
    parsed["log_path"] = str(log_path.relative_to(REPO_DIR))
    return parsed


def build_summary(battle_id: str, replicates: list[dict[str, Any]]) -> dict[str, Any]:
    champion = json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))
    successes = [rep for rep in replicates if rep["status"] == "passed"]
    treatment_slug = treatment_name()
    treatment_script_target = treatment_target()

    mean_post_quant_bpb = mean_or_none(
        [float(rep["metrics"]["post_quant_val_bpb"]) for rep in successes if "post_quant_val_bpb" in rep["metrics"]]
    )
    mean_post_quant_loss = mean_or_none(
        [float(rep["metrics"]["post_quant_val_loss"]) for rep in successes if "post_quant_val_loss" in rep["metrics"]]
    )
    mean_total_bytes = mean_or_none(
        [float(rep["metrics"]["artifact_bytes_total"]) for rep in successes if "artifact_bytes_total" in rep["metrics"]]
    )

    control_bpb = champion_metric(champion, "post_quant_val_bpb")
    treatment_success_rate = len(successes) / len(replicates) if replicates else 0.0
    delta_bpb = mean_post_quant_bpb - control_bpb if mean_post_quant_bpb is not None else None

    status = "treatment_failed"
    if mean_post_quant_bpb is not None:
        status = "treatment_improved" if mean_post_quant_bpb < control_bpb else "treatment_regressed"

    return {
        "comparison_id": f"pg_arena_battle_{battle_id}_{treatment_slug}",
        "title": f"Parameter Golf ARENA battle {battle_id}: champion control vs {treatment_slug} treatment",
        "date": date.today().isoformat(),
        "arena_tier": "tier1_battle",
        "control_arm": "CONTROL",
        "treatment_arm": "TREATMENT",
        "status": status,
        "tracking_uri": "https://ca-mlflow-alpha.calmcliff-5a10adea.westus2.azurecontainerapps.io",
        "repo": {
            "name": "parameter-golf",
            "fork_url": "https://github.com/micoverde/parameter-golf.git",
            "branch": os.environ.get("BRANCH", "feature/arena-battle-loop"),
            "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR, text=True).strip(),
            "control_script": champion["arm"].get("run_name", "seq4096_control_smoke"),
            "treatment_script": treatment_script_target,
        },
        "provenance": {
            "pod_id": os.environ.get("POD_ID", ""),
            "gpu_type": os.environ.get("GPU_TYPE", ""),
            "template_id": os.environ.get("TEMPLATE_ID", "y5cejece4j"),
            "image_name": os.environ.get("IMAGE_NAME", "runpod/parameter-golf:latest"),
            "battle_id": battle_id,
            "guide_source": "/tmp/plexor-main-arena-doc/docs/guides/ARENA_OPS_AND_DEVELOPMENT_GUIDE.md",
        },
        "fixture": {
            "dataset_variant": "fineweb10B_sp1024",
            "tokenizer_path": "./data/tokenizers/fineweb_1024_bpe.model",
            "train_shards": int(os.environ.get("TRAIN_SHARDS", "1")),
            "train_seq_len": 4096,
            "iterations": int(os.environ.get("ITERATIONS", "20")),
            "warmup_steps": int(os.environ.get("WARMUP_STEPS", "0")),
            "max_wallclock_seconds": int(os.environ.get("MAX_WALLCLOCK_SECONDS", "0")),
            "replicates_per_arm": len(replicates),
        },
        "summary_metrics": {
            "control_post_quant_val_bpb": control_bpb,
            "control_post_quant_val_loss": champion_metric(champion, "post_quant_val_loss"),
            "treatment_success_rate": treatment_success_rate,
            "treatment_mean_post_quant_val_bpb": mean_post_quant_bpb,
            "treatment_mean_post_quant_val_loss": mean_post_quant_loss,
            "treatment_mean_total_submission_bytes": mean_total_bytes,
            "treatment_vs_control_delta_bpb": delta_bpb,
            "treatment_failures": len(replicates) - len(successes),
        },
        "arms": [
            champion["arm"],
            {
                "arm_id": "TREATMENT",
                "run_name": f"{treatment_slug}_battle_{battle_id}",
                "status": status,
                "metrics": {
                    "mean_post_quant_val_bpb": mean_post_quant_bpb,
                    "mean_post_quant_val_loss": mean_post_quant_loss,
                    "mean_total_submission_bytes": mean_total_bytes,
                    "success_rate": treatment_success_rate,
                    "eval_batch_seqs": int(os.environ.get("EVAL_BATCH_SEQS", "64")),
                },
                "params": {
                    "train_batch_tokens": 393216,
                    "train_seq_len": 4096,
                    "iterations": int(os.environ.get("ITERATIONS", "20")),
                    "matrix_lr": 0.02,
                    "scalar_lr": 0.02,
                    "tied_embed_lr": 0.03,
                    "muon_momentum": 0.99,
                    "eval_mode": "sliding_window" if os.environ.get("EVAL_STRIDE", "64") != "0" else "fixed_window",
                    "eval_batch_seqs": int(os.environ.get("EVAL_BATCH_SEQS", "64")),
                    "treatment_name": treatment_slug,
                },
                "replicates": replicates,
            },
        ],
        "next_action": {
            "recommendation": "promote treatment only if mean_post_quant_val_bpb beats control and success rate is 1.0",
            "candidate_eval_batch_seqs": [32, 64, 128],
        },
    }


def main() -> int:
    battle_id = os.environ.get("BATTLE_ID", "b03")
    seed_values = os.environ.get(
        "SEEDS", "1337,1338,1339,1340,1341,1342,1343,1344,1345,1346"
    )
    seeds = [int(token.strip()) for token in seed_values.split(",") if token.strip()]
    log_dir = REPO_DIR / "battle_results" / battle_id
    log_dir.mkdir(parents=True, exist_ok=True)

    replicates = [run_one(seed, battle_id, log_dir) for seed in seeds]
    summary = build_summary(battle_id, replicates)
    out_path = REPO_DIR / "experiments" / "arena_runs" / f"{date.today().isoformat()}_{battle_id}_{treatment_name()}_battle.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_path.relative_to(REPO_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
