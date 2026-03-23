#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
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

DEFAULT_CHAMPION_PATH = REPO_DIR / "experiments" / "champions" / "frontier_public_clean_current_reference.json"
CONTEST_MAX_ARTIFACT_BYTES = 16_000_000


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0") not in {"0", "false", "False", ""}


def champion_path() -> Path:
    raw = os.environ.get("CHAMPION_PATH")
    if not raw:
        return DEFAULT_CHAMPION_PATH
    path = Path(raw)
    return path if path.is_absolute() else (REPO_DIR / path)


def lane_name() -> str:
    return os.environ.get("LANE_NAME", "smoke")


def champion_metric(champion: dict[str, Any], base_key: str) -> float:
    metrics = champion["arm"]["metrics"]
    if base_key in metrics:
        return float(metrics[base_key])
    mean_key = f"mean_{base_key}"
    if mean_key in metrics:
        return float(metrics[mean_key])
    raise KeyError(f"champion metric not found: {base_key}")


def champion_metric_or_none(champion: dict[str, Any], base_key: str) -> float | None:
    metrics = champion.get("arm", {}).get("metrics", {})
    raw = metrics.get(base_key, metrics.get(f"mean_{base_key}"))
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def metric_base_key() -> str:
    return os.environ.get("METRIC_BASE_KEY", "sliding_window_val_bpb")


def metric_loss_key() -> str:
    base_key = metric_base_key()
    if base_key == "sliding_window_val_bpb":
        return "sliding_window_val_loss"
    if base_key == "ttt_lora_val_bpb":
        return "ttt_lora_val_loss"
    return os.environ.get("METRIC_LOSS_KEY", "post_quant_val_loss")


def optional_path_env(name: str) -> str:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ""
    path = Path(raw)
    return str(path if path.is_absolute() else (REPO_DIR / path))


def legal_run(rep: dict[str, Any], score_key: str) -> bool:
    if rep.get("status") != "passed":
        return False
    metrics = rep.get("metrics", {})
    if score_key not in metrics:
        return False
    total_bytes = metrics.get("artifact_bytes_total")
    if total_bytes is None:
        return False
    try:
        return float(total_bytes) <= CONTEST_MAX_ARTIFACT_BYTES
    except (TypeError, ValueError):
        return False


def quarantine_reason(rep: dict[str, Any]) -> str | None:
    metrics = rep.get("metrics", {})
    reasons: list[str] = []
    artifact_floor_bytes = env_int("ARTIFACT_FLOOR_BYTES", 0)
    quant_gap_quarantine_bpb = env_float("QUANT_GAP_QUARANTINE_BPB", math.inf)
    low_artifact_probe_ok = env_bool("ALLOW_LOW_ARTIFACT_PROBE", False)
    if artifact_floor_bytes > 0 and not low_artifact_probe_ok:
        total_bytes = metrics.get("artifact_bytes_total")
        if total_bytes is None or float(total_bytes) < artifact_floor_bytes:
            reasons.append(
                f"artifact_below_floor:{int(float(total_bytes)) if total_bytes is not None else 'missing'}<{artifact_floor_bytes}"
            )
    if math.isfinite(quant_gap_quarantine_bpb):
        quant_gap = metrics.get("post_quant_gap_bpb")
        if quant_gap is None:
            reasons.append("quant_gap_missing")
        elif float(quant_gap) > quant_gap_quarantine_bpb:
            reasons.append(f"quant_gap_exceeds:{float(quant_gap):.6f}>{quant_gap_quarantine_bpb:.6f}")
    return ",".join(reasons) if reasons else None


def consistent_param(replicates: list[dict[str, Any]], key: str) -> Any:
    values = {rep.get("params", {}).get(key) for rep in replicates if key in rep.get("params", {})}
    if len(values) == 1:
        return next(iter(values))
    return None


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
    parsed["quarantine_reason"] = quarantine_reason(parsed) if parsed.get("status") == "passed" else None
    parsed["run_name"] = env["RUN_ID"]
    parsed["returncode"] = completed.returncode
    parsed["log_path"] = str(log_path.relative_to(REPO_DIR))
    return parsed


def build_summary(battle_id: str, replicates: list[dict[str, Any]]) -> dict[str, Any]:
    champion = json.loads(champion_path().read_text(encoding="utf-8"))
    champion_arm = copy.deepcopy(champion["arm"])
    champion_arm["arm_id"] = "CONTROL"
    score_key = metric_base_key()
    loss_key = metric_loss_key()
    successes = [rep for rep in replicates if rep["status"] == "passed"]
    legal_successes = [rep for rep in successes if legal_run(rep, score_key)]
    sane_legal_successes = [rep for rep in legal_successes if not rep.get("quarantine_reason")]
    quarantined_successes = [rep for rep in legal_successes if rep.get("quarantine_reason")]
    treatment_slug = treatment_name()
    treatment_script_target = treatment_target()
    train_path_id = os.environ.get("TRAIN_PATH_ID", treatment_slug)
    export_path_id = os.environ.get("EXPORT_PATH_ID", treatment_slug)
    eval_profile_id = os.environ.get("EVAL_PROFILE_ID", score_key)
    rules_profile = os.environ.get("RULES_PROFILE", champion.get("rules_status", "unknown"))
    ttt_compliance_mode = os.environ.get("TTT_COMPLIANCE_MODE", "none")

    mean_primary_metric = mean_or_none(
        [float(rep["metrics"][score_key]) for rep in sane_legal_successes if score_key in rep["metrics"]]
    )
    mean_primary_loss = mean_or_none(
        [float(rep["metrics"][loss_key]) for rep in sane_legal_successes if loss_key in rep["metrics"]]
    )
    mean_total_bytes = mean_or_none(
        [float(rep["metrics"]["artifact_bytes_total"]) for rep in sane_legal_successes if "artifact_bytes_total" in rep["metrics"]]
    )
    mean_quant_gap = mean_or_none(
        [float(rep["metrics"]["post_quant_gap_bpb"]) for rep in sane_legal_successes if "post_quant_gap_bpb" in rep["metrics"]]
    )
    mean_step_avg_ms = mean_or_none(
        [float(rep["metrics"]["step_avg_ms"]) for rep in sane_legal_successes if "step_avg_ms" in rep["metrics"]]
    )

    control_bpb = champion_metric(champion, score_key)
    control_loss = champion_metric_or_none(champion, loss_key)
    control_quant_gap = champion_metric_or_none(champion, "post_quant_gap_bpb")
    control_total_bytes = champion_metric_or_none(champion, "artifact_bytes_total")
    control_step_avg_ms = champion_metric_or_none(champion, "step_avg_ms")
    treatment_success_rate = len(successes) / len(replicates) if replicates else 0.0
    legal_success_rate = len(legal_successes) / len(replicates) if replicates else 0.0
    sane_legal_success_rate = len(sane_legal_successes) / len(replicates) if replicates else 0.0
    delta_bpb = mean_primary_metric - control_bpb if mean_primary_metric is not None else None

    status = "treatment_failed"
    if mean_primary_metric is not None and sane_legal_successes:
        status = "treatment_improved" if mean_primary_metric < control_bpb else "treatment_regressed"
    elif legal_successes and quarantined_successes and not sane_legal_successes:
        status = "treatment_quarantined"
    elif successes and not legal_successes:
        status = "treatment_illegal"

    return {
        "comparison_id": f"pg_arena_{lane_name()}_battle_{battle_id}_{treatment_slug}",
        "title": f"Parameter Golf ARENA {lane_name()} battle {battle_id}: champion control vs {treatment_slug} treatment",
        "date": date.today().isoformat(),
        "arena_tier": "tier1_battle",
        "lane_name": lane_name(),
        "comparison_mode": "lexicographic_contest",
        "promotion_decision_mode": "manual_review_required",
        "primary_metric_key": score_key,
        "primary_loss_key": loss_key,
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
        "control_source": {
            "champion_path": str(champion_path().relative_to(REPO_DIR)),
            "champion_id": champion.get("comparison_id", ""),
            "summary_run_id": champion.get("summary_run_id", ""),
            "arm_run_id": champion.get("arm_run_id", ""),
            "comparison_json": champion.get("comparison_json", ""),
            "control_class": champion.get("control_class", champion.get("status", "unknown")),
            "evidence_level": champion.get("evidence_level", "unknown"),
            "rules_status": champion.get("rules_status", "unknown"),
            "compute_scope": champion.get("compute_scope", "unknown"),
            "promotion_eligible": champion.get("promotion_eligible", False),
            "primary_metric_key": score_key,
            "primary_loss_key": loss_key,
        },
        "reference_roles": {
            "public_reference_path": optional_path_env("PUBLIC_REFERENCE_PATH")
            or str(champion_path().resolve()),
            "local_executable_control_path": optional_path_env("LOCAL_EXECUTABLE_CONTROL_PATH"),
            "proxy_control_path": optional_path_env("PROXY_CONTROL_PATH"),
            "historical_scaffold_path": optional_path_env("HISTORICAL_SCAFFOLD_PATH"),
            "provisional_watch_path": optional_path_env("PROVISIONAL_WATCH_PATH"),
        },
        "execution_profile": {
            "train_path_id": train_path_id,
            "export_path_id": export_path_id,
            "eval_profile_id": eval_profile_id,
            "rules_profile": rules_profile,
            "ttt_compliance_mode": ttt_compliance_mode,
        },
        "provenance": {
            "pod_id": os.environ.get("POD_ID", ""),
            "gpu_type": os.environ.get("GPU_TYPE", ""),
            "template_id": os.environ.get("TEMPLATE_ID", "y5cejece4j"),
            "image_name": os.environ.get("IMAGE_NAME", "runpod/parameter-golf:latest"),
            "battle_id": battle_id,
            "flash_attn_available": consistent_param(replicates, "flash_attn_available"),
            "world_size": consistent_param(replicates, "world_size"),
            "grad_accum_steps": consistent_param(replicates, "grad_accum_steps"),
            "sdp_backend_cudnn": consistent_param(replicates, "sdp_backend_cudnn"),
            "sdp_backend_flash": consistent_param(replicates, "sdp_backend_flash"),
            "sdp_backend_mem_efficient": consistent_param(replicates, "sdp_backend_mem_efficient"),
            "sdp_backend_math": consistent_param(replicates, "sdp_backend_math"),
            "guide_source": "/tmp/plexor-main-arena-doc/docs/guides/ARENA_OPS_AND_DEVELOPMENT_GUIDE.md",
            "backend_fingerprint": {
                "gpu_type": os.environ.get("GPU_TYPE", ""),
                "image_name": os.environ.get("IMAGE_NAME", "runpod/parameter-golf:latest"),
                "pytorch_version": os.environ.get("PYTORCH_VERSION", ""),
                "cuda_version": os.environ.get("CUDA_VERSION", ""),
                "compile_enabled": consistent_param(replicates, "compile_enabled"),
                "flash_attn_available": consistent_param(replicates, "flash_attn_available"),
                "sdp_backend_cudnn": consistent_param(replicates, "sdp_backend_cudnn"),
                "sdp_backend_flash": consistent_param(replicates, "sdp_backend_flash"),
                "sdp_backend_mem_efficient": consistent_param(replicates, "sdp_backend_mem_efficient"),
                "sdp_backend_math": consistent_param(replicates, "sdp_backend_math"),
            },
        },
        "fixture": {
            "dataset_variant": os.environ.get("DATASET_VARIANT", "fineweb10B_sp1024"),
            "tokenizer_path": os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"),
            "train_shards": env_int("TRAIN_SHARDS", 1),
            "train_seq_len": env_int("FIXTURE_TRAIN_SEQ_LEN", env_int("TRAIN_SEQ_LEN", 4096)),
            "eval_seq_len": env_int("FIXTURE_EVAL_SEQ_LEN", env_int("EVAL_SEQ_LEN", 0)),
            "eval_stride": env_int("EVAL_STRIDE", 64),
            "iterations": env_int("ITERATIONS", 20),
            "warmup_steps": env_int("WARMUP_STEPS", 0),
            "max_wallclock_seconds": env_int("MAX_WALLCLOCK_SECONDS", 0),
            "replicates_per_arm": len(replicates),
        },
        "summary_metrics": {
            "primary_control_val_bpb": control_bpb,
            "primary_control_val_loss": control_loss,
            "primary_treatment_mean_val_bpb": mean_primary_metric,
            "primary_treatment_mean_val_loss": mean_primary_loss,
            "primary_treatment_delta_bpb": delta_bpb,
            "control_metric_key": score_key,
            "control_metric_loss_key": loss_key,
            "control_primary_metric_value": control_bpb,
            "control_primary_loss_value": control_loss,
            "control_post_quant_gap_bpb": control_quant_gap,
            "control_artifact_bytes_total": control_total_bytes,
            "control_step_avg_ms": control_step_avg_ms,
            "treatment_success_rate": treatment_success_rate,
            "treatment_legal_success_rate": legal_success_rate,
            "treatment_sane_legal_success_rate": sane_legal_success_rate,
            "treatment_quarantined_runs": len(quarantined_successes),
            "treatment_primary_metric_value": mean_primary_metric,
            "treatment_primary_loss_value": mean_primary_loss,
            "treatment_mean_post_quant_gap_bpb": mean_quant_gap,
            "treatment_mean_total_submission_bytes": mean_total_bytes,
            "treatment_mean_step_avg_ms": mean_step_avg_ms,
            "treatment_vs_control_delta_bpb": delta_bpb,
            "treatment_failures": len(replicates) - len(successes),
            "artifact_floor_bytes": env_int("ARTIFACT_FLOOR_BYTES", 0),
            "quant_gap_quarantine_bpb": env_float("QUANT_GAP_QUARANTINE_BPB", math.inf),
            "train_path_id": train_path_id,
            "export_path_id": export_path_id,
            "eval_profile_id": eval_profile_id,
            "rules_profile": rules_profile,
            "ttt_compliance_mode": ttt_compliance_mode,
        },
        "arms": [
            champion_arm,
            {
                "arm_id": "TREATMENT",
                "run_name": f"{treatment_slug}_battle_{battle_id}",
                "status": status,
                "metrics": {
                    "mean_primary_metric_value": mean_primary_metric,
                    "mean_primary_loss_value": mean_primary_loss,
                    "mean_post_quant_gap_bpb": mean_quant_gap,
                    "mean_total_submission_bytes": mean_total_bytes,
                    "mean_step_avg_ms": mean_step_avg_ms,
                    "success_rate": treatment_success_rate,
                    "legal_success_rate": legal_success_rate,
                    "sane_legal_success_rate": sane_legal_success_rate,
                    "quarantined_runs": len(quarantined_successes),
                    "eval_batch_seqs": int(os.environ.get("EVAL_BATCH_SEQS", "64")),
                    "metric_base_key": score_key,
                    "metric_loss_key": loss_key,
                    "artifact_floor_bytes": env_int("ARTIFACT_FLOOR_BYTES", 0),
                    "quant_gap_quarantine_bpb": env_float("QUANT_GAP_QUARANTINE_BPB", math.inf),
                    "train_path_id": train_path_id,
                    "export_path_id": export_path_id,
                    "eval_profile_id": eval_profile_id,
                    "rules_profile": rules_profile,
                    "ttt_compliance_mode": ttt_compliance_mode,
                },
                "params": {
                    "train_batch_tokens": env_int("TRAIN_BATCH_TOKENS", 393216),
                    "train_seq_len": env_int("TRAIN_SEQ_LEN", 4096),
                    "eval_seq_len": env_int("EVAL_SEQ_LEN", 0),
                    "iterations": env_int("ITERATIONS", 20),
                    "matrix_lr": env_float("MATRIX_LR", 0.02),
                    "scalar_lr": env_float("SCALAR_LR", 0.02),
                    "tied_embed_lr": env_float("TIED_EMBED_LR", 0.03),
                    "muon_momentum": env_float("MUON_MOMENTUM", 0.99),
                    "eval_mode": "sliding_window" if os.environ.get("EVAL_STRIDE", "64") != "0" else "fixed_window",
                    "eval_batch_seqs": env_int("EVAL_BATCH_SEQS", 64),
                    "treatment_name": treatment_slug,
                },
                "replicates": replicates,
            },
        ],
        "next_action": {
            "recommendation": "promote only if the run is legal, survives rung-1 quarantine, sliding-window BPB improves, quant gap does not regress materially, and bytes/step time stay acceptable",
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
