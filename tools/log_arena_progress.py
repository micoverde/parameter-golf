#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log a Parameter Golf ARENA-style comparison into MLflow.")
    parser.add_argument("comparison_json", help="Path to the comparison JSON artifact.")
    parser.add_argument(
        "--tracking-uri",
        default="https://ca-mlflow-alpha.calmcliff-5a10adea.westus2.azurecontainerapps.io",
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment",
        default="parameter-golf-arena-smoke-2026-03-19",
        help="MLflow experiment name.",
    )
    return parser.parse_args()


def flatten(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(full_key, value))
        else:
            out[full_key] = value
    return out


def log_arm(mlflow: Any, arm: dict[str, Any], common_tags: dict[str, str], common_params: dict[str, Any]) -> None:
    run_name = arm["run_name"]
    with mlflow.start_run(run_name=run_name, nested=True):
        tags = dict(common_tags)
        tags.update(
            {
                "arena.arm_id": arm["arm_id"],
                "arena.arm_status": arm["status"],
            }
        )
        if arm.get("failure_kind"):
            tags["arena.failure_kind"] = str(arm["failure_kind"])
        if arm.get("failure_stage"):
            tags["arena.failure_stage"] = str(arm["failure_stage"])
        mlflow.set_tags(tags)

        arm_params = dict(common_params)
        arm_params.update(arm.get("params", {}))
        mlflow.log_params({k: str(v) for k, v in arm_params.items()})

        metrics = {
            key: float(value)
            for key, value in arm.get("metrics", {}).items()
            if isinstance(value, (int, float))
        }
        if metrics:
            mlflow.log_metrics(metrics)

        mlflow.log_dict(arm, f"{arm['arm_id'].lower()}_arm.json")

        for index, replicate in enumerate(arm.get("replicates", [])):
            replicate_run_name = replicate.get("run_name", f"{run_name}_replicate_{index}")
            with mlflow.start_run(run_name=replicate_run_name, nested=True):
                replicate_tags = dict(tags)
                replicate_tags.update(
                    {
                        "arena.replicate": "true",
                        "arena.replicate_index": str(index),
                        "arena.replicate_status": str(replicate.get("status", "unknown")),
                    }
                )
                if replicate.get("params", {}).get("seed") is not None:
                    replicate_tags["arena.seed"] = str(replicate["params"]["seed"])
                if replicate.get("failure_kind"):
                    replicate_tags["arena.failure_kind"] = str(replicate["failure_kind"])
                if replicate.get("failure_stage"):
                    replicate_tags["arena.failure_stage"] = str(replicate["failure_stage"])
                mlflow.set_tags(replicate_tags)

                replicate_params = dict(arm_params)
                replicate_params.update(replicate.get("params", {}))
                mlflow.log_params({k: str(v) for k, v in replicate_params.items()})

                replicate_metrics = {
                    key: float(value)
                    for key, value in replicate.get("metrics", {}).items()
                    if isinstance(value, (int, float))
                }
                if replicate_metrics:
                    mlflow.log_metrics(replicate_metrics)

                mlflow.log_dict(
                    replicate,
                    f"{arm['arm_id'].lower()}_replicate_{index}.json",
                )


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.comparison_json).read_text(encoding="utf-8"))

    import mlflow  # type: ignore

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    common_tags = {
        "arena.tier": str(payload["arena_tier"]),
        "arena.control_arm": str(payload["control_arm"]),
        "arena.treatment_arm": str(payload["treatment_arm"]),
        "arena.status": str(payload["status"]),
        "repo.name": str(payload["repo"]["name"]),
        "repo.branch": str(payload["repo"]["branch"]),
        "repo.commit": str(payload["repo"]["commit"]),
        "runpod.pod_id": str(payload["provenance"]["pod_id"]),
        "runpod.gpu_type": str(payload["provenance"]["gpu_type"]),
    }
    common_params = flatten("", payload["fixture"])

    with mlflow.start_run(run_name=payload["comparison_id"]):
        mlflow.set_tags(
            {
                **common_tags,
                "mlflow.runName": payload["comparison_id"],
                "arena.control_fixed": "true",
                "arena.next_action": str(payload["next_action"]["recommendation"]),
            }
        )
        mlflow.log_params({k: str(v) for k, v in common_params.items()})
        mlflow.log_params(
            {
                "repo.fork_url": str(payload["repo"]["fork_url"]),
                "repo.control_script": str(payload["repo"]["control_script"]),
                "repo.treatment_script": str(payload["repo"]["treatment_script"]),
                "runpod.template_id": str(payload["provenance"]["template_id"]),
                "runpod.image_name": str(payload["provenance"]["image_name"]),
            }
        )
        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in payload.get("summary_metrics", {}).items()
                if isinstance(value, (int, float))
            }
        )
        mlflow.log_dict(payload, "arena_comparison.json")

        for arm in payload["arms"]:
            log_arm(mlflow, arm, common_tags, common_params)

    print(f"logged comparison to experiment '{args.experiment}' at {args.tracking_uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
