#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge split Parameter Golf ARENA battle JSON files.")
    parser.add_argument("output_json", help="Path for merged output JSON.")
    parser.add_argument("input_jsons", nargs="+", help="Input battle JSON files to merge.")
    parser.add_argument(
        "--comparison-id",
        default=None,
        help="Optional override for merged comparison_id.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional override for merged title.",
    )
    return parser.parse_args()


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def load_payload(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def treatment_arm(payload: dict[str, Any]) -> dict[str, Any]:
    for arm in payload["arms"]:
        if arm["arm_id"] == "TREATMENT":
            return arm
    raise ValueError("missing TREATMENT arm")


def main() -> int:
    args = parse_args()
    payloads = [load_payload(path) for path in args.input_jsons]
    if not payloads:
        raise SystemExit("no payloads provided")

    base = payloads[0]
    merged = json.loads(json.dumps(base))

    combined_replicates: list[dict[str, Any]] = []
    for payload in payloads:
        combined_replicates.extend(treatment_arm(payload).get("replicates", []))

    successes = [rep for rep in combined_replicates if rep.get("status") == "passed"]
    mean_post_quant_bpb = mean_or_none(
        [
            float(rep["metrics"]["post_quant_val_bpb"])
            for rep in successes
            if "post_quant_val_bpb" in rep.get("metrics", {})
        ]
    )
    mean_post_quant_loss = mean_or_none(
        [
            float(rep["metrics"]["post_quant_val_loss"])
            for rep in successes
            if "post_quant_val_loss" in rep.get("metrics", {})
        ]
    )
    mean_total_bytes = mean_or_none(
        [
            float(rep["metrics"]["artifact_bytes_total"])
            for rep in successes
            if "artifact_bytes_total" in rep.get("metrics", {})
        ]
    )

    control_bpb = float(merged["summary_metrics"]["control_post_quant_val_bpb"])
    success_rate = len(successes) / len(combined_replicates) if combined_replicates else 0.0
    delta_bpb = mean_post_quant_bpb - control_bpb if mean_post_quant_bpb is not None else None

    status = "treatment_failed"
    if mean_post_quant_bpb is not None:
        status = "treatment_improved" if mean_post_quant_bpb < control_bpb else "treatment_regressed"

    merged["comparison_id"] = args.comparison_id or merged["comparison_id"]
    merged["title"] = args.title or merged["title"]
    merged["status"] = status
    merged["summary_metrics"]["treatment_success_rate"] = success_rate
    merged["summary_metrics"]["treatment_mean_post_quant_val_bpb"] = mean_post_quant_bpb
    merged["summary_metrics"]["treatment_mean_post_quant_val_loss"] = mean_post_quant_loss
    merged["summary_metrics"]["treatment_mean_total_submission_bytes"] = mean_total_bytes
    merged["summary_metrics"]["treatment_vs_control_delta_bpb"] = delta_bpb
    merged["summary_metrics"]["treatment_failures"] = len(combined_replicates) - len(successes)
    merged["fixture"]["replicates_per_arm"] = len(combined_replicates)

    arm = treatment_arm(merged)
    arm["status"] = status
    arm["metrics"]["mean_post_quant_val_bpb"] = mean_post_quant_bpb
    arm["metrics"]["mean_post_quant_val_loss"] = mean_post_quant_loss
    arm["metrics"]["mean_total_submission_bytes"] = mean_total_bytes
    arm["metrics"]["success_rate"] = success_rate
    arm["replicates"] = combined_replicates

    Path(args.output_json).write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
