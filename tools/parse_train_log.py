#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


STEP_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+)\s+val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+train_time:(?P<train_time_ms>\d+)ms\s+step_avg:(?P<step_avg_ms>[0-9.]+)ms"
)
FINAL_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
FINAL_SLIDING_RE = re.compile(
    r"final_sliding_window_exact val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
FINAL_TTT_RE = re.compile(
    r"final_int8_ttt_lora val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
BYTES_TOTAL_RE = re.compile(r"Total submission size int8\+zlib: (?P<value>\d+) bytes")
BYTES_MODEL_RE = re.compile(r"Serialized model int8\+zlib: (?P<value>\d+) bytes")
BYTES_CODE_RE = re.compile(r"Code size: (?P<value>\d+) bytes")
EVAL_TIME_RE = re.compile(r"eval_time:(?P<value>\d+)ms")
SEED_RE = re.compile(r"seed:(?P<seed>\d+)")
OOM_RE = re.compile(r"(CUDA out of memory|OutOfMemoryError)")
EVAL_MODE_RE = re.compile(r"final_eval_mode:(?P<mode>[a-zA-Z_]+)\s+stride:(?P<stride>\d+)\s+batch_seqs:(?P<batch>\d+)")
PEAK_MEM_RE = re.compile(
    r"peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB"
)


def _to_number(value: str) -> float | int:
    if "." in value:
        return float(value)
    return int(value)


def parse_train_log(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    result: dict[str, Any] = {
        "status": "unknown",
        "metrics": {},
        "params": {},
        "failure_kind": None,
        "failure_stage": None,
        "failure_message": None,
    }

    seed_match = SEED_RE.search(text)
    if seed_match:
        result["params"]["seed"] = int(seed_match.group("seed"))

    step_matches = list(STEP_RE.finditer(text))
    if step_matches:
        last = step_matches[-1]
        result["metrics"].update(
            {
                "train_step_final": int(last.group("step")),
                "train_step_final_val_loss": float(last.group("val_loss")),
                "train_step_final_val_bpb": float(last.group("val_bpb")),
                "train_time_ms": int(last.group("train_time_ms")),
                "step_avg_ms": float(last.group("step_avg_ms")),
            }
        )
        result["params"]["iterations"] = int(last.group("iterations"))

    final_match = FINAL_RE.search(text)
    if final_match:
        result["metrics"]["post_quant_val_loss"] = float(final_match.group("val_loss"))
        result["metrics"]["post_quant_val_bpb"] = float(final_match.group("val_bpb"))
        result["status"] = "passed"

    sliding_match = FINAL_SLIDING_RE.search(text)
    if sliding_match:
        result["metrics"]["sliding_window_val_loss"] = float(sliding_match.group("val_loss"))
        result["metrics"]["sliding_window_val_bpb"] = float(sliding_match.group("val_bpb"))
        result["status"] = "passed"

    ttt_match = FINAL_TTT_RE.search(text)
    if ttt_match:
        result["metrics"]["ttt_lora_val_loss"] = float(ttt_match.group("val_loss"))
        result["metrics"]["ttt_lora_val_bpb"] = float(ttt_match.group("val_bpb"))
        result["status"] = "passed"

    total_bytes = BYTES_TOTAL_RE.search(text)
    if total_bytes:
        result["metrics"]["artifact_bytes_total"] = int(total_bytes.group("value"))
    model_bytes = BYTES_MODEL_RE.search(text)
    if model_bytes:
        result["metrics"]["artifact_bytes_model_int8_zlib"] = int(model_bytes.group("value"))
    code_bytes = BYTES_CODE_RE.search(text)
    if code_bytes:
        result["metrics"]["artifact_bytes_code"] = int(code_bytes.group("value"))
    eval_time = EVAL_TIME_RE.search(text)
    if eval_time:
        result["metrics"]["eval_time_ms"] = int(eval_time.group("value"))
    peak_mem = PEAK_MEM_RE.search(text)
    if peak_mem:
        result["metrics"]["peak_memory_allocated_mib"] = int(peak_mem.group("allocated"))
        result["metrics"]["peak_memory_reserved_mib"] = int(peak_mem.group("reserved"))
    eval_mode = EVAL_MODE_RE.search(text)
    if eval_mode:
        result["params"]["eval_mode"] = eval_mode.group("mode")
        result["metrics"]["eval_stride"] = int(eval_mode.group("stride"))
        result["metrics"]["eval_batch_seqs"] = int(eval_mode.group("batch"))

    if OOM_RE.search(text):
        result["status"] = "failed"
        result["failure_kind"] = "cuda_oom"
        result["failure_stage"] = "post_quant_sliding_eval" if "final_eval_mode" in text else "training_or_eval"
        result["failure_message"] = "CUDA OOM detected in log"
    elif result["status"] == "unknown" and step_matches:
        result["status"] = "incomplete"

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse a Parameter Golf training log into JSON.")
    parser.add_argument("log_path")
    args = parser.parse_args()
    print(json.dumps(parse_train_log(args.log_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
