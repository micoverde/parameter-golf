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
TRAIN_STEP_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+)\s+train_loss:(?P<train_loss>[0-9.]+)\s+train_time:(?P<train_time_ms>\d+)ms\s+step_avg:(?P<step_avg_ms>[0-9.]+)ms"
)
FINAL_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
FINAL_INT6_RE = re.compile(
    r"final_int6_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
FINAL_INT6_ZSTD_RE = re.compile(
    r"final_int6_zstd_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
FINAL_INT6_WITH_TIME_RE = re.compile(
    r"final_int6_roundtrip val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+eval_time:(?P<eval_time_ms>\d+)ms"
)
FINAL_INT6_ZSTD_WITH_TIME_RE = re.compile(
    r"final_int6_zstd_roundtrip val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+eval_time:(?P<eval_time_ms>\d+)ms"
)
FINAL_SLIDING_RE = re.compile(
    r"final_sliding_window_exact val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
FINAL_INT6_SLIDING_RE = re.compile(
    r"final_int6_sliding_window_exact val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
FINAL_INT6_SLIDING_WITH_TIME_RE = re.compile(
    r"final_int6_sliding_window val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+stride:(?P<stride>\d+)\s+eval_time:(?P<eval_time_ms>\d+)ms"
)
FINAL_INT6_SLIDING_S64_RE = re.compile(
    r"final_int6_sliding_window_s64_exact val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
FINAL_INT6_SLIDING_S64_WITH_TIME_RE = re.compile(
    r"final_int6_sliding_window_s64 val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+stride:64\s+eval_time:(?P<eval_time_ms>\d+)ms"
)
FINAL_TTT_RE = re.compile(
    r"final_int8_ttt_lora val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
POST_SWA_RE = re.compile(
    r"DIAGNOSTIC post_swa val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)"
)
POST_TIGHTSWA_RE = re.compile(
    r"DIAGNOSTIC post_tightswa val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+eval_time:(?P<eval_time_ms>\d+)ms"
)
POST_EMA_RE = re.compile(
    r"DIAGNOSTIC post_ema val_loss:(?P<val_loss>[0-9.]+)\s+val_bpb:(?P<val_bpb>[0-9.]+)\s+eval_time:(?P<eval_time_ms>\d+)ms"
)
BYTES_TOTAL_RE = re.compile(r"Total submission size int(?:5|6|8)\+(?:zlib|zstd): (?P<value>\d+) bytes")
BYTES_MODEL_RE = re.compile(r"Serialized model int(?:5|6|8)\+(?:zlib|zstd): (?P<value>\d+) bytes")
BYTES_CODE_RE = re.compile(r"Code size: (?P<value>\d+) bytes")
BYTES_TOTAL_RAW_RE = re.compile(r"Total submission size: (?P<value>\d+) bytes")
BYTES_MODEL_RAW_RE = re.compile(r"Serialized model: (?P<value>\d+) bytes")
EVAL_TIME_RE = re.compile(r"eval_time:(?P<value>\d+)ms")
SEED_RE = re.compile(r"seed:(?P<seed>\d+)")
OOM_RE = re.compile(r"(CUDA out of memory|OutOfMemoryError)")
EVAL_MODE_RE = re.compile(
    r"final_eval_mode:(?P<mode>[a-zA-Z_]+)(?:\s+stride:(?P<stride>\d+)\s+batch_seqs:(?P<batch>\d+))?"
)
PEAK_MEM_RE = re.compile(
    r"peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB"
)
FLASH_ATTN_RE = re.compile(r"flash_attn_available:(?P<value>[01])")
WORLD_SIZE_RE = re.compile(r"world_size:(?P<world_size>\d+)\s+grad_accum_steps:(?P<grad_accum_steps>\d+)")
SDP_BACKENDS_RE = re.compile(
    r"sdp_backends:cudnn=(?P<cudnn>\w+)\s+flash=(?P<flash>\w+)\s+mem_efficient=(?P<mem_efficient>\w+)\s+math=(?P<math>\w+)"
)
EMA_STATE_RE = re.compile(r"ema_state_tensors:(?P<count>\d+)\s+ema_decay:(?P<decay>[0-9.]+)")
EXPORT_STATE_RE = re.compile(
    r"export_state_tensors:(?P<export>\d+)\s+model_state_tensors:(?P<model>\d+)\s+ema_enabled:(?P<ema>[01])"
)
LATE_QAT_RE = re.compile(r"late_qat:enabled step:(?P<step>\d+)\s+scale:(?P<scale>[0-9.]+)")
MODEL_TOTAL_COMPACT_RE = re.compile(r"model:(?P<model>\d+)\s+code:(?P<code>\d+)\s+total:(?P<total>\d+)")
PRUNE_RE = re.compile(
    r"prune:zeroed\s+(?P<zeroed>\d+)/(?P<total>\d+)\s+int6 weights\s+\((?P<pct>[0-9.]+)%\)\s+threshold=(?P<threshold>-?[0-9.]+)"
)
STAGE_METRIC_RE = re.compile(
    r"STAGE_METRIC\s+name:(?P<name>[a-zA-Z0-9_]+)\s+val_loss:(?P<val_loss>-?[0-9.]+)\s+val_bpb:(?P<val_bpb>-?[0-9.]+)\s+eval_time:(?P<eval_time_ms>\d+)ms"
)
STAGE_STATE_RE = re.compile(
    r"STAGE_STATE\s+name:(?P<name>[a-zA-Z0-9_]+)\s+ref:(?P<ref>[a-zA-Z0-9_]+)\s+tensors:(?P<tensors>\d+)\s+norm_ratio:(?P<norm_ratio>[a-zA-Z0-9.eE+-]+)\s+cosine:(?P<cosine>[a-zA-Z0-9.eE+-]+)"
)
QUANT_STATS_RE = re.compile(
    r"QUANT_STATS\s+stage:(?P<stage>[a-zA-Z0-9_]+)\s+tensors:(?P<tensors>\d+)\s+int6_tensors:(?P<int6_tensors>\d+)\s+int8_tensors:(?P<int8_tensors>\d+)\s+scale_min:(?P<scale_min>[a-zA-Z0-9.eE+-]+)\s+scale_p50:(?P<scale_p50>[a-zA-Z0-9.eE+-]+)\s+scale_max:(?P<scale_max>[a-zA-Z0-9.eE+-]+)\s+clip_frac:(?P<clip_frac>[a-zA-Z0-9.eE+-]+)\s+near_zero_frac:(?P<near_zero_frac>[a-zA-Z0-9.eE+-]+)"
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

    flash_attn_match = FLASH_ATTN_RE.search(text)
    if flash_attn_match:
        result["params"]["flash_attn_available"] = bool(int(flash_attn_match.group("value")))

    world_size_match = WORLD_SIZE_RE.search(text)
    if world_size_match:
        result["params"]["world_size"] = int(world_size_match.group("world_size"))
        result["params"]["grad_accum_steps"] = int(world_size_match.group("grad_accum_steps"))

    sdp_backends_match = SDP_BACKENDS_RE.search(text)
    if sdp_backends_match:
        result["params"]["sdp_backend_cudnn"] = sdp_backends_match.group("cudnn") == "True"
        result["params"]["sdp_backend_flash"] = sdp_backends_match.group("flash") == "True"
        result["params"]["sdp_backend_mem_efficient"] = sdp_backends_match.group("mem_efficient") == "True"
        result["params"]["sdp_backend_math"] = sdp_backends_match.group("math") == "True"

    ema_state_match = EMA_STATE_RE.search(text)
    if ema_state_match:
        result["params"]["ema_state_tensors"] = int(ema_state_match.group("count"))
        result["params"]["ema_decay"] = float(ema_state_match.group("decay"))

    export_state_match = EXPORT_STATE_RE.search(text)
    if export_state_match:
        result["params"]["export_state_tensors"] = int(export_state_match.group("export"))
        result["params"]["model_state_tensors"] = int(export_state_match.group("model"))
        result["params"]["ema_enabled"] = bool(int(export_state_match.group("ema")))

    late_qat_match = LATE_QAT_RE.search(text)
    if late_qat_match:
        result["params"]["late_qat_enabled_step"] = int(late_qat_match.group("step"))
        result["params"]["late_qat_enabled_scale"] = float(late_qat_match.group("scale"))

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

    train_step_matches = list(TRAIN_STEP_RE.finditer(text))
    if train_step_matches:
        last_train = train_step_matches[-1]
        result["metrics"].update(
            {
                "train_step_latest": int(last_train.group("step")),
                "train_step_latest_loss": float(last_train.group("train_loss")),
                "train_time_ms": int(last_train.group("train_time_ms")),
                "step_avg_ms": float(last_train.group("step_avg_ms")),
            }
        )
        result["params"].setdefault("iterations", int(last_train.group("iterations")))

    final_match = FINAL_RE.search(text) or FINAL_INT6_RE.search(text) or FINAL_INT6_ZSTD_RE.search(text)
    if final_match:
        result["metrics"]["post_quant_val_loss"] = float(final_match.group("val_loss"))
        result["metrics"]["post_quant_val_bpb"] = float(final_match.group("val_bpb"))
        result["status"] = "passed"

    sliding_match = FINAL_SLIDING_RE.search(text) or FINAL_INT6_SLIDING_RE.search(text)
    if sliding_match:
        result["metrics"]["sliding_window_val_loss"] = float(sliding_match.group("val_loss"))
        result["metrics"]["sliding_window_val_bpb"] = float(sliding_match.group("val_bpb"))
        result["status"] = "passed"

    sliding_s64_match = FINAL_INT6_SLIDING_S64_RE.search(text)
    if sliding_s64_match:
        result["metrics"]["sliding_window_s64_val_loss"] = float(sliding_s64_match.group("val_loss"))
        result["metrics"]["sliding_window_s64_val_bpb"] = float(sliding_s64_match.group("val_bpb"))
        # Treat the canonical stride-64 score as the primary sliding metric when present.
        result["metrics"]["sliding_window_val_loss"] = float(sliding_s64_match.group("val_loss"))
        result["metrics"]["sliding_window_val_bpb"] = float(sliding_s64_match.group("val_bpb"))
        result["status"] = "passed"

    ttt_match = FINAL_TTT_RE.search(text)
    if ttt_match:
        result["metrics"]["ttt_lora_val_loss"] = float(ttt_match.group("val_loss"))
        result["metrics"]["ttt_lora_val_bpb"] = float(ttt_match.group("val_bpb"))
        result["status"] = "passed"

    post_swa_match = POST_SWA_RE.search(text)
    if post_swa_match:
        result["metrics"]["post_swa_val_loss"] = float(post_swa_match.group("val_loss"))
        result["metrics"]["post_swa_val_bpb"] = float(post_swa_match.group("val_bpb"))

    post_tightswa_match = POST_TIGHTSWA_RE.search(text)
    if post_tightswa_match:
        result["metrics"]["post_tightswa_val_loss"] = float(post_tightswa_match.group("val_loss"))
        result["metrics"]["post_tightswa_val_bpb"] = float(post_tightswa_match.group("val_bpb"))
        result["metrics"]["post_tightswa_eval_time_ms"] = int(post_tightswa_match.group("eval_time_ms"))

    post_ema_match = POST_EMA_RE.search(text)
    if post_ema_match:
        result["metrics"]["post_ema_val_loss"] = float(post_ema_match.group("val_loss"))
        result["metrics"]["post_ema_val_bpb"] = float(post_ema_match.group("val_bpb"))
        result["metrics"]["post_ema_eval_time_ms"] = int(post_ema_match.group("eval_time_ms"))

    post_quant_time_match = FINAL_INT6_WITH_TIME_RE.search(text) or FINAL_INT6_ZSTD_WITH_TIME_RE.search(text)
    if post_quant_time_match:
        result["metrics"]["post_quant_eval_time_ms"] = int(post_quant_time_match.group("eval_time_ms"))

    sliding_time_match = FINAL_INT6_SLIDING_WITH_TIME_RE.search(text)
    if sliding_time_match:
        result["metrics"]["sliding_window_eval_time_ms"] = int(sliding_time_match.group("eval_time_ms"))
        result["metrics"]["eval_stride"] = int(sliding_time_match.group("stride"))

    sliding_s64_time_match = FINAL_INT6_SLIDING_S64_WITH_TIME_RE.search(text)
    if sliding_s64_time_match:
        result["metrics"]["sliding_window_s64_eval_time_ms"] = int(sliding_s64_time_match.group("eval_time_ms"))
        result["metrics"]["sliding_window_eval_time_ms"] = int(sliding_s64_time_match.group("eval_time_ms"))
        result["metrics"]["eval_stride"] = 64

    total_bytes = BYTES_TOTAL_RE.search(text)
    if total_bytes:
        result["metrics"]["artifact_bytes_total"] = int(total_bytes.group("value"))
    total_bytes_raw = BYTES_TOTAL_RAW_RE.search(text)
    if total_bytes_raw:
        result["metrics"]["artifact_bytes_total_raw"] = int(total_bytes_raw.group("value"))
    model_total_compact = MODEL_TOTAL_COMPACT_RE.search(text)
    if model_total_compact:
        result["metrics"]["artifact_bytes_model_quantized"] = int(model_total_compact.group("model"))
        result["metrics"]["artifact_bytes_model_int8_zlib"] = int(model_total_compact.group("model"))
        result["metrics"]["artifact_bytes_code"] = int(model_total_compact.group("code"))
        result["metrics"]["artifact_bytes_total"] = int(model_total_compact.group("total"))
    model_bytes = BYTES_MODEL_RE.search(text)
    if model_bytes:
        result["metrics"]["artifact_bytes_model_quantized"] = int(model_bytes.group("value"))
        # Backward-compatible alias for older consumers.
        result["metrics"]["artifact_bytes_model_int8_zlib"] = int(model_bytes.group("value"))
    model_bytes_raw = BYTES_MODEL_RAW_RE.search(text)
    if model_bytes_raw:
        result["metrics"]["artifact_bytes_model_raw"] = int(model_bytes_raw.group("value"))
    code_bytes = BYTES_CODE_RE.search(text)
    if code_bytes:
        result["metrics"]["artifact_bytes_code"] = int(code_bytes.group("value"))
    eval_time = EVAL_TIME_RE.search(text)
    if eval_time and "eval_time_ms" not in result["metrics"]:
        result["metrics"]["eval_time_ms"] = int(eval_time.group("value"))
    peak_mem = PEAK_MEM_RE.search(text)
    if peak_mem:
        result["metrics"]["peak_memory_allocated_mib"] = int(peak_mem.group("allocated"))
        result["metrics"]["peak_memory_reserved_mib"] = int(peak_mem.group("reserved"))
    eval_mode = EVAL_MODE_RE.search(text)
    if eval_mode:
        result["params"]["eval_mode"] = eval_mode.group("mode")
        if eval_mode.group("stride") is not None:
            result["metrics"]["eval_stride"] = int(eval_mode.group("stride"))
        if eval_mode.group("batch") is not None:
            result["metrics"]["eval_batch_seqs"] = int(eval_mode.group("batch"))

    prune_match = PRUNE_RE.search(text)
    if prune_match:
        result["metrics"]["pruned_int6_weights"] = int(prune_match.group("zeroed"))
        result["metrics"]["pruned_int6_total_weights"] = int(prune_match.group("total"))
        result["metrics"]["pruned_int6_fraction"] = float(prune_match.group("pct")) / 100.0
        result["metrics"]["pruned_int6_threshold"] = float(prune_match.group("threshold"))

    for stage_match in STAGE_METRIC_RE.finditer(text):
        stage_name = stage_match.group("name")
        result["metrics"][f"stage_{stage_name}_val_loss"] = float(stage_match.group("val_loss"))
        result["metrics"][f"stage_{stage_name}_val_bpb"] = float(stage_match.group("val_bpb"))
        result["metrics"][f"stage_{stage_name}_eval_time_ms"] = int(stage_match.group("eval_time_ms"))

    for state_match in STAGE_STATE_RE.finditer(text):
        stage_name = state_match.group("name")
        result["metrics"][f"stage_{stage_name}_ref"] = state_match.group("ref")
        result["metrics"][f"stage_{stage_name}_tensors"] = int(state_match.group("tensors"))
        result["metrics"][f"stage_{stage_name}_norm_ratio"] = float(state_match.group("norm_ratio"))
        result["metrics"][f"stage_{stage_name}_cosine"] = float(state_match.group("cosine"))

    for quant_match in QUANT_STATS_RE.finditer(text):
        stage_name = quant_match.group("stage")
        result["metrics"][f"quant_{stage_name}_tensors"] = int(quant_match.group("tensors"))
        result["metrics"][f"quant_{stage_name}_int6_tensors"] = int(quant_match.group("int6_tensors"))
        result["metrics"][f"quant_{stage_name}_int8_tensors"] = int(quant_match.group("int8_tensors"))
        result["metrics"][f"quant_{stage_name}_scale_min"] = float(quant_match.group("scale_min"))
        result["metrics"][f"quant_{stage_name}_scale_p50"] = float(quant_match.group("scale_p50"))
        result["metrics"][f"quant_{stage_name}_scale_max"] = float(quant_match.group("scale_max"))
        result["metrics"][f"quant_{stage_name}_clip_frac"] = float(quant_match.group("clip_frac"))
        result["metrics"][f"quant_{stage_name}_near_zero_frac"] = float(quant_match.group("near_zero_frac"))

    if OOM_RE.search(text):
        result["status"] = "failed"
        result["failure_kind"] = "cuda_oom"
        eval_mode_value = result["params"].get("eval_mode")
        if eval_mode_value == "sliding_window":
            result["failure_stage"] = "post_quant_sliding_eval"
        elif eval_mode_value == "standard":
            result["failure_stage"] = "post_quant_standard_eval"
        else:
            result["failure_stage"] = "training_or_eval"
        result["failure_message"] = "CUDA OOM detected in log"
    elif result["status"] == "unknown" and step_matches:
        result["status"] = "incomplete"

    metrics = result["metrics"]
    if "train_step_final_val_bpb" in metrics and "post_quant_val_bpb" in metrics:
        metrics["post_quant_gap_bpb"] = float(metrics["post_quant_val_bpb"]) - float(metrics["train_step_final_val_bpb"])
    if "train_step_final_val_bpb" in metrics and "post_swa_val_bpb" in metrics:
        metrics["swa_penalty_bpb"] = float(metrics["post_swa_val_bpb"]) - float(metrics["train_step_final_val_bpb"])
    if "train_step_final_val_bpb" in metrics and "post_tightswa_val_bpb" in metrics:
        metrics["tightswa_penalty_bpb"] = float(metrics["post_tightswa_val_bpb"]) - float(metrics["train_step_final_val_bpb"])
    if "train_step_final_val_bpb" in metrics and "post_ema_val_bpb" in metrics:
        metrics["ema_penalty_bpb"] = float(metrics["post_ema_val_bpb"]) - float(metrics["train_step_final_val_bpb"])
    if "sliding_window_eval_time_ms" in metrics:
        metrics["eval_time_ms"] = int(metrics["sliding_window_eval_time_ms"])
    elif "post_quant_eval_time_ms" in metrics:
        metrics["eval_time_ms"] = int(metrics["post_quant_eval_time_ms"])
    elif "post_ema_eval_time_ms" in metrics:
        metrics["eval_time_ms"] = int(metrics["post_ema_eval_time_ms"])

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse a Parameter Golf training log into JSON.")
    parser.add_argument("log_path")
    args = parser.parse_args()
    print(json.dumps(parse_train_log(args.log_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
