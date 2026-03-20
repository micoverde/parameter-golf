#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import os
import zlib
from pathlib import Path

DEFAULTS = {
    "RUN_ID": "wdq_layout_adaptive_export_v0",
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
    "LAYOUT_ADAPTIVE_MIN_NUMEL": "16384",
}


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("wdq_train_gpt", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to load delegated trainer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_layout_adaptive_export(module) -> None:
    min_numel = int(os.environ.get("LAYOUT_ADAPTIVE_MIN_NUMEL", "16384"))
    original_quantize = module.quantize_state_dict_int8
    original_dequantize = module.dequantize_state_dict_int8

    def compressed_len(q) -> int:
        return len(zlib.compress(q.numpy().tobytes(), level=1))

    def quantize_state_dict_int8(state_dict):
        obj, stats = original_quantize(state_dict)
        qmeta = dict(obj.get("qmeta", {}))
        changed = False
        for name, q in list(obj["quantized"].items()):
            if q.ndim != 2 or q.numel() < min_numel:
                continue
            q_t = q.T.contiguous()
            if compressed_len(q_t) >= compressed_len(q):
                continue
            obj["quantized"][name] = q_t
            meta = dict(qmeta.get(name, {}))
            meta["stored_transposed"] = True
            qmeta[name] = meta
            changed = True
        if changed:
            obj["qmeta"] = qmeta
        return obj, stats

    def dequantize_state_dict_int8(obj):
        qmeta = obj.get("qmeta", {})
        needs_fix = any(meta.get("stored_transposed") for meta in qmeta.values())
        if not needs_fix:
            return original_dequantize(obj)

        fixed = copy.deepcopy(obj)
        for name, meta in fixed.get("qmeta", {}).items():
            if meta.get("stored_transposed"):
                fixed["quantized"][name] = fixed["quantized"][name].T.contiguous()
                meta = dict(meta)
                meta.pop("stored_transposed", None)
                fixed["qmeta"][name] = meta
        return original_dequantize(fixed)

    module.quantize_state_dict_int8 = quantize_state_dict_int8
    module.dequantize_state_dict_int8 = dequantize_state_dict_int8


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

    module = load_module(target)
    patch_layout_adaptive_export(module)
    module.main()


if __name__ == "__main__":
    main()
