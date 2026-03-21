#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import io
import inspect
import json
import os
from pathlib import Path
import zlib

import sentencepiece as spm
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a quantized Parameter Golf artifact across multiple eval sequence lengths.")
    parser.add_argument("--module-path", default="train_gpt.py", help="Trainer module to import.")
    parser.add_argument("--artifact", default="final_model.int8.ptz", help="Path to quantized artifact.")
    parser.add_argument("--seq-lens", default="512,1024,1408,2048,4096,8192", help="Comma-separated eval seq lengths.")
    parser.add_argument("--stride", type=int, default=64, help="Sliding-window stride. Set 0 to disable sliding eval.")
    parser.add_argument("--output-json", default="", help="Optional path to write JSON results.")
    return parser.parse_args()


def load_module(module_path: str):
    path = Path(module_path).resolve()
    spec = importlib.util.spec_from_file_location("pg_eval_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args_ns = parse_args()
    module = load_module(args_ns.module_path)
    args = module.Hyperparameters()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for eval sweep")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    seq_lens = [int(token.strip()) for token in args_ns.seq_lens.split(",") if token.strip()]
    max_seq_len = max(seq_lens)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = module.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    val_tokens = module.load_validation_tokens(args.val_files, max_seq_len)

    gpt_kwargs = {
        "vocab_size": args.vocab_size,
        "num_layers": args.num_layers,
        "model_dim": args.model_dim,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "mlp_mult": args.mlp_mult,
        "tie_embeddings": args.tie_embeddings,
        "tied_embed_init_std": args.tied_embed_init_std,
        "logit_softcap": args.logit_softcap,
        "rope_base": args.rope_base,
        "qk_gain_init": args.qk_gain_init,
    }
    if "mlp_hidden" in inspect.signature(module.GPT).parameters:
        gpt_kwargs["mlp_hidden"] = getattr(args, "mlp_hidden", 0)
    if "rope_train_seq_len" in inspect.signature(module.GPT).parameters:
        gpt_kwargs["rope_train_seq_len"] = getattr(args, "rope_train_seq_len", 1024)
    model = module.GPT(**gpt_kwargs).to(device).bfloat16()
    for submodule in model.modules():
        if isinstance(submodule, getattr(module, "CastedLinear")):
            submodule.float()
    module.restore_low_dim_params_to_fp32(model)

    with open(args_ns.artifact, "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu")
    model.load_state_dict(module.dequantize_state_dict_int8(quant_state), strict=True)

    results: list[dict[str, float | int]] = []
    for seq_len in seq_lens:
        eval_val_kwargs = {
            "args": args,
            "model": model,
            "rank": 0,
            "world_size": 1,
            "device": device,
            "grad_accum_steps": 1,
            "val_tokens": val_tokens,
            "base_bytes_lut": base_bytes_lut,
            "has_leading_space_lut": has_leading_space_lut,
            "is_boundary_token_lut": is_boundary_token_lut,
        }
        if "eval_seq_len" in inspect.signature(module.eval_val).parameters:
            eval_val_kwargs["eval_seq_len"] = seq_len
        val_loss, val_bpb = module.eval_val(**eval_val_kwargs)
        row: dict[str, float | int] = {
            "eval_seq_len": seq_len,
            "exact_val_loss": val_loss,
            "exact_val_bpb": val_bpb,
        }
        if args_ns.stride > 0 and hasattr(module, "eval_val_sliding"):
            slide_loss, slide_bpb = module.eval_val_sliding(
                args,
                model,
                rank=0,
                world_size=1,
                device=device,
                val_tokens=val_tokens,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
                eval_seq_len=seq_len,
                eval_stride=args_ns.stride,
            )
            row["sliding_val_loss"] = slide_loss
            row["sliding_val_bpb"] = slide_bpb
            row["eval_stride"] = args_ns.stride
        results.append(row)

    payload = {
        "module_path": str(Path(args_ns.module_path).resolve()),
        "artifact": str(Path(args_ns.artifact).resolve()),
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args_ns.output_json:
        Path(args_ns.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
