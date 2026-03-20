#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

DEFAULTS = {
    "RUN_ID": "seq10l_fp16_maskbos_v0",
    "MASK_BOS_TARGETS": "1",
    "EVAL_STRIDE": "64",
    "EVAL_BATCH_SEQS": "64",
}

BOS_ID = 1


def load_champion_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("seq10l_fp16_champion", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"unable to load delegated trainer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_masked_loss(module) -> None:
    def masked_forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = module.F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x_norm = self.final_norm(x)
        x_flat = x_norm.reshape(-1, x_norm.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = module.F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * module.torch.tanh(logits_proj / self.logit_softcap)
        per_token = module.F.cross_entropy(logits.float(), targets, reduction="none")

        if os.environ.get("MASK_BOS_TARGETS", "1") != "1":
            return per_token.mean()

        keep = (targets != BOS_ID).to(dtype=per_token.dtype)
        return (per_token * keep).sum() / keep.sum().clamp_min(1.0)

    module.GPT.forward = masked_forward


def main() -> None:
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)

    champion_path = (
        Path(__file__).resolve().parents[2]
        / "records"
        / "track_10min_16mb"
        / "2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit"
        / "train_gpt.py"
    )
    if not champion_path.exists():
        raise SystemExit(f"missing delegated trainer: {champion_path}")

    champion = load_champion_module(champion_path)
    patch_masked_loss(champion)
    champion.main()


if __name__ == "__main__":
    main()
