# seq10l_fp16_maskbos

Loss-only challenger that masks `BOS` targets during training while keeping the
March 19 10-layer FP16-embedding sliding-window champion architecture and eval
path unchanged.

Hypothesis:

- the trainer currently spends capacity learning cross-document transitions into
  the synthetic `BOS` token
- ignoring those targets should reallocate capacity toward in-document token
  prediction with a small positive effect on `post_quant_val_bpb`

This wrapper is meant for ARENA-style control/treatment testing, not yet a
submission record.
