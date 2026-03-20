# wdq_layout_adaptive_export

Wild-but-bounded treatment on top of the `WarmdownQuantization` family.

Training stays unchanged. The only intervention is export layout selection for
large 2D quantized tensors:

- quantize as usual
- compare compressed size of `q` vs `q.T.contiguous()`
- store the smaller layout
- transpose back during dequantization if needed

This is aimed at improving serialized artifact efficiency without changing model
semantics.
