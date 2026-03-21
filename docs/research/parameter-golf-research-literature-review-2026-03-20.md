# Parameter Golf: Research Literature Review & Technical Analysis

**Date:** March 20, 2026
**Purpose:** Evidence-backed technical reference for Plexor Labs' Parameter Golf competition campaign
**Competition:** OpenAI Model Craft Challenge — 16MB artifact, 10 min 8×H100, lowest BPB wins
**Deadline:** April 30, 2026

---

## Evidence status

This note now spans three evidence classes and should be read that way:

- `proxy-only`: local Azure T4 runs used for feasibility, runtime, artifact-size sanity, and pre/post-quant directionality
- `promotion-relevant`: matched serious-lane runs on the intended artifact and evaluation path
- `submission-relevant`: contest-valid `8×H100` runs with enough seeds for statistical claims

Current added local evidence:

- capped Azure T4 proxy on the imported March 20 top int5 family
- pre-quant `val_bpb = 3.1599`
- post-quant `val_bpb = 3.20873382`
- proxy PTQ gap `≈ +0.0488`
- total submission bytes `15,536,094`

This is useful `proxy-only` evidence. It is not leaderboard-comparable and should not be interpreted as a contest-grade score.

---

## 1. Competition framing

The competition is explicitly an L(N) optimization problem: minimize validation loss given a fixed parameter budget. The artifact constraint is:

```text
code_bytes + compressed_model_bytes ≤ 16,000,000 (decimal)
```

The scoring metric is tokenizer-agnostic bits per byte (BPB):

```text
BPB = (cross_entropy_loss / ln(2)) × (tokens_per_byte)
```

OpenAI references Kaplan et al. (2020) neural scaling laws, where loss scales as a power law with parameter count: `L(N) ∝ N^(-α)`, α ≈ 0.076 for transformers on language data. At 16MB we are in the extreme small-N regime where every freed byte has disproportionate value.

### Current leaderboard (March 20, 2026)

| Rank | Score | Key techniques |
|------|-------|----------------|
| 1 | 1.1428 | Int5-MLP, BigramHash(10240), SWA(0.4), 10 layers |
| 2 | 1.1458 | Int6, MLP3x, SmearGate, BigramHash, OrthoInit, SWA |
| 3 | 1.1502 | 11 layers, MLP3x, Int6 QAT, zstd-22, sliding eval |
| 4 | 1.1556 | SmearGate, BigramHash, MLP3x, Int6 STE QAT |
| 9 | 1.1928 | LoRA test-time training |
| 14 | 1.2244 | Naive baseline (9 layer, 512 dim, 1024 vocab) |

Important caveat: leaderboard labels and README summaries do not always perfectly match the exact checked-in trainer code. For local planning, the checked-in `train_gpt.py` files are the source of truth for mechanism claims.

---

## 2. Quantization

### 2.1 Fundamentals

Post-training quantization (PTQ) maps float weights to k-bit integers with per-row or per-tensor scale factors:

```text
x_q = clamp(round(x / s), -2^(k-1)+1, 2^(k-1)-1)
x_hat = x_q × s
MSE ∝ σ² / 2^(2k)
```

Byte cost for a `(m, n)` weight matrix at `k` bits with fp16 per-row scales:

```text
bytes = m × n × k/8 + m × 2
```

Savings from reducing bit width directly translate to freed bytes for more parameters.

### 2.2 Quantization-aware training (QAT)

QAT simulates quantization during the forward pass using the straight-through estimator (STE):

- **Forward:** `W_q = quantize(W)`, use `W_q` for matmul
- **Backward:** `∂L/∂W = ∂L/∂W_q` (treat quantize as identity)

The optimizer learns weights robust to the quantization grid. Models trained with QAT consistently lose less accuracy than PTQ at the same bit width.

Current repo caveat:

- the checked-in March 19 WDQ lineage is PTQ-based
- the imported March 20 frontier trainers are also verified in local code as mixed-quant export plus roundtrip eval
- QAT is therefore still a treatment lane in our local work, not a verified property of every imported frontier family

### 2.3 Literature: mixed precision is the key

**GPTQ (Frantar et al., ICLR 2023):** Achieves optimal post-training quantization at 4-bit precision but experiences decline at 3-bit. Enables models with hundreds of billions of parameters to be compressed to 3-4 bits per parameter without significant accuracy loss.

**LLM-QAT (Liu et al., 2023):** Demonstrates better accuracy with 4-bit weights + 8-bit activations compared to uniform 4-bit. Uses data-free distillation from the full-precision model.

**pQuant (2025):** Decoupled linear QAT. Key finding: parameters possess varying sensitivity, with a small subset of "sensitive" parameters disproportionately influencing model output. Even state-of-the-art 1-bit QAT-Scratch models exhibit a non-negligible accuracy gap, and scaling efficiency is poor — performance gains grow sublinearly as model size increases.

**QuEST (ICLR 2025 Workshop):** The most relevant paper for Parameter Golf. Provides stable QAT convergence from INT1 through INT8. Key result: **INT4 weights and activations are Pareto-optimal** in terms of accuracy at a given model size and inference cost. INT4 + 2:4 sparsity can provide better scaling than any purely quantized representation tested. This directly supports pushing toward int4 for MLP weights.

**BitNet (Wang et al., 2023):** Pioneered QAT from scratch for 1-bit LLMs, achieving 90.1% FP16 parity on downstream tasks. BitNet 1.58b enables near-lossless 2-bit quantization at 3B scale. However, these results are at much larger model scales than Parameter Golf's ~14M parameter regime.

**CrossQuant (2024):** Finds that the "quantization kernel" (small-valued elements rounded to zero) is the primary source of accuracy degradation, not the large-valued outliers. Per-column absolute-maximum vectors are typically smaller than per-row, enabling better quantization. Relevant for improving per-row scale quality.

### 2.4 MXFP4: an unexploited format

**Comprehensive Evaluation on Quantization (2025):** MXFP4 (microscaling floating-point 4-bit) handles long-tail weight distributions better than INT4, since its values are non-uniformly distributed. LLM weights approximately follow normal distributions, and MXFP4 more fully utilizes the representational capacity of 4-bit precision compared to INT4's uniform grid. No leaderboard entry uses FP4 formats — this is a potential differentiation.

### 2.5 Mixed precision strategy for Parameter Golf

The leaderboard's current SOTA uses int5 for MLP weights and int6 for everything else. The literature supports this pattern and suggests pushing further:

| Component | Current SOTA | Literature suggests | Sensitivity |
|-----------|-------------|-------------------|-------------|
| MLP weights | int5 | int4 with QAT (QuEST) | Low — ReLU² kills half activations |
| Attention Q/K/V | int6 | int6 (keep) | High — dot product amplifies noise |
| Embeddings | int8/fp16 | int8/fp16 (keep) | High — first input, propagates |
| Control tensors | fp16 | fp16 (keep) | Very high — gates/scales |

### 2.6 Compression: zstd vs zlib

Multiple leaderboard entries use zstd-22 instead of zlib. Zstd at level 22 is the maximum compression level — slower to compress but the competition only counts compressed size, not compression time. Weight matrices have structured patterns (correlated rows, smooth distributions) that high-effort compressors exploit. The byte savings are modest (~1-3%) but free.

Operational caveat:

- the imported March 20 trainers silently fall back to `zlib` if `zstandard` is unavailable
- some final log labels still say `zlib` even when `zstd` was used
- artifact numbers therefore need the compressor recorded explicitly

---

## 3. Architecture: MLP expansion

### 3.1 Why 3× MLP works

The baseline uses a 2× MLP: hidden = 2 × model_dim. The MLP computes:

```text
MLP(x) = W_down × (relu(W_up × x))²
```

At small model scales, the MLP is where most "knowledge storage" happens. Attention captures token-relationship patterns; the MLP memorizes facts and word associations. Information capacity scales with parameter count, so a 3× MLP stores 50% more information.

This is only viable because aggressive quantization freed the bytes. At int8, a 3× MLP wouldn't fit in 16MB. At int5/int6, it does. Quantization and MLP expansion are complementary — you trade precision per parameter for more parameters, and the net effect is positive because the MLP was capacity-starved.

### 3.2 Squared ReLU

The codebase uses `relu(x)²` rather than standard ReLU or GELU. Squared ReLU creates sparser activations, which has been shown to improve quality at small scale. The sparsity means fewer activations participate in each forward pass, reducing effective interference between stored "facts" in the MLP. This is related to the finding that sparse MoE models are parameter-efficient — squared ReLU gives a soft version of the same effect.

### 3.3 Depth: 10-11 layers

The leaderboard moved from 9 layers (baseline) to 10-11 layers. Each additional layer adds one attention block + one MLP block. At dim=512 with 3× MLP and GQA (4 KV heads), each layer is approximately:

```text
attention: 3 × (512 × 128) + (512 × 512) = 458K params (Q, K, V, O)
MLP: 2 × (512 × 1536) = 1,573K params
per-layer total: ~2.03M params
```

Going from 9 to 11 layers adds ~4M params. At int6, that's ~3MB. The byte budget math works because int5/int6 MLP compression freed enough space.

---

## 4. BigramHash embeddings

### 4.1 Concept

Standard token embeddings are a lookup table `E` of shape `(vocab_size, dim)`. Each token gets one embedding independent of context.

BigramHash adds a second table `B` of shape `(hash_buckets, dim)` that captures token-pair information:

```text
h = hash(t_{i-1}, t_i) mod hash_buckets
embedding(t_i) = E[t_i] + B[h]
```

The hash maps ~1M possible bigram pairs (`1024²`) down to a fixed number of buckets (`10,240` in SOTA).

### 4.2 Why it works

Language has strong local dependencies. The probability of the next token depends heavily on the immediately preceding token. BigramHash front-loads local context at the embedding layer, giving the model a better starting representation before the first transformer layer runs.

### 4.3 Parameter cost

The top imported March 20 int5 family does not implement BigramHash as a raw `10240 × 512` table. The checked-in trainer uses:

- a `10240 × 128` embedding table
- a `128 -> 512` projection
- a learned scalar scale

That is `1,376,257` parameters for the top int5 family, not `5.24M`. This materially changes the byte-budget story and makes BigramHash more affordable than a naive reading suggests.

### 4.4 Hash function design

A good hash distributes bigram pairs uniformly with minimal collision between frequent pairs. The simplest approach:

```text
out[0] = special_bucket
out[t] = xor(36313 * curr, 27191 * prev) mod (hash_buckets - 1)
```

In practice, BigramHash is not a generic “next feature.” It is a family-level choice that depends on already having a compression/export stack that can afford it.

### 4.5 Literature connection

Feature hashing (the "hashing trick") is well-established in ML (Weinberger et al., 2009). BigramHash applies this to n-gram embeddings specifically. The idea also relates to byte-level or character-level n-gram embeddings (FastText, Bojanowski et al. 2017), where subword information is captured through hashing. The key insight for Parameter Golf is that at `vocab_size=1024` with a BPE tokenizer, tokens are short enough that bigram dependencies capture significant prediction signal.

---

## 5. SmearGate

### 5.1 Concept

The checked-in March 20 public trainers implement SmearGate as a small input-stage previous-token blend, not as an input-dependent attention gate.

```text
g = sigmoid(gate_vector)
x_prev = shift_right(x)
out = (1 - g) * x + g * x_prev
```

This happens after embedding addition and RMSNorm, before the transformer blocks.

### 5.2 Why it helps at small scale

In the verified public code, SmearGate is a cheap local-context inductive bias. It lets the model blend each token representation with the previous token before deeper processing. That is simpler and cheaper than a full attention-residual gate, which is part of why it fits cleanly in the frontier artifact budget.

### 5.3 Literature connection: gated attention

**Gated Attention (Qwen Team, NeurIPS 2025 Best Paper):** Adds a learnable sigmoid gate after attention computation. The key insight is that standard softmax attention has a low-rank bottleneck: `W_V × W_O` collapses to a rank-`d_head` matrix. A sigmoid gate breaks this bottleneck because it cannot be factored out or merged with linear projections, increasing expressiveness.

The SmearGate in Parameter Golf is likely a simplified version: per-dimension scalar gating (`dim,`) rather than full matrix gating (`dim, dim`), keeping parameter cost minimal (~`1K` params per layer).

### 5.4 Connection to the baseline's existing gates

The baseline already has `attn_scale`, `mlp_scale`, and `resid_mix` per-layer control tensors. SmearGate is different: it is a global learned previous-token blend applied at the input side, not a per-token MLP gate over attention outputs.

---

## 6. Sliding window evaluation

### 6.1 Concept

Standard eval: chop validation set into non-overlapping chunks of length `L`, score each independently. First token in each chunk has zero context.

Sliding window with stride `S < L`: score the full window of `L` tokens, but only count loss on the last `S` tokens (which had full `L`-length context):

```text
total_loss = Σ_i loss(tokens[i*S : i*S + L], count_only_last_S=True)
```

### 6.2 Mathematical justification

Language models are conditional: `P(t_i | t_1...t_{i-1})` improves monotonically with more context (up to the model's effective context length). By ensuring every scored token has substantial context, you measure the model's actual conditional distribution rather than an artificially degraded version.

### 6.3 Cost

Computational: you process `L/S` times more tokens. The competition allows `10` minutes for evaluation, so the tradeoff is eval compute time vs. BPB improvement.

The leaderboard entry "Sliding Window Eval" (`1.1925`) beat the baseline (`1.2244`) by `0.032` BPB — purely from evaluation protocol changes with no model modification.

---

## 7. Stochastic Weight Averaging (SWA)

### 7.1 Concept

Instead of using the final checkpoint, SWA averages weights from the last `K` training steps. In exponential moving average form with coefficient `α`:

```text
W_SWA = α × W_SWA + (1-α) × W_current
```

SWA(`0.4`) means `α=0.4`: the running average gives `40%` weight to accumulated average, `60%` to each new checkpoint.

### 7.2 Theory

**Izmailov et al. (2018):** SGD with learning rate schedules traverses a valley in the loss landscape but oscillates around the minimum. The average of oscillating points lies closer to the valley center, which generalizes better. SWA finds flatter minima correlated with better generalization.

### 7.3 Parameter Golf application

Applied during the warmdown phase when learning rates decay. The model is already converging, and averaging smooths out noise from final gradient steps. Zero additional parameters, zero additional compute — purely free BPB.

---

## 8. Orthogonal initialization (OrthoInit)

### 8.1 Concept

Set weight matrices to random orthogonal matrices `Q` where `Q^T Q = I`.

### 8.2 Why it matters for small, time-limited models

For a layer `y = Wx`, gradient backpropagation is `∂L/∂x = W^T × ∂L/∂y`. If `W` is orthogonal, `W^T W = I`, so gradient magnitude is preserved exactly. This prevents vanishing/exploding gradients at initialization.

In a `10-11` layer network trained for only `10` minutes, faster early learning translates directly to lower final loss. At convergence the weights are no longer orthogonal, but the initial dynamics are smoother.

### 8.3 Literature

**Saxe et al. (2013):** Exact solutions to the nonlinear dynamics of learning in deep linear neural networks show that orthogonal initialization leads to faster convergence. **Hu et al. (2020):** Provable benefits of orthogonal initialization in training deep networks.

---

## 9. Test-time training (TTT) with LoRA

### 9.1 Concept

Standard inference: `P(t_i | t_1...t_{i-1}) = model(t_1...t_{i-1})`

TTT adds adaptation before scoring:

1. Freeze base model weights `W`
2. Attach LoRA adapters: `W' = W + A × B` (`rank r ≪ dim`)
3. Run gradient steps on a prefix of the validation document
4. Score the continuation with the adapted model
5. Discard adapter weights, start fresh for next document

### 9.2 LoRA fundamentals

**Hu et al. (2021):** Pre-trained language models have a low intrinsic dimension. Weight updates during adaptation have low intrinsic rank — even rank `r=1` or `r=2` suffices when full rank is `12,288`. LoRA reduces trainable parameters by `10,000×` and GPU memory by `3×` compared to full fine-tuning, with on-par or better quality.

For Parameter Golf with rank `r=4` and dim=`512`:

- Per adapted matrix: `2 × 512 × 4 = 4,096` params
- Adapters on `Q`, `K`, `V`, `O` across `10` layers: ~`160K` trainable params
- Small enough to optimize in a few gradient steps

### 9.3 TLM: Test-Time Learning for LLMs

**Hu et al. (2025):** Proposes TTL paradigm for LLMs. Key findings directly applicable:

1. **Perplexity minimization as objective:** More accurate predictions from LLMs can be achieved by minimizing input perplexity of unlabeled test data. This is exactly what next-token prediction gives you — the validation text is its own training signal.

2. **High-perplexity samples are more informative:** Prioritize adapting on the hardest parts of validation data, not uniformly across all chunks. In Parameter Golf, this means spending more TTT gradient steps on high-loss validation segments.

3. **LoRA prevents catastrophic forgetting:** Full-parameter optimization risks destroying base model knowledge; LoRA's low-rank constraint acts as a regularizer.

4. **20% improvement on domain-shifted data.** However, this was measured on out-of-distribution data. Parameter Golf trains and evaluates on FineWeb — same distribution — so TTT gains will be smaller.

### 9.4 Test-Time Training on Nearest Neighbors (Hardt & Sun, ICLR 2024)

Showed TTT works for LLMs specifically. The approach retrieves nearest-neighbor examples at test time and uses them for adaptation. For Parameter Golf, the "nearest neighbor" is the document prefix itself.

### 9.5 VDS-TTT (2025)

Uses a learned verifier to score generated responses, selecting high-confidence pseudo-labeled examples for TTT. Key technique: filtering adaptation signal quality. For Parameter Golf, this suggests that not all prefix tokens are equally useful for adaptation — the model should weight its TTT gradient steps toward tokens where the base model is most uncertain.

### 9.6 Parameter Golf leaderboard evidence

The sole TTT submission scored `1.1928` (9th place), ~`0.03` BPB better than its non-TTT base but `0.05` BPB behind SOTA. This suggests TTT gives real but modest in-distribution gains. The question is whether TTT on a stronger backbone (`1.1428` SOTA rather than ~`1.22` base) compounds or saturates.

### 9.7 Eval-time compute budget

The competition allows `10` minutes for evaluation on `8×H100`. TTT inner loop budget per chunk:

```text
available_seconds = 600
num_val_chunks = total_val_tokens / eval_seq_len
seconds_per_chunk = 600 / num_val_chunks
```

With ~`50K` validation documents, this is ~`12ms` per document — tight. Practical TTT may need to batch documents or adapt on chunks rather than per-document.

---

## 10. Depth recurrence / weight sharing

### 10.1 Concept

Instead of `N` unique layers, train `K` unique layers and loop them `N/K` times. You get the depth of `N` layers with the parameter cost of `K` layers. Freed bytes go into wider dimensions or more capacity.

Example: `3` unique layers looped `4×` = `12` effective layers at the parameter cost of `3`.

### 10.2 Literature

**Universal Transformer (Dehghani et al., 2018):** Applies the same transformation repeatedly with per-step adaptive halting. Shows that weight sharing across layers can match or exceed unique-layer models.

**ALBERT (Lan et al., 2019):** Cross-layer parameter sharing reduces parameters by ~`90%` with minimal accuracy loss. Shows that factorized embedding parameters + cross-layer sharing is effective for BERT-scale models.

**Block Recurrence Transformer (Hutchins et al., 2022):** Applies recurrence at the block level, enabling longer effective context with fixed parameter count.

### 10.3 Per-iteration adapters (5×2 loop LoRA)

The most parameter-efficient version: share base weights across loops but add tiny per-iteration LoRA adapters. Each loop gets a unique low-rank perturbation, allowing the model to specialize each pass without duplicating the full layer.

Cost: for rank `r=2`, each adapter per layer is `2 × 512 × 2 = 2,048` params. Across `5` unique layers × `2` loops × `4` projection matrices = `40K` params total — negligible.

### 10.4 Parameter Golf opportunity

**Nobody on the leaderboard uses depth recurrence.** All top entries use `9-11` unique layers. If looped layers + adapters can match `11` unique layers at the parameter cost of `5-6` layers, the freed bytes buy significantly more MLP capacity or wider dimensions. This is the largest unexploited architectural direction.

---

## 11. Structured sparsity

### 11.1 Concept

`N:M` structured sparsity (e.g., `2:4`) zeroes out `N` elements in every group of `M`, creating a fixed sparsity pattern that hardware can exploit. Combined with quantization, it's multiplicative: you save bytes from both.

### 11.2 Literature

**QuEST (2025):** INT4 + 2:4 sparsity provides better scaling than any purely quantized representation tested. This is the strongest evidence that sparsity + quantization is the right frontier for extreme compression.

**NVIDIA Ampere 2:4 sparsity:** Hardware-native support for 50% structured sparsity with no throughput penalty on A100/H100 tensor cores.

### 11.3 Parameter Golf opportunity

No leaderboard entry uses sparsity. At `50%` sparsity + int4 quantization, effective bits per parameter drops to `2` bits, enabling a model with roughly `4×` more parameters than int8 in the same byte budget. The risk is that `2` effective bits per parameter may be too aggressive at `16MB` scale — but QuEST's results suggest otherwise with QAT.

---

## 12. Non-uniform quantization

### 12.1 Concept

Standard int-k uses a uniform grid of `2^k` levels. Non-uniform quantization uses a learned codebook of `2^k` centroids optimized to minimize reconstruction error.

### 12.2 Literature

**GPTVQ (2024):** The "blessing of dimensionality" for LLM quantization — vector quantization in high-dimensional weight space achieves better rate-distortion tradeoffs than scalar quantization.

**QuIP / QuIP# (Cornell, 2023-2024):** Achieves the first viable 2-bit LLM quantization by combining incoherence processing with lattice codebooks. Uses random orthogonal rotations to spread outlier information across dimensions.

### 12.3 Parameter Golf opportunity

All leaderboard entries use uniform integer quantization. Non-uniform codebooks could squeeze more information from the same bit budget. Implementation complexity is higher — you need to design the codebook, train with it, and include the codebook in the artifact. But the rate-distortion improvement is theoretically guaranteed.

---

## 13. Warmdown schedule

### 13.1 Concept

The training script uses a wallclock-aware warmdown: in the final phase, learning rates decay based on elapsed time rather than step count. This adapts automatically to `8×H100` vs `1×H100`.

```python
def lr_mul(step, elapsed_ms):
    remaining_ms = max(max_wallclock_ms - elapsed_ms, 0)
    return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
```

### 13.2 Interaction with WarmdownQuantization (WDQ)

The checked-in March 19 WDQ lineage in this repo is best understood as a warmdown-based training recipe paired with post-training quantization and roundtrip evaluation.

That means:

- WDQ is a verified serious control family
- QAT is still a legitimate treatment on top of that family
- a large post-quant gap in a repro should not be described as “WDQ failed to run built-in QAT,” because the checked-in lineage is PTQ-based

### 13.3 Muon optimizer

The codebase uses Muon (Kellerjordan, 2024) for matrix parameters: orthogonalizes gradients via Newton-Schulz iteration before applying momentum. This is a second-order-like update that's particularly effective for small models where the gradient landscape is noisier. The Muon weight decay parameter (`WD=0.04` in SOTA) prevents overfitting during the extended training.

---

## 14. Unexplored directions from literature

### 14.1 Knowledge distillation at small scale

**FBI-LLM (Ma et al., 2024):** Trains fully binarized LLMs from scratch via autoregressive distillation from a larger teacher. For Parameter Golf, you could train a large teacher unconstrained by the `16MB` limit, then distill into the artifact-constrained student. This is legal — the competition only constrains the final artifact and the `10`-minute training window, but hyperparameter tuning and external compute for teacher training are allowed.

### 14.2 Learned rotation for quantization

**SpinQuant (Liu et al., 2024) / QuaRot (Ashkboos et al., 2024):** Apply learned orthogonal rotations to weight matrices before quantization, spreading outlier values across dimensions for more uniform distributions. This makes the quantization grid more efficient without changing bit width.

### 14.3 Mixture of Experts (MoE) at tiny scale

No leaderboard entry uses MoE. At `16MB`, you could potentially fit a sparse MoE with `2-4` experts per layer, where only `1` expert is active per token. This doubles effective capacity with the same compute (one expert forward pass) and similar byte cost (experts share quantization overhead). Risk: MoE routing overhead may not pay off at `512` dimensions.

### 14.4 Custom tokenizer optimization

The baseline uses a `1024`-token BPE vocabulary. Larger vocab means more embedding bytes but fewer tokens to predict; smaller vocab means cheaper embeddings but harder per-token prediction. The optimal vocabulary size is an empirical question that depends on the exact byte budget tradeoff. No leaderboard entry has explored vocab sizes beyond the baseline `1024`.

### 14.5 Attention alternatives

**GLA Transformer (Yang et al., 2024):** Gated Linear Attention achieves competitive performance with linear-time inference complexity. At `16MB` scale, the reduced compute per layer could enable more layers within the `10`-minute training budget.

---

## 15. References

### Quantization

- Frantar, E. & Alistarh, D. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR 2023.
- Liu, Z. et al. (2023). LLM-QAT: Data-Free Quantization Aware Training for Large Language Models. arXiv.
- Wang, H., Ma, S. & Wei, F. (2023). BitNet: Scaling 1-Bit Transformers for Large Language Models. arXiv.
- Ma, S. et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arXiv.
- pQuant (2025). Towards Effective Low-Bit Language Models via Decoupled Linear QAT. arXiv.
- QuEST (2025). Quantization with Efficient Scaling for Training. ICLR Workshop on Sparsity in LLMs.
- CrossQuant (2024). Post-Training Quantization with Smaller Quantization Kernel. arXiv.
- Mixed-Precision Quantization for Language Models: Techniques and Prospects (2025). arXiv.

### Adaptation and TTT

- Hu, E.J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Hu, J. et al. (2025). Test-Time Learning for Large Language Models (TLM). arXiv.
- Hardt, M. & Sun, Y. (2024). Test-Time Training on Nearest Neighbors for Large Language Models. ICLR 2024.
- VDS-TTT (2025). Continuous Self-Improvement of LLMs by TTT with Verifier-Driven Sample Selection. arXiv.

### Architecture

- Dehghani, M. et al. (2018). Universal Transformers. ICLR 2019.
- Lan, Z. et al. (2019). ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. ICLR 2020.
- Yang, S. et al. (2024). Gated Linear Attention Transformers with Hardware-Efficient Training. ICML 2024.
- Gated Attention for Large Language Models (Qwen Team). NeurIPS 2025 Best Paper.

### Scaling Laws and Optimization

- Kaplan, J. et al. (2020). Scaling Laws for Neural Language Models. arXiv.
- Izmailov, P. et al. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. UAI 2018.
- Saxe, A. et al. (2013). Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Networks.

### Feature Hashing and Embeddings

- Weinberger, K. et al. (2009). Feature Hashing for Large Scale Multitask Learning. ICML 2009.
- Bojanowski, P. et al. (2017). Enriching Word Vectors with Subword Information. TACL.

### Perplexity-Guided Compression and Adaptation

- Jiang, H. et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. EMNLP 2023.
- Pan, Z. et al. (2024). LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. ACL Findings 2024.
- Hu, J. et al. (2025). Test-Time Learning for Large Language Models (TLM). arXiv.
- VDS-TTT (2025). Continuous Self-Improvement of LLMs by TTT with Verifier-Driven Sample Selection. arXiv.

---

## 16. Perplexity-guided techniques (LLMlingua family)

### 16.1 Core insight

LLMlingua uses a small language model to score each token's perplexity in context. High-perplexity tokens carry more information (surprising, hard to predict). Low-perplexity tokens carry less (predictable, redundant). The original application is prompt compression — dropping low-perplexity tokens to shrink prompts by `10-20×` while preserving meaning. But the underlying principle that a model's own uncertainty identifies where information is concentrated applies broadly to the Parameter Golf setting.

BPB is literally a compression metric. The model IS a compressor. Perplexity-aware techniques can make it a better one by allocating resources (compute, precision, adaptation) where the information density is highest.

### 16.2 Application 1: Perplexity-weighted TTT (highest value)

**Status: directly applicable, implementable now, nobody on leaderboard is doing it**

Standard TTT runs gradient steps uniformly across a document prefix. But not all tokens in the prefix are equally informative for adaptation. "The" and "of the" teach the model nothing. A domain-specific term or an unusual syntactic structure teaches it a lot.

The TLM paper (Hu et al. 2025) already found that high-perplexity samples are more informative for test-time learning. LLMlingua formalizes this further. The combination suggests:

**Implementation:**

1. During TTT, compute per-token loss on the prefix in a single forward pass
2. Create a mask: keep tokens with loss above a threshold (e.g., top `30%` by perplexity)
3. Backpropagate only through high-perplexity tokens (zero gradient for easy tokens)
4. This concentrates the adaptation signal on informative tokens

**Why this matters for Parameter Golf specifically:**

The eval-time compute budget is `10` minutes for ~`50K` validation documents. That's ~`12ms` per document. With standard uniform TTT, most of those gradient steps are wasted on trivially predictable tokens. Perplexity-weighted TTT spends `100%` of the adaptation budget on tokens that actually move the loss.

Expected effect: same or better BPB improvement per TTT step, enabling either (a) fewer steps for the same gain (faster eval) or (b) more effective adaptation in the same time budget (better BPB).

**Connection to existing TTT submission:**

The public LoRA TTT record's ablation revealed that most of its gain came from document isolation and strided evaluation, not the LoRA adaptation itself. This is consistent with the hypothesis that uniform adaptation is weak — the LoRA steps are diluted by easy tokens. Perplexity-weighted TTT would specifically address this weakness.

### 16.3 Application 2: Adaptive computation per token (speculative)

**Status: speculative, requires architecture change, higher implementation cost**

Standard evaluation spends equal compute on every token — all layers run for every position. But some tokens are trivially predictable ("the" after "of") and some are genuinely hard ("quantum" after "the study of").

**Early-exit mechanism:**

1. After each transformer layer, compute a confidence score (e.g., entropy of the current logit distribution)
2. If confidence exceeds a threshold, skip remaining layers for that token
3. Easy tokens exit after `3-4` layers; hard tokens use all `10-11` layers

The byte cost is minimal — just the early-exit logic and threshold. The compute savings from early-exiting easy tokens could be reinvested in:

- More TTT steps for hard tokens
- Longer eval context for hard spans
- Running the model at higher effective depth for the tokens that matter most

**Challenges:**

- The causal attention mask means skipping a layer for token `i` affects the key/value cache for tokens `i+1`, `i+2`, etc. Need careful implementation to avoid corrupting future predictions.
- May require training with the early-exit mechanism active (similar to Universal Transformer's adaptive halting) for the model to learn to produce good intermediate predictions.
- Adds code complexity to `train_gpt.py`, consuming artifact bytes.

### 16.4 Application 3: Perplexity-guided quantization precision (novel)

**Status: novel hypothesis, untested, potentially high-value for sub-1.0 push**

LLMlingua's insight is that different tokens have different information density. The analogous insight for weights: different weight rows have different impact on high-perplexity predictions.

**Implementation:**

1. Run a calibration forward pass on a subset of training data
2. Compute per-token losses and identify high-perplexity tokens
3. Compute the gradient magnitude of each weight row with respect to the loss on high-perplexity tokens only
4. Weight rows with high gradient magnitude on hard tokens → quantize at int8 (preserve precision where it matters for hard predictions)
5. Weight rows with low gradient magnitude → quantize at int4 (save bytes on weights that only matter for easy predictions)

This is a form of sensitivity-aware mixed precision, but driven by the perplexity distribution of actual text rather than generic Hessian heuristics.

**Connection to existing literature:**

- pQuant (2025) identifies sensitive parameters through decoupled analysis — this approach uses perplexity as the sensitivity signal instead
- GPTQ uses Hessian diagonal for sensitivity — this is cheaper (one forward pass with loss decomposition vs. second-order computation)
- OWQ (AAAI 2024) uses outlier-aware weight quantization based on Hessian matrices — the perplexity-guided version targets the same goal with a task-specific signal

**Byte budget impact:**

If `30%` of weight rows are "important for hard predictions" and quantized at int8, while `70%` are quantized at int4, the average is ~`5.2` bits per parameter. Compared to uniform int6 (`6` bits), this saves ~`13%` of model bytes. Those bytes can fund more parameters, a wider MLP, or additional layers. The key bet is that the quality distribution of predictions is better — you preserve precision exactly where it matters most.

### 16.5 What's NOT legal

- Cannot use LLMlingua to compress the validation data itself — BPB must cover all bytes
- Cannot use an external model during evaluation — everything must come from the `16MB` artifact
- Cannot skip tokens during BPB scoring — every byte in the validation set contributes to the score

All perplexity-guided techniques must use the model's own perplexity estimates in real time as part of the forward pass, not an external oracle.

### 16.6 Priority ranking within this family

| Technique | Feasibility | Expected impact | Implementation cost |
|-----------|-------------|-----------------|---------------------|
| Perplexity-weighted TTT | High — simple masking during existing TTT loop | `0.005-0.015` BPB improvement over uniform TTT | Low — ~20 lines of code |
| Perplexity-guided quantization | Medium — requires calibration pass | `0.005-0.02` BPB via better precision allocation | Medium — custom quantizer |
| Adaptive early-exit | Low — requires architecture + training changes | Unknown | High — significant complexity |

---

## 17. Peer review findings (March 21, 2026)

Key corrections from external review of the research note:

1. **Artifact utilization is massively under-spent.** Current serious control uses `6.58MB` of the `16MB` budget (`41%`). Leaders use `15.8-15.96MB`. This is the single largest gap — not technique, but byte budget allocation.
   This criticism applies to the older internal proxy/control framing. The imported March 20 frontier families are already near cap, and the capped Azure T4 proxy on the top int5 family still used `15,536,094` bytes. The practical lesson is family selection, not “spend more bytes at all costs.”

2. **eval@2048 is neutral-to-negative for well-trained 8×H100 runs.** The official WDQ writeup says eval@1408 is preferred. This kills the eval2048 treatment for the serious lane.

3. **TTT's actual incremental contribution is small.** The public TTT submission's ablation shows most gain came from document isolation and strided evaluation, not the LoRA adaptation step itself. TTT is a bonus lane, not the centerpiece.

4. **The correct ordering is: contest-grade baseline → public board stack → TTT.** Not eval → TTT → structure.

5. **Sub-1.0 is a moonshot requiring a step-change.** Current frontier ablations are in the `0.0001-0.003` BPB range. The big one-time eval gains have been harvested. A new inductive bias or evaluation method is needed.

6. **Evidence tiers must be explicit.** Proxy-only runs can justify “escalate” or “do not escalate,” but not leaderboard comparisons, serious-lane promotion, or contest-facing p-values.

---

## 18. Summary: what to test, ranked by evidence strength

*Updated March 21, 2026 to reflect peer review and leaderboard analysis*

| Priority | Technique | Evidence | Expected BPB gain | Compute cost |
|----------|-----------|----------|--------------------|--------------|
| 1 | Reproduce the public frontier family on contest hardware | Verified public record code + leaderboard proof | Match the active family, not just the headline score | 1 serious run |
| 2 | Keep serious-lane family selection explicit | Imported metadata + peer review | Avoid wasting effort on branches with no byte headroom | Low |
| 3 | Family-aligned QAT / export-robustness treatment | QuEST + capped T4 proxy gap `+0.0488` | Meaningful post-quant improvement if aligned to the right exporter | 3-5 runs |
| 4 | Int4/int5 mixed-precision MLP experiments on the frontier exporter | QuEST + public board pattern | `0.005-0.015` | 3-5 runs |
| 5 | Depth / capacity moves within the already-near-cap frontier family | Public board pattern | `0.005-0.02` | 3-5 runs |
| 6 | Perplexity-weighted LoRA TTT | TLM + LLMlingua + leaderboard TTT ablation | Bonus-lane upside after the base family is strong | 5-10 runs |
| 7 | Depth recurrence with per-loop adapters | Universal Transformer + ALBERT | `0.01-0.05` (high variance) | 5-10 runs |
| 8 | INT4 + 2:4 structured sparsity | QuEST | `0.01-0.03` | 5-10 runs |
| 9 | Perplexity-guided mixed-precision quantization | pQuant + LLMlingua (novel synthesis) | `0.005-0.02` | Medium impl. cost |
| 10 | Non-uniform quantization (vector codebook) | GPTVQ, QuIP# | `0.005-0.02` | High impl. cost |
| 11 | Learned rotation pre-quantization | SpinQuant, QuaRot | `0.005-0.01` | Moderate |
| 12 | Adaptive early-exit per token | Theoretical + LLMlingua insight | Unknown | High impl. cost |
| 13 | Knowledge distillation from larger teacher | FBI-LLM | Unknown | Needs teacher training |

---

## 19. Current local evidence

### 19.1 Proxy-only evidence

Current reliable local proxy result:

- family: imported March 20 top int5 frontier record
- hardware: Azure T4
- eval path: capped proxy validation (`VAL_MAX_SEQS=512`)
- pre-quant `val_bpb = 3.1599`
- post-quant `val_bpb = 3.20873382`
- proxy PTQ gap `≈ +0.0488`
- total submission bytes `15,536,094`

What this does support:

- the frontier family runs cleanly on the proxy lane
- the artifact remains near contest scale
- post-quant degradation is real but not catastrophic

What this does not support:

- leaderboard comparison
- contest-grade significance
- serious-lane promotion by itself

---

*This document is a living reference. Update as new leaderboard entries land and as ARENA battles produce measured results.*
