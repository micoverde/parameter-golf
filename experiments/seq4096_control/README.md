# Seq4096 Control Candidate

This control wrapper delegates directly to the accepted in-repo `TrainingOptSeq4096` record script.

Use it as the fixed control arm when evaluating the incremental value of the sliding-window treatment on the same `seq_len=4096` training regime.

Run with:

```bash
RUN_ID=seq4096_control_v0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 \
  experiments/seq4096_control/train_gpt.py
```

This is a research scaffold, not a contest submission folder yet.
