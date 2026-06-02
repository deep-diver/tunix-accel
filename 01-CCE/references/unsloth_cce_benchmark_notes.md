# Unsloth / CCE Benchmark Notes

This note fixes the target shape for the Tunix/JAX TPU work. The goal is not a
generic "some memory went down" story; it is the specific Cut Cross-Entropy
story reported by Apple CCE and Unsloth.

## What They Claim

- Apple CCE targets the CE/logits bottleneck directly: it avoids materializing
  the full `[tokens, vocab]` logits matrix, computes the correct-token logit,
  and evaluates log-sum-exp over vocabulary on the fly. Their Gemma 2 2B example
  reports loss memory dropping from 24 GB to 1 MB, and classifier-head
  training-time memory from 28 GB to 1 GB.
- Their benchmark script reports `cce`, `torch_compile`, and `baseline` rows for
  loss forward, backward, and forward+backward. On A100 SMX4 / PyTorch 2.4.1 /
  CUDA 12.4, the README's Gemma2 example reports baseline loss forward+backward
  at 208.8 ms / 28000 MB and CCE at 134.8 ms / 1164 MB.
- Unsloth's long-context materials report the user-visible version of the same
  improvement: lower VRAM, longer context, and no speed/accuracy degradation.
  Their 500K context write-up reports 60% lower VRAM and 3.2x longer context via
  fused/chunked CE. Their Llama 3.3 write-up reports >75% less VRAM and 12-13x
  longer context when CCE is combined with Unsloth gradient checkpointing.

## What We Should Reproduce On TPU

1. CE-local memory:
   Compare default full-logits CE against streaming CCE on the isolated LM-head
   loss. The primary memory proxy is the largest temporary logits tile:
   `[token_chunk, vocab_chunk]`, not `[tokens, vocab]`.

2. OOM frontier:
   For each model/TPU topology, find the largest passing `(batch, seq_len)` for
   default and for streaming CCE. This mirrors Unsloth's "longer context" story
   better than a post-training pprof snapshot.

3. Runtime:
   Record loss forward+backward time and end-to-end train step time. CCE should
   not buy memory by catastrophically slowing the step.

4. Loss parity:
   With the same fixed batch and LR=0 or a single deterministic step, final loss
   should match dense CE within numerical tolerance.

## Sources

- Apple CCE / Cut Your Losses:
  https://github.com/apple/ml-cross-entropy
- Unsloth CCE fork and benchmark table:
  https://github.com/unslothai/cut-cross-entropy
- Unsloth 500K context write-up:
  https://docs.unsloth.ai/new/500k-context-length-fine-tuning
- Unsloth Llama 3.3 long-context write-up:
  https://unsloth.ai/blog/llama3-3
