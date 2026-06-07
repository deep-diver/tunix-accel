# Gemma3 270M Packing Experiment Design

This workstream treats Gemma3 270M as the exhaustive base case for sequence
packing. Larger models should only be transfer checks after the 270M story is
clear.

## Claim

Sequence packing converts padding waste into useful training tokens. It should
be reported as a data-efficiency and throughput optimization, not as a model
memory optimizer.

## Fixed Scope

| Field | Value |
| --- | --- |
| Model | `google/gemma-3-270m-it` |
| Training | LoRA SFT |
| Dataset | OPUS100 EN-FR, Gemma IT translation wrapper |
| Loss mask | Target-only |
| TPU | Cloud TPU `v5litepod-1` for primary 270M runs |
| CE path | Default Tunix CE; CCE disabled unless explicitly testing composition |

## Implementation Checks

1. Core packing preserves `input_ids`, labels, loss mask, positions, segment ids,
   and block-causal attention.
2. Packed rows reset positions at each original example boundary.
3. Packed attention blocks cross-example reads.
4. A tiny causal LM produces the same loss for packed examples and separate
   examples.
5. A tiny Tunix/Gemma3 model produces the same default-loss and CCE-loss values
   for packed examples and separate examples.
6. The package-installed API exposes `PeftTrainer.with_gen_model_input_fn(...,
   packing=...)` while leaving ordinary Tunix code inert when `packing` is
   omitted.

## 270M Experiment Matrix

### A. No-Model Length/Density Sweep

Purpose: establish whether the dataset length distribution has packing
opportunity before loading Gemma or TPU.

| Knob | Values |
| --- | --- |
| Dataset | OPUS100 EN-FR train subset |
| Examples | 5k, 20k |
| Batch size | 8, 16, 32, 64 |
| Max length | 256, 512, 1024, 2048 |
| Variants | fixed unpacked, dynamic unpacked estimate, packed |

Primary metrics:

- valid token density
- row reduction
- density gain vs fixed unpacked
- density gain vs dynamic unpacked
- packing CPU time

### B. Gemma Tokenizer Density Sweep

Purpose: replace regex token proxy with the actual Gemma tokenizer and prompt
format.

| Knob | Values |
| --- | --- |
| Tokenizer | `google/gemma-3-270m-it` tokenizer |
| Examples | 5k, 20k |
| Batch size | 8, 16, 32, 64 |
| Max length | 256, 512, 1024, 2048 |

Primary metrics:

- valid token density
- target/loss token density
- prompt vs answer length distribution
- packed rows and row reduction

### C. 270M Short Throughput Matrix

Purpose: show that packed batches do not materially change fixed-shape step
time but massively increase useful tokens per step.

| Knob | Values |
| --- | --- |
| TPU | `v5litepod-1`, 1 chip |
| Batch size | 8, 16, 32 |
| Max length | 512, 1024 |
| Steps | 50 |
| Variants | unpacked, packed |
| Quality eval | disabled |

Primary metrics:

- mean step time excluding first logged step
- valid tokens/sec
- loss tokens/sec
- final cumulative loss tokens after 50 steps
- train loss vs cumulative loss tokens
- runtime HBM snapshots

### D. 270M Useful-Token-Budget Parity

Purpose: compare learning behavior at roughly matched useful target-token
budget, not matched optimizer steps.

Initial target:

| Variant | Steps | Batch | Length |
| --- | ---: | ---: | ---: |
| unpacked | 5000 | 16 | 512 |
| packed | 1000 | 16 | 512 |

This is allowed to change after the density sweep. The final report should use
the actually observed target-token counts and state the mismatch if exact
matching is not possible.

Primary metrics:

- loss vs cumulative loss tokens
- wall time
- target/loss tokens/sec
- final train loss
- eval loss on an unpacked validation set
- optional generation samples for sanity, not as headline quality evidence

## Success Criteria

- Packed target-token density is much higher than unpacked on the same dataset.
- Packed useful-token throughput increases approximately in proportion to the
  density gain.
- Same-shape step time stays in the same band.
- Packed and unpacked loss curves are comparable when plotted against useful
  training tokens.
- Boundary leakage tests pass before TPU experiments are trusted.

## Non-Claims

- Packing is not expected to reduce XLA planned HBM for a fixed
  `batch_size * max_length` model shape.
- Packing is not a replacement for CCE or activation-memory optimizations.
- Short EN-FR generation samples are sanity checks, not a translation benchmark.
