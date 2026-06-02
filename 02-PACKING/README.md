# 02-PACKING

This workstream benchmarks and implements padding-free / uncontaminated packing
for Tunix SFT. The goal is to reproduce the part of the Unsloth story that does
not come from the loss kernel itself: short examples should stop wasting most of
the model sequence length as padding.

## Why This Comes After CCE

CCE made larger batch/context combinations trainable by attacking the full vocab
logits tensor. Packing attacks a different waste source:

```text
ordinary padded batch cost ~= batch_size * max_length_in_batch
packed cost                ~= sum(real example lengths), rounded into rows
```

For real SFT datasets with many short rows, increasing batch size often increases
padding waste. Packing should turn that wasted capacity into valid training
tokens while preserving loss correctness.

## Implemented In This Branch

Core implementation:

- `tunix_accel/packing.py`

Tests:

- `tests/test_packing.py`

The packer is model-agnostic and Tunix-light. It accepts tokenized examples and
returns fixed-length rows with:

- `input_ids`
- `labels`
- `loss_mask`
- `input_mask`
- `positions`
- `segment_ids`
- optional block-causal `attention_mask`

The correctness rule is simple: each packed segment behaves like its own sample.
Positions reset to zero at segment boundaries, and the attention mask blocks
tokens from attending across segment boundaries.

## Usage

```python
from tunix_accel.packing import pack_records

records = [
    {"id": "a", "input_ids": [10, 11, 12, 13]},
    {"id": "b", "input_ids": [20, 21]},
    {"id": "c", "input_ids": [30, 31, 32]},
]

packed = pack_records(
    records,
    max_length=6,
    pad_token_id=0,
    strategy="best_fit_decreasing",
)

batch = packed.as_numpy()
```

`batch["attention_mask"]` has shape `[batch, query, key]`. If a Tunix model path
needs a singleton head axis, callers can expand it before feeding the trainer.

## Benchmark Plan

The first benchmark should mirror Unsloth's cleanest packing story:

1. Choose a real SFT dataset with varied sequence lengths.
2. Tokenize once and retain the per-example lengths.
3. Compare ordinary padded batches vs packed batches at the same `max_length`.
4. Report valid-token ratio, tokens/sec, step time, XLA planned HBM, and loss
   curve parity.
5. Repeat with CCE enabled and disabled so the interaction is visible.

The plot set should stay simple:

- valid-token ratio by batch size
- tokens/sec by batch size
- XLA peak HBM by batch size
- loss vs consumed tokens for packed vs unpacked

## Current Status

This branch contains the first reusable packing implementation and local unit
tests. It has not yet run the TPU/Tunix benchmark. The next step is wiring this
into the Tunix data pipeline used for the Gemma3 270M EN-FR run, then producing
the same style of before/after report we used for `01-CCE`.
