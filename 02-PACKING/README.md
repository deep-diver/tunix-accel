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
- `02-PACKING/run_efficiency_benchmark.py`
- `02-PACKING/run_gemma_tokenizer_benchmark.py`

Tests:

- `tests/test_packing.py`
- `tests/test_packing_model_parity.py`
- `tests/test_tunix_gemma_packing_smoke.py`

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

For Tunix trainers, use:

```python
tunix_batch = packed.as_tunix()
```

This intentionally maps `loss_mask` to the Tunix argument named `input_mask`.
That naming is easy to trip over: in Tunix's decoder-LM loss path, `input_mask`
is applied to shifted next-token targets. The token-valid mask remains available
as `valid_mask`.

## Validation So Far

Gemma-free validation:

```bash
python -m pytest -q tests/test_packing.py tests/test_packing_model_parity.py
```

This verifies the core invariant with a tiny JAX causal LM:

- packed rows use fewer batch rows than separate examples
- packed loss equals separate-example loss
- replacing the block-causal mask with a plain causal mask changes the loss,
  proving that cross-segment contamination would be observable

Optional Gemma/Tunix validation:

```bash
python -m pytest -q tests/test_tunix_gemma_packing_smoke.py
```

This test skips when Tunix is not installed. In a Tunix environment, it builds a
tiny random Gemma3 model and checks that Tunix's default decoder-LM loss matches
between separate examples and packed examples.

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

## No-Model Efficiency Benchmark

The first model-free benchmark has been run on 5,000 OPUS100 EN-FR training
examples using a simple regex token-count proxy. This does not require Gemma,
Tunix, or a TPU; it only measures sequence-length packing efficiency.

Artifacts:

- `02-PACKING/results/no-model/README.md`
- `02-PACKING/results/no-model/packing_efficiency.csv`
- `02-PACKING/results/no-model/packing_efficiency_overview.png`
- `02-PACKING/results/no-model/packing_batch_sensitivity.png`

Headline at batch 16:

| Max Length | Fixed Unpacked | Dynamic Unpacked | Packed | Rows Reduction | Gain vs Fixed | Gain vs Dynamic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 17.5% | 35.5% | 99.4% | 5.66x | 5.67x | 2.80x |
| 512 | 8.8% | 34.5% | 99.5% | 11.26x | 11.28x | 2.88x |
| 1024 | 4.4% | 34.4% | 99.6% | 22.52x | 22.56x | 2.90x |
| 2048 | 2.2% | 34.4% | 99.6% | 45.05x | 45.12x | 2.90x |

Interpretation: fixed max-length padding is the harsh baseline and shows why
long-context SFT can waste almost all sequence slots on short datasets. Dynamic
padding is a stronger baseline; packing still improved useful-token density by
about 2.8-2.9x at batch 16 on this sample.

## Gemma Tokenizer Benchmark

The next benchmark uses the actual `google/gemma-3-270m-it` tokenizer and a
Gemma-style turn format for OPUS100 EN-FR. It still does not instantiate model
weights; the purpose is to verify that the data path and sequence lengths remain
favorable under real Gemma tokenization.

Artifacts:

- `02-PACKING/results/gemma-tokenizer/README.md`
- `02-PACKING/results/gemma-tokenizer/gemma_tokenizer_packing.csv`
- `02-PACKING/results/gemma-tokenizer/gemma_tokenizer_packing_overview.png`

Headline at batch 16:

| Max Length | Fixed Unpacked | Dynamic Unpacked | Packed | Rows Reduction | Gain vs Fixed | Gain vs Dynamic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 22.7% | 38.6% | 98.4% | 4.33x | 4.33x | 2.55x |
| 512 | 11.5% | 37.1% | 99.0% | 8.62x | 8.63x | 2.67x |
| 1024 | 5.7% | 36.9% | 99.4% | 17.30x | 17.33x | 2.70x |
| 2048 | 2.9% | 36.9% | 99.8% | 34.72x | 34.78x | 2.71x |

Interpretation: the real Gemma tokenizer makes examples slightly longer than the
regex proxy, but the story remains the same. Packing turns a short-example
translation workload from roughly 37% dynamic-padding token density to about
99% packed density at batch 16.

## Current Status

This branch contains the first reusable packing implementation, local parity
tests, optional Gemma/Tunix smoke validation, a no-model packing efficiency
benchmark, and a real Gemma-tokenizer packing benchmark.

## Actual Tunix Training Benchmark

The real Tunix benchmark now runs `PeftTrainer` steps on Gemma3 270M LoRA SFT.
It is intentionally a Default CE baseline; launch it with
`TUNIX_ACCEL_DISABLE_AUTOPATCH=1` so the CCE patch is not installed.

On a TPU VM, install TPU JAX explicitly rather than using the local CPU
requirement as-is:

```bash
python -m pip install google-tunix==0.1.6 kagglehub==0.4.3 \
  datasets matplotlib transformers importlib_resources gcsfs==2026.2.0
python -m pip install -U "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -e .
```

```bash
TUNIX_ACCEL_DISABLE_AUTOPATCH=1 python 02-PACKING/run_gemma_training_benchmark.py \
  --variants unpacked,packed \
  --batch-size 16 \
  --max-length 512 \
  --max-steps 50 \
  --num-examples 5000 \
  --outdir 02-PACKING/results/gemma-training-default-ce
```

For a cheap local data-path check that does not instantiate Gemma or Tunix:

```bash
TUNIX_ACCEL_DISABLE_AUTOPATCH=1 python 02-PACKING/run_gemma_training_benchmark.py \
  --prepare-only \
  --tokenizer-source huggingface \
  --variants unpacked,packed \
  --batch-size 16 \
  --max-length 512 \
  --num-examples 512 \
  --outdir /tmp/tunix-packing-prepare-test
```

The training run writes:

- per-variant `summary.json`
- per-variant `history.csv`
- combined `summary.json`
- combined `history.csv`
- `training_comparison.png`

The key readouts are final loss, step time, valid tokens/sec, loss tokens/sec,
and packed token density. This tells us whether packing improves actual Tunix
training throughput before mixing in CCE.

Artifacts from the first TPU run:

- `02-PACKING/results/gemma-training-default-ce/README.md`
- `02-PACKING/results/gemma-training-default-ce/summary.json`
- `02-PACKING/results/gemma-training-default-ce/history.csv`
- `02-PACKING/results/gemma-training-default-ce/training_comparison.png`

Run environment:

- Cloud TPU `v5litepod-1`, one TPU chip
- Project `gcp-ml-172005`, zone `us-west4-a`
- `google-tunix==0.1.6`, `jax==0.10.1`, `libtpu==0.0.41`
- Model `google/gemma-3-270m-it`
- Dataset OPUS100 EN-FR train split
- Batch 16, max length 512, LoRA rank 16, learning rate 2e-4
- Default CE only; CCE autopatch disabled

Headline result for 50 optimizer steps:

| Variant | Token density | Step time | Valid tok/s | Loss tok/s | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unpacked | 10.5% | 0.108s | 4,936 | 1,538 | 2.2959 |
| Packed | 99.3% | 0.107s | 75,899 | 33,000 | 1.8844 |

Interpretation: packing did not make each optimizer step materially slower in
this small 270M setup, but each step carried far more real training tokens. That
is the useful result for this branch: without touching CE, padding waste alone
can dominate the amount of learning signal delivered per TPU second.

The final loss values above are same-step results, not same-token-budget quality
results. Packed consumed 178,077 loss tokens over 50 steps, while unpacked
consumed 8,414. For quality parity, the next comparison should match consumed
loss tokens or train both variants to the same validation budget.
