# Reproducing the Sequence Packing Experiments

This guide records how to reproduce the final sequence-packing experiment family
after removing raw intermediate artifacts and checkpoints.

## 1. What Remains

The final retained artifacts are:

- Report: `02-PACKING/TECHNICAL_REPORT.md`
- Figures: `02-PACKING/assets/*.png`
- Summary data:
  - `02-PACKING/data/no_model_packing_efficiency.csv`
  - `02-PACKING/data/no_model_length_summary.json`
  - `02-PACKING/data/gemma_tokenizer_packing.csv`
  - `02-PACKING/data/gemma_tokenizer_length_summary.json`
  - `02-PACKING/data/gemma3_270m_enfr_quality_summary.csv`
  - `02-PACKING/data/gemma3_270m_enfr_translation_samples.md`
  - `02-PACKING/data/gemma3_1b_4b_scale_smoke_summary.csv`
  - `02-PACKING/data/gemma4_base_packing_tpu_l2048_b1.csv`

Removed artifacts include raw TPU run directories, full per-step histories,
checkpoint directories, old smoke outputs, and intermediate result folders.

## 2. Patch Code

The patch code is kept at the repository root:

- `tunix_accel/packing.py`
- `tunix_accel/tunix_packing.py`

Install the package into the Python environment that runs Tunix:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The core packer can be used directly:

```python
from tunix_accel.packing import pack_records

packed = pack_records(
    records,
    max_length=512,
    pad_token_id=0,
    strategy="best_fit_decreasing",
)
batch = packed.as_tunix()
```

For existing Tunix datasets that already yield padded batches, keep the normal
Tunix trainer flow and add an optional packing config when registering the model
input function:

```python
from tunix_accel import TunixPackingConfig

trainer = trainer.with_gen_model_input_fn(
    gen_model_input_fn,
    packing=TunixPackingConfig(max_length=512, pad_token_id=0),
)
trainer.train(train_ds, eval_ds)
```

The adapter changes the training dataset and model input function only for that
trainer. It does not patch the model, optimizer, or loss formula. Omit
`packing=` to run Tunix normally.

## 3. Verification Tests

Run the local packing tests:

```bash
python -m pytest -q \
  tests/test_packing.py \
  tests/test_packing_model_parity.py \
  tests/test_tunix_packing.py \
  tests/test_tunix_gemma_packing_smoke.py
```

The Gemma/Tunix smoke test skips when Tunix is not installed. In a Tunix
environment, it builds a tiny random Gemma3 model and checks that packed examples
match separate examples through Tunix's decoder-LM loss path.

## 4. Reproduce the No-Model Efficiency Benchmark

Purpose: check whether the dataset length distribution is favorable before
loading any model.

```bash
python 02-PACKING/run_efficiency_benchmark.py \
  --dataset opus100-en-fr \
  --num-examples 5000 \
  --batch-sizes 1,2,4,8,16,32,64 \
  --max-lengths 256,512,1024,2048 \
  --outdir /tmp/tunix-packing-no-model
```

Retain:

```text
packing_efficiency.csv
length_summary.json
packing_efficiency_overview.png
packing_batch_sensitivity.png
```

Final copies are stored as:

```text
02-PACKING/data/no_model_packing_efficiency.csv
02-PACKING/data/no_model_length_summary.json
02-PACKING/assets/no_model_packing_efficiency_overview.png
02-PACKING/assets/no_model_packing_batch_sensitivity.png
```

## 5. Reproduce the Gemma Tokenizer Benchmark

Purpose: verify that the opportunity remains under the actual Gemma tokenizer
and prompt wrapper.

```bash
python 02-PACKING/run_gemma_tokenizer_benchmark.py \
  --model-id google/gemma-3-270m-it \
  --num-examples 5000 \
  --batch-sizes 1,2,4,8,16,32,64 \
  --max-lengths 256,512,1024,2048 \
  --outdir /tmp/tunix-packing-gemma-tokenizer
```

Retain:

```text
gemma_tokenizer_packing.csv
length_summary.json
gemma_tokenizer_packing_overview.png
```

Final copies are stored as:

```text
02-PACKING/data/gemma_tokenizer_packing.csv
02-PACKING/data/gemma_tokenizer_length_summary.json
02-PACKING/assets/gemma_tokenizer_packing_overview.png
```

## 6. TPU Setup

The retained TPU experiments used Cloud TPU v5e in `us-west4-a`.

Recommended TPU shapes:

| Experiment | Model | TPU | Chips |
| --- | --- | --- | ---: |
| 270M 50-step smoke | `google/gemma-3-270m-it` | `v5litepod-1` | 1 |
| 270M quality sanity run | `google/gemma-3-270m-it` | `v5litepod-1` | 1 |
| 1B scale smoke | `google/gemma-3-1b-it` | `v5litepod-4` | 4 |
| 4B scale smoke | `google/gemma-3-4b-it` | `v5litepod-4` | 4 |
| Gemma4 E2B boundary row | `google/gemma-4-E2B` | `v5litepod-4` | 4 |
| Gemma4 E4B boundary row | `google/gemma-4-E4B` | `v5litepod-8` | 8 |

Create a TPU VM:

```bash
gcloud compute tpus tpu-vm create TPU_NAME \
  --project=PROJECT_ID \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-4 \
  --version=tpu-ubuntu2204-base
```

On the TPU VM, install Python 3.11 or newer, create a virtualenv, install TPU
JAX, then install this package:

```bash
python -m pip install google-tunix==0.1.6 kagglehub==0.4.3 \
  datasets matplotlib transformers importlib_resources gcsfs==2026.2.0 \
  sacrebleu pytest
python -m pip install -U "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -e .
```

Use the repository-level disable flag for isolated packing runs:

```bash
export TUNIX_ACCEL_DISABLE_AUTOPATCH=true
```

Boolean environment values are case-insensitive. The parser accepts
`true/false`, `1/0`, `yes/no`, and `on/off`; examples use `true/false` for
readability.

## 7. Reproduce the 270M Training Smoke

Purpose: verify that packing improves actual Tunix target-token throughput.

Common settings:

| Field | Value |
| --- | --- |
| Model | `google/gemma-3-270m-it` |
| Dataset | OPUS100 EN-FR |
| Batch | 16 |
| Max length | 512 |
| Steps | 50 |
| LoRA rank | 16 |
| Learning rate | 2e-4 |

Run:

```bash
TUNIX_ACCEL_DISABLE_AUTOPATCH=true python 02-PACKING/run_gemma_training_benchmark.py \
  --model-id google/gemma-3-270m-it \
  --variants unpacked,packed \
  --num-examples 5000 \
  --batch-size 16 \
  --max-length 512 \
  --max-steps 50 \
  --skip-quality-eval \
  --outdir /tmp/tunix-packing-270m-smoke
```

The key expected qualitative result is nearly unchanged step time but much
higher target-token throughput for packed batches.

## 8. Reproduce the 270M Quality Sanity Run

Purpose: check that the training path is not obviously broken when packing is
used for a longer run.

Run two jobs:

```bash
TUNIX_ACCEL_DISABLE_AUTOPATCH=true python 02-PACKING/run_gemma_training_benchmark.py \
  --model-id google/gemma-3-270m-it \
  --variants unpacked \
  --num-examples 5000 \
  --batch-size 16 \
  --max-length 512 \
  --max-steps 5000 \
  --outdir /tmp/tunix-packing-270m-unpacked-5k
```

```bash
TUNIX_ACCEL_DISABLE_AUTOPATCH=true python 02-PACKING/run_gemma_training_benchmark.py \
  --model-id google/gemma-3-270m-it \
  --variants packed \
  --num-examples 5000 \
  --batch-size 16 \
  --max-length 512 \
  --max-steps 1000 \
  --outdir /tmp/tunix-packing-270m-packed-1k
```

Aggregate:

```bash
python 02-PACKING/aggregate_training_quality.py \
  --run /tmp/tunix-packing-270m-unpacked-5k \
  --run /tmp/tunix-packing-270m-packed-1k \
  --outdir /tmp/tunix-packing-270m-quality \
  --sample-limit 16
```

Retain:

```text
summary.csv
translation_samples.md
loss_curves.png
metric_bars.png
```

Final copies are stored as:

```text
02-PACKING/data/gemma3_270m_enfr_quality_summary.csv
02-PACKING/data/gemma3_270m_enfr_translation_samples.md
02-PACKING/assets/gemma3_270m_enfr_loss_curves.png
02-PACKING/assets/gemma3_270m_enfr_metric_scorecard.png
```

## 9. Reproduce the 1B/4B Scale Smoke

Purpose: verify that the same throughput effect appears for larger Gemma3
models.

1B:

```bash
TUNIX_ACCEL_DISABLE_AUTOPATCH=true python 02-PACKING/run_gemma_training_benchmark.py \
  --model-id google/gemma-3-1b-it \
  --model-path gs://gemma-data/checkpoints/gemma3-1b-it \
  --variants unpacked,packed \
  --num-examples 5000 \
  --batch-size 8 \
  --max-length 512 \
  --max-steps 50 \
  --mesh-fsdp 4 \
  --mesh-tp 1 \
  --skip-quality-eval \
  --outdir /tmp/tunix-packing-1b-smoke
```

4B:

```bash
TUNIX_ACCEL_DISABLE_AUTOPATCH=true python 02-PACKING/run_gemma_training_benchmark.py \
  --model-id google/gemma-3-4b-it \
  --model-path gs://gemma-data/checkpoints/gemma3-4b-it \
  --variants unpacked,packed \
  --num-examples 5000 \
  --batch-size 4 \
  --max-length 512 \
  --max-steps 50 \
  --mesh-fsdp 4 \
  --mesh-tp 1 \
  --skip-quality-eval \
  --outdir /tmp/tunix-packing-4b-smoke
```

Aggregate:

```bash
python 02-PACKING/aggregate_scale_smoke.py \
  --run Gemma3-1B=/tmp/tunix-packing-1b-smoke \
  --run Gemma3-4B=/tmp/tunix-packing-4b-smoke \
  --outdir /tmp/tunix-packing-scale-smoke \
  --accelerator-type v5litepod-4
```

Retain:

```text
summary.csv
loss_vs_useful_tokens.png
throughput_and_density.png
```

Final copies are stored as:

```text
02-PACKING/data/gemma3_1b_4b_scale_smoke_summary.csv
02-PACKING/assets/gemma3_1b_4b_loss_vs_useful_tokens.png
02-PACKING/assets/gemma3_1b_4b_throughput_and_density.png
```

## 10. Reproduce the Gemma4 Base Negative Control

Purpose: confirm that packing does not change the compile boundary for a fixed
already-full Gemma4 base shape. This is a memory negative control, not a
useful-token throughput benchmark.

Run default and packed rows on E2B `v5litepod-4`, then repeat on E4B
`v5litepod-8`:

```bash
python tools/run_gemma4_base_benchmark.py \
  --model-size e2b \
  --variant default \
  --batch-size 1 \
  --max-length 2048 \
  --max-steps 3 \
  --num-examples 128 \
  --lora-rank 16 \
  --outdir /tmp/gemma4-packing-boundary

python tools/run_gemma4_base_benchmark.py \
  --model-size e2b \
  --variant packed \
  --batch-size 1 \
  --max-length 2048 \
  --max-steps 3 \
  --num-examples 128 \
  --lora-rank 16 \
  --outdir /tmp/gemma4-packing-boundary
```

The retained table is:

```text
02-PACKING/data/gemma4_base_packing_tpu_l2048_b1.csv
```

## 11. Verification Checklist

After rerunning, the expected qualitative result is:

- packed rows maintain local loss parity with separate examples
- L512 packed density is roughly 99% on this OPUS100 EN-FR setup
- packed and unpacked step times are close at the same static shape
- packed target-token throughput is much higher because each step is denser
- 270M translation sanity metrics remain in the same rough band

The exact retained run showed:

| Metric | Unpacked | Packed |
| --- | ---: | ---: |
| Gemma3 270M quality run wall time | 604s | 181s |
| Gemma3 270M quality run target tokens | 1,753,490 | 3,330,580 |
| Gemma3 1B smoke target tok/s | 403 | 8,437 |
| Gemma3 4B smoke target tok/s | 199 | 4,615 |
