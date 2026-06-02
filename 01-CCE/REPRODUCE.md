# Reproducing the Cut Cross Entropy Experiments

This guide records how to reproduce the final experiment family after removing
raw intermediate artifacts and checkpoints.

## 1. What Remains

The final retained artifacts are:

- Report: `01-CCE/TECHNICAL_REPORT.md`
- Figures: `01-CCE/assets/*.png`
- Summary data:
  - `01-CCE/data/kernel_matrix.csv`
  - `01-CCE/data/gemma3_270m_1b_context_frontier.csv`
  - `01-CCE/data/gemma3_270m_1b_context_summary.csv`
  - `01-CCE/data/gemma3_b16_aggregate_hbm.csv`
  - `01-CCE/data/quality_training_summary.csv`
  - `01-CCE/data/quality_summary.csv`
  - `01-CCE/data/side_by_side.jsonl`

Removed artifacts include raw `*.xplane.pb` TPU traces, full XLA dump trees,
checkpoint directories, old smoke runs, and intermediate profile visualizations.

## 2. Patch Code

The patch code is kept at the repository root:

- `tunix_accel/chunked_linear_ce.py` - implementation module for Cut Cross
  Entropy; the filename describes the chunked/streaming mechanism.
- `tunix_accel/model_adapters.py`
- `tunix_accel/tunix_lora_ce.py`
- `tunix_accel/tunix_patch.py`
- `tunix_accel/autopatch.py`
- `sitecustomize.py`

Install the package into the Python environment that runs Tunix:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

By default, the installed package uses `sitecustomize.py` to register a lazy
Tunix import hook. When `tunix.sft.peft_trainer` is imported, the patch replaces
Tunix's default decoder-LM loss for supported models.

Useful environment knobs:

```bash
export TUNIX_ACCEL_CE_TOKEN_CHUNK=128
export TUNIX_ACCEL_CE_VOCAB_CHUNK=8192
export TUNIX_ACCEL_DISABLE_AUTOPATCH=1
```

Use `TUNIX_ACCEL_DISABLE_AUTOPATCH=1` for the Default CE baseline. Leave it unset
for the CCE run, or explicitly install the patch:

```python
from tunix_accel.tunix_patch import install

install(token_chunk=128, vocab_chunk=8192)
```

For explicit per-trainer use:

```python
from tunix_accel.tunix_lora_ce import use_frozen_lm_head_ce

trainer = peft_trainer.PeftTrainer(...).with_gen_model_input_fn(...)
trainer = use_frozen_lm_head_ce(
    trainer,
    token_chunk=128,
    vocab_chunk=8192,
)
trainer.train(train_ds, eval_ds)
```

## 3. TPU Setup

The retained experiments used Cloud TPU v5e in `us-west4-a`.

Recommended TPU shapes:

| Experiment | Model | TPU | Chips |
| --- | --- | --- | --- |
| Context sweep 270M | `google/gemma-3-270m-it` | `v5litepod-1` | 1 |
| Context sweep 1B | `google/gemma-3-1b-it` | `v5litepod-4` | 4 |
| b16/L2048 4B pressure point | `google/gemma-3-4b-it` | `v5litepod-8` | 8 |
| EN-FR quality run | `google/gemma-3-270m-it` | `v5litepod-1` | 1 |

Create a TPU VM:

```bash
gcloud compute tpus tpu-vm create TPU_NAME \
  --project=PROJECT_ID \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-1 \
  --version=tpu-ubuntu2204-base
```

Copy the repository to the TPU VM, install Python dependencies, then install
this package with `pip install -e .`.

To capture XLA planned HBM reports, set:

```bash
export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=/tmp/xla_dump"
```

For compact artifacts, keep only `*memory-usage-report.txt` files from the dump
directory after each run.

## 4. Reproduce the Context Frontier

Purpose: isolate memory pressure from the logits tensor. Use synthetic fixed
shape SFT data. The content is intentionally not realistic.

Common settings:

| Field | Value |
| --- | --- |
| Variants | Default CE, CCE |
| LoRA rank | 16 |
| Remat | block |
| Dataset | synthetic |
| Steps | 1 is enough for compile/frontier checks |
| CCE token chunk | 256 |
| CCE vocab chunk | 32768 |

Sweep grid:

| Model | Batches | Context lengths |
| --- | --- | --- |
| Gemma3 270M | 8, 16, 32, 64 | 512, 1024, 2048, 4096, 8192 |
| Gemma3 1B | 8, 16, 32, 64 | 512, 1024, 2048, 4096, 8192, 16384 |

For each `(model, batch, context)` pair:

1. Run baseline with `TUNIX_ACCEL_DISABLE_AUTOPATCH=1`.
2. Run CCE with autopatch enabled and `TUNIX_ACCEL_CE_TOKEN_CHUNK=256`,
   `TUNIX_ACCEL_CE_VOCAB_CHUNK=32768`.
3. Record status as `ok`, `oom`, or `skipped`.
4. Parse the train-step XLA memory report and record max planned HBM.

The final retained frontier table is:

```text
01-CCE/data/gemma3_270m_1b_context_frontier.csv
```

The final retained memory table is:

```text
01-CCE/data/gemma3_270m_1b_context_summary.csv
```

## 5. Reproduce the b16/L2048 Model-Size Pressure Point

Purpose: compare the same batch/context pressure point across model sizes.

Common settings:

| Field | Value |
| --- | --- |
| Batch | 16 |
| Context length | 2048 |
| Dataset | synthetic |
| LoRA rank | 16 |
| Variants | Default CE, CCE |

TPU allocation:

| Model | TPU | Chips |
| --- | --- | --- |
| Gemma3 270M | `v5litepod-1` | 1 |
| Gemma3 1B | `v5litepod-4` | 4 |
| Gemma3 4B | `v5litepod-8` | 8 |

Report aggregate HBM as:

```text
aggregate_xla_hbm_gib = max_per_chip_xla_peak_gib * allocated_chip_count
```

This is a reporting convention for comparing sharded runs. It is not one
contiguous memory pool and not a live TPU profiler allocation sample.

The final retained table is:

```text
01-CCE/data/gemma3_b16_aggregate_hbm.csv
```

## 6. Reproduce the Real EN-FR Quality Run

Purpose: verify that the CCE training path preserves real training behavior.

Common settings:

| Field | Value |
| --- | --- |
| Model | `google/gemma-3-270m-it` |
| Dataset | OPUS100 EN-FR |
| TPU | Cloud TPU v5e `v5litepod-1`, 1 chip |
| Batch | 16 |
| Max length | 512 |
| Steps | 5000 |
| LoRA rank | 16 |
| Learning rate | 2e-4 |
| CCE token chunk | 128 |
| CCE vocab chunk | 8192 |

Run two jobs:

1. Default CE baseline with `TUNIX_ACCEL_DISABLE_AUTOPATCH=1`.
2. Cut Cross Entropy with autopatch enabled.

Save only final metrics, final generated predictions, and summary CSV/JSONL.
Checkpoints can be deleted after evaluation.

Important generation note: the loss path intercepts the model to expose hidden
states. Before generation/evaluation, restore the normal decode path so the
sampler receives vocab logits, not hidden states.

Retained outputs:

```text
01-CCE/data/quality_training_summary.csv
01-CCE/data/quality_summary.csv
01-CCE/data/side_by_side.jsonl
```

## 7. Verification Checklist

After rerunning, the expected qualitative result is:

- Context frontier improves materially with CCE.
- b16/L2048 planned HBM is lower with CCE.
- EN-FR eval loss remains at parity.
- BLEU remains at parity on the small sanity set.
- Same-batch CCE train steps are slower than Default CE.

The exact retained run showed:

| Metric | Default CE b16 | CCE b16 |
| --- | ---: | ---: |
| Eval loss | 1.680377 | 1.680899 |
| XLA train-step peak | 10.21 GiB | 2.21 GiB |
| Profiled step avg | 0.092 s | 0.160 s |
| BLEU | 22.294 | 22.289 |
