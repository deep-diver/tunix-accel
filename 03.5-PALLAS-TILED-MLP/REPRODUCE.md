# Reproducing the Pallas Tiled MLP Experiments

This guide records how to reproduce the final `03.5` experiment family.

## 1. What Remains

The retained artifacts are:

- Report: `03.5-PALLAS-TILED-MLP/TECHNICAL_REPORT.md`
- Figures:
  - `03.5-PALLAS-TILED-MLP/assets/gemma3_4b_pallas_vs_xla_context.png`
  - `03.5-PALLAS-TILED-MLP/assets/gemma3_4b_pallas_validation_smoke.png`
  - `03.5-PALLAS-TILED-MLP/assets/gemma3_4b_pallas_cce_composition.png`
- Summary data:
  - `03.5-PALLAS-TILED-MLP/data/gemma3_4b_pallas_vs_xla_context.csv`
  - `03.5-PALLAS-TILED-MLP/data/gemma3_4b_pallas_vs_xla_validation_summary.csv`
  - `03.5-PALLAS-TILED-MLP/data/gemma3_4b_pallas_vs_xla_validation_history.csv`
  - `03.5-PALLAS-TILED-MLP/data/gemma3_4b_pallas_direct_parity.json`
  - `03.5-PALLAS-TILED-MLP/data/gemma3_4b_pallas_cce_composition_summary.csv`
  - `03.5-PALLAS-TILED-MLP/data/gemma3_4b_pallas_vs_03_delta_summary.csv`
- Raw final records: `03.5-PALLAS-TILED-MLP/data/raw/`

Intermediate TPU dumps, checkpoints, exploratory logs, and full XLA dump
folders are not part of the final package.

## 2. Patch Code

The patch code is kept at the repository root:

- `tunix_accel/tiled_mlp.py`
- `tunix_accel/gemma3_tiled_mlp.py`
- `tunix_accel/autopatch.py`
- `sitecustomize.py`

Install the package into the Python environment that runs Tunix:

```bash
python -m pip install -r requirements.txt
python -m pip install .
```

Use a wheel install for startup-hook validation. Editable installs are fine for
code hacking, but some TPU images already ship a system `sitecustomize.py`; the
wheel install also places a `.pth` startup hook in site-packages.

Useful environment knobs:

```bash
export TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK=128
export TUNIX_ACCEL_TILED_MLP_BACKEND=pallas
export TUNIX_ACCEL_TILED_MLP_BACKEND=xla
export TUNIX_ACCEL_DISABLE_TILED_MLP=1
export TUNIX_ACCEL_DISABLE_CE=1
export TUNIX_ACCEL_DISABLE_AUTOPATCH=1
```

The backend defaults to `xla`. Set `TUNIX_ACCEL_TILED_MLP_BACKEND=pallas` to run
the Pallas TPU matmul backend.

## 3. Local Verification Tests

Run:

```bash
python -m pytest -q tests/test_tiled_mlp.py tests/test_gemma3_tiled_mlp.py tests/test_autopatch.py
```

The tests cover dense and LoRA tiled MLP parity, Pallas backend fallback parity,
Gemma3 block integration, remat call-path parity, and autopatch backend
selection.

## 4. TPU Setup

The retained TPU experiments used Cloud TPU v5e in `us-west4-a`.

| Experiment | Model | TPU | Chips |
| --- | --- | --- | ---: |
| 4B context keypoints | `google/gemma-3-4b-it` | `v5litepod-8` | 8 |
| 4B 500-step validation smoke | `google/gemma-3-4b-it` | `v5litepod-8` | 8 |
| 4B same-model parity | `google/gemma-3-4b-it` | `v5litepod-8` | 8 |
| 4B CCE composition smoke | `google/gemma-3-4b-it` | `v5litepod-8` | 8 |

Create a TPU VM:

```bash
gcloud compute tpus tpu-vm create TPU_NAME \
  --project=PROJECT_ID \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-8 \
  --version=tpu-ubuntu2204-base
```

Copy the repository to the TPU VM, install dependencies, install `jax[tpu]`,
then install this package.

To capture XLA planned HBM reports:

```bash
export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=/tmp/xla_dump"
```

Retain only parsed `*memory-usage-report.txt` summaries after each run.

## 5. Reproduce Same-Model Parity

Purpose: compare Default MLP and Pallas Tiled MLP on the same loaded model
instance and same first batch.

```bash
TUNIX_ACCEL_DISABLE_AUTOPATCH=1 \
TUNIX_ACCEL_DISABLE_CE=1 \
python 03-TILED-MLP/run_gemma3_tiled_mlp_parity.py \
  --model-size 4b \
  --batch-size 1 \
  --max-length 512 \
  --num-examples 64 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --mesh-fsdp 8 \
  --mesh-tp 1 \
  --tiled-mlp-token-chunk 128 \
  --tiled-mlp-backend pallas \
  --outdir /tmp/pallas-tiled-mlp/parity
```

Retain:

```text
parity_summary.json
```

## 6. Reproduce Context Keypoints

Purpose: compare 03 XLA tiled MLP and 03.5 Pallas tiled MLP at the same context
pressure points.

Common settings:

| Field | Value |
| --- | --- |
| Model | Gemma3 4B IT |
| TPU | v5litepod-8 |
| Mesh | `fsdp=8,tp=1` |
| Batch | 1 |
| LoRA rank | 16 |
| CCE | disabled |
| Steps | 5 |
| Contexts | 2048, 4096 |

Pallas Tiled MLP:

```bash
export TUNIX_ACCEL_DISABLE_CE=1
export TUNIX_ACCEL_TILED_MLP_BACKEND=pallas
export TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK=128
export XLA_FLAGS="--xla_dump_to=/tmp/xla_pallas_l4096"

python 03-TILED-MLP/run_gemma_training_benchmark.py \
  --model-size 4b \
  --mlp-variant tiled \
  --batch-size 1 \
  --max-length 4096 \
  --max-steps 5 \
  --num-examples 64 \
  --lora-rank 16 \
  --mesh-fsdp 8 \
  --mesh-tp 1 \
  --skip-quality-eval \
  --tiled-mlp-token-chunk 128 \
  --tiled-mlp-backend pallas \
  --outdir /tmp/pallas-tiled-mlp/context/l4096
```

Repeat for L2048 and L4096. Compare against the retained 03 XLA tiled records in
`03-TILED-MLP/data/gemma3_4b_context_keypoints.csv`.

## 7. Reproduce the 500-Step Validation Smoke

Purpose: check training loss, eval loss, runtime memory, and step time for the
Pallas backend at the same 03 smoke shape.

```bash
export TUNIX_ACCEL_DISABLE_CE=1
export TUNIX_ACCEL_TILED_MLP_BACKEND=pallas
export TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK=128

python 03-TILED-MLP/run_gemma_training_benchmark.py \
  --model-size 4b \
  --mlp-variant tiled \
  --batch-size 1 \
  --max-length 2048 \
  --max-steps 500 \
  --num-examples 5000 \
  --lora-rank 16 \
  --mesh-fsdp 8 \
  --mesh-tp 1 \
  --eval-examples 256 \
  --eval-batches 64 \
  --generation-examples 32 \
  --tiled-mlp-token-chunk 128 \
  --tiled-mlp-backend pallas \
  --outdir /tmp/pallas-tiled-mlp/quality
```

Retain:

```text
tiled/summary.json
tiled/history.csv
tiled/translations.jsonl
```

## 8. Reproduce CCE Composition

Purpose: verify that Pallas Tiled MLP composes with Cut Cross Entropy through
the installed autopatch path.

```bash
export TUNIX_ACCEL_DISABLE_CE=0
export TUNIX_ACCEL_TILED_MLP_BACKEND=pallas
export TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK=128
export TUNIX_ACCEL_CE_TOKEN_CHUNK=128
export TUNIX_ACCEL_CE_VOCAB_CHUNK=8192
export XLA_FLAGS="--xla_dump_to=/tmp/xla_pallas_cce_l4096"

python 03-TILED-MLP/run_gemma_training_benchmark.py \
  --model-size 4b \
  --mlp-variant tiled \
  --batch-size 1 \
  --max-length 4096 \
  --max-steps 5 \
  --num-examples 64 \
  --lora-rank 16 \
  --mesh-fsdp 8 \
  --mesh-tp 1 \
  --skip-quality-eval \
  --tiled-mlp-token-chunk 128 \
  --tiled-mlp-backend pallas \
  --outdir /tmp/pallas-tiled-mlp/composition/cce_pallas
```

Compare against:

```text
03-TILED-MLP/data/gemma3_4b_cce_composition_summary.csv
```

## 9. Clean Up

When the run is complete and summaries have been copied locally:

```bash
gcloud compute tpus tpu-vm delete TPU_NAME \
  --project=PROJECT_ID \
  --zone=us-west4-a \
  --quiet
```
