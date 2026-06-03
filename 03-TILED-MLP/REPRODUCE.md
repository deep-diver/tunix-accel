# Reproducing the Gemma3 Tiled MLP Experiments

This guide records how to reproduce the final Tiled MLP experiment family after
removing intermediate result folders and checkpoints.

## 1. What Remains

The final retained artifacts are:

- Report: `03-TILED-MLP/TECHNICAL_REPORT.md`
- Figures:
  - `03-TILED-MLP/assets/gemma3_4b_context_boundary_memory.png`
  - `03-TILED-MLP/assets/gemma3_4b_validation_summary.png`
  - `03-TILED-MLP/assets/gemma3_4b_cce_composition_smoke.png`
- Summary data:
  - `03-TILED-MLP/data/gemma3_4b_context_keypoints.csv`
  - `03-TILED-MLP/data/gemma3_4b_validation_summary.csv`
  - `03-TILED-MLP/data/gemma3_4b_validation_history.csv`
  - `03-TILED-MLP/data/gemma3_4b_direct_parity.json`
  - `03-TILED-MLP/data/gemma3_4b_cce_composition_summary.csv`
  - `03-TILED-MLP/data/gemma3_4b_translation_samples.md`
- Raw final records: `03-TILED-MLP/data/raw/`

Removed artifacts include old plot attempts, exploratory smoke folders, TPU log
files, full checkpoint directories, and intermediate result folders.

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
export TUNIX_ACCEL_TILED_MLP_LORA_ALPHA=32.0
export TUNIX_ACCEL_DISABLE_TILED_MLP=1
export TUNIX_ACCEL_DISABLE_CE=1
export TUNIX_ACCEL_DISABLE_AUTOPATCH=1
```

Use `TUNIX_ACCEL_DISABLE_TILED_MLP=1` for the Default MLP baseline. Use
`TUNIX_ACCEL_DISABLE_CE=1` to isolate MLP experiments from the CCE patch.

## 3. Local Verification Tests

Run:

```bash
python -m pytest -q tests/test_tiled_mlp.py tests/test_gemma3_tiled_mlp.py
```

These tests cover dense-vs-tiled forward parity, gradient parity, JIT parity,
Gemma3 `FeedForward.block` integration, Gemma3 remat call-path parity, Tunix SFT
loss smoke, and Qwix-LoRA projection handling.

## 4. TPU Setup

The retained TPU experiments used Cloud TPU v5e in `us-west4-a`.

Recommended shape:

| Experiment | Model | TPU | Chips |
| --- | --- | --- | ---: |
| 4B context keypoints | `google/gemma-3-4b-it` | `v5litepod-8` | 8 |
| 4B 500-step validation | `google/gemma-3-4b-it` | `v5litepod-8` | 8 |
| 4B same-model parity | `google/gemma-3-4b-it` | `v5litepod-8` | 8 |

Create a TPU VM:

```bash
gcloud compute tpus tpu-vm create TPU_NAME \
  --project=PROJECT_ID \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-8 \
  --version=tpu-ubuntu2204-base
```

Copy the repository to the TPU VM, install dependencies, then install this
package.

To capture XLA planned HBM reports:

```bash
export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=/tmp/xla_dump"
```

Retain only the relevant `*memory-usage-report.txt` files or compact parsed JSON
summaries after each run.

## 5. Reproduce Same-Model Parity

Purpose: compare Default MLP and Tiled MLP on the same loaded model instance and
same first batch.

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
  --outdir /tmp/tiled-mlp-parity
```

Retain:

```text
parity_summary.json
```

Final copy:

```text
03-TILED-MLP/data/gemma3_4b_direct_parity.json
```

## 6. Reproduce Context Keypoints

Purpose: find whether the same 4B LoRA setup completes when context length is
increased and only the MLP implementation changes.

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

Default MLP:

```bash
TUNIX_ACCEL_DISABLE_CE=1 \
TUNIX_ACCEL_DISABLE_TILED_MLP=1 \
python 03-TILED-MLP/run_gemma_training_benchmark.py \
  --model-size 4b \
  --mlp-variant default \
  --batch-size 1 \
  --max-length 2048 \
  --max-steps 5 \
  --num-examples 64 \
  --lora-rank 16 \
  --mesh-fsdp 8 \
  --mesh-tp 1 \
  --skip-quality-eval \
  --outdir /tmp/tiled-mlp-keypoints/l2048/default
```

Tiled MLP:

```bash
TUNIX_ACCEL_DISABLE_CE=1 \
TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK=128 \
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
  --outdir /tmp/tiled-mlp-keypoints/l4096/tiled
```

Repeat for the four `(context, variant)` combinations. Record status, runtime
peak memory, mean step time, final loss, and parsed XLA planned HBM.

Final copy:

```text
03-TILED-MLP/data/gemma3_4b_context_keypoints.csv
```

## 7. Reproduce the 500-Step Validation Smoke

Purpose: check that the memory-saving path trains and evaluates in the same
rough loss band. This is not a completed translation-quality benchmark.

Default MLP:

```bash
TUNIX_ACCEL_DISABLE_CE=1 \
TUNIX_ACCEL_DISABLE_TILED_MLP=1 \
python 03-TILED-MLP/run_gemma_training_benchmark.py \
  --model-size 4b \
  --mlp-variant default \
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
  --outdir /tmp/tiled-mlp-quality/default
```

Tiled MLP:

```bash
TUNIX_ACCEL_DISABLE_CE=1 \
TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK=128 \
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
  --outdir /tmp/tiled-mlp-quality/tiled
```

Retain:

```text
summary.json
history.csv
translations.jsonl
```

Final compact copies:

```text
03-TILED-MLP/data/gemma3_4b_validation_summary.csv
03-TILED-MLP/data/gemma3_4b_validation_history.csv
03-TILED-MLP/data/gemma3_4b_translation_samples.md
03-TILED-MLP/data/raw/
```

## 8. Reproduce the Composition Smoke

Purpose: verify that Tiled MLP can coexist with CCE in the same installed
package.

Use Gemma3 4B IT, batch 1, max length 4096, LoRA rank 16, v5litepod-8. Run the
four variants:

- Default CE + Default MLP
- CCE + Default MLP
- Default CE + Tiled MLP
- CCE + Tiled MLP

Retain status, XLA planned HBM, runtime peak memory, mean step time, and final
loss. Final compact copy:

```text
03-TILED-MLP/data/gemma3_4b_cce_composition_summary.csv
```
