# Reproducing the Gemma3 Activation Policy Experiments

This guide records how to reproduce the retained activation remat/offload
experiment family.

## 1. What Remains

The retained artifacts are:

- Report: `04-ACTIVATION-POLICY/TECHNICAL_REPORT.md`
- Figures:
  - `04-ACTIVATION-POLICY/assets/gemma3_4b_activation_hbm_headroom.png`
  - `04-ACTIVATION-POLICY/assets/gemma3_4b_activation_before_after_memory.png`
  - `04-ACTIVATION-POLICY/assets/gemma3_4b_l4096_activation_frontier.png`
  - `04-ACTIVATION-POLICY/assets/gemma3_4b_l2048_activation_tradeoff.png`
  - `04-ACTIVATION-POLICY/assets/gemma3_small_model_activation_followup.png`
- Summary data:
  - `04-ACTIVATION-POLICY/data/gemma3_4b_activation_policy_keypoints.csv`
  - `04-ACTIVATION-POLICY/data/gemma3_4b_activation_policy_parity.json`
  - `04-ACTIVATION-POLICY/data/gemma3_small_model_activation_followup.csv`
  - `04-ACTIVATION-POLICY/data/MANIFEST.json`
- Raw final records: `04-ACTIVATION-POLICY/data/raw/`
- Small-model raw follow-up:
  `04-ACTIVATION-POLICY/results/small-model-splash-activation-ablation/`

Removed artifacts include checkpoints, full exploratory result folders, TPU log
streams, and duplicate tarballs.

## 2. Patch Code

The patch code is kept at the repository root:

- `tunix_accel/gemma3_activation_policy.py`
- `tunix_accel/autopatch.py`
- `sitecustomize.py`

Install the package into the Python environment that runs Tunix:

```bash
python -m pip install -r requirements.txt
python -m pip install .
```

Useful environment knobs:

```bash
export TUNIX_ACCEL_ACTIVATION_POLICY=split_offload
export TUNIX_ACCEL_ACTIVATION_PREVENT_CSE=0
export TUNIX_ACCEL_ACTIVATION_OFFLOAD_SRC=device
export TUNIX_ACCEL_ACTIVATION_OFFLOAD_DST=pinned_host
export TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY=1
export TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION=1
export TUNIX_ACCEL_DISABLE_CE=1
export TUNIX_ACCEL_DISABLE_TILED_MLP=1
export TUNIX_ACCEL_DISABLE_AUTOPATCH=1
```

Use `TUNIX_ACCEL_ACTIVATION_POLICY=none` or unset it for the baseline. Use
`TUNIX_ACCEL_DISABLE_CE=1` and `TUNIX_ACCEL_DISABLE_TILED_MLP=1` to isolate the
activation policy from the other package patches.

## 3. Local Verification Tests

Run:

```bash
python -m pytest -q tests/test_gemma3_activation_policy.py tests/test_autopatch.py
```

The broader local smoke used before committing the branch was:

```bash
python -m pytest -q \
  tests/test_gemma3_activation_policy.py \
  tests/test_autopatch.py \
  tests/test_gemma3_tiled_mlp.py
```

## 4. TPU Setup

The retained TPU experiments used Cloud TPU v5e in `us-west4-a`.

| Experiment | Model | TPU | Chips |
| --- | --- | --- | ---: |
| 4B activation keypoints | `google/gemma-3-4b-it` | `v5litepod-8` | 8 |
| 4B same-model parity | `google/gemma-3-4b-it` | `v5litepod-8` | 8 |
| 4B Splash + offload ablation | `google/gemma-3-4b-it` | `v5litepod-16` | 16 |
| 270M small-model follow-up | `google/gemma-3-270m-it` | `v5litepod-1` | 1 |
| 1B small-model follow-up | `google/gemma-3-1b-it` | `v5litepod-4` | 4 |

Create a TPU VM:

```bash
gcloud compute tpus tpu-vm create TPU_NAME \
  --project=PROJECT_ID \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-8 \
  --version=tpu-ubuntu2204-base
```

Copy the repository to the TPU VM, install dependencies, then install this
package. The retained run used Python 3.11, `google-tunix==0.1.6`, JAX `0.10.1`,
and `libtpu==0.0.41`.

To capture XLA planned HBM reports:

```bash
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
```

Retain the `*jit__train_step*memory-usage-report.txt` files for the summary.

## 5. Reproduce L2048 Tradeoff

Purpose: measure before/after activation-offload overhead at a context length
that both variants can complete.

Common settings:

| Field | Value |
| --- | --- |
| Model | Gemma3 4B IT |
| TPU | v5litepod-8 |
| Mesh | `fsdp=8,tp=1` |
| Batch | 1 |
| LoRA rank | 16 |
| CCE | disabled |
| Tiled MLP | disabled |
| Steps | 5 |
| Context | 2048 |

Baseline:

```bash
TUNIX_ACCEL_DISABLE_CE=1 \
TUNIX_ACCEL_DISABLE_TILED_MLP=1 \
XLA_FLAGS="--xla_dump_to=/tmp/xla_act_l2048_none" \
python 04-ACTIVATION-POLICY/run_gemma_training_benchmark.py \
  --model-size 4b \
  --activation-policy none \
  --batch-size 1 \
  --max-length 2048 \
  --max-steps 5 \
  --num-examples 64 \
  --lora-rank 16 \
  --mesh-fsdp 8 \
  --mesh-tp 1 \
  --skip-quality-eval \
  --outdir /tmp/activation-policy-04/context/l2048/none
```

Split offload:

```bash
TUNIX_ACCEL_DISABLE_CE=1 \
TUNIX_ACCEL_DISABLE_TILED_MLP=1 \
XLA_FLAGS="--xla_dump_to=/tmp/xla_act_l2048_split_offload" \
python 04-ACTIVATION-POLICY/run_gemma_training_benchmark.py \
  --model-size 4b \
  --activation-policy split_offload \
  --batch-size 1 \
  --max-length 2048 \
  --max-steps 5 \
  --num-examples 64 \
  --lora-rank 16 \
  --mesh-fsdp 8 \
  --mesh-tp 1 \
  --skip-quality-eval \
  --outdir /tmp/activation-policy-04/context/l2048/split_offload
```

## 6. Reproduce L4096 Frontier

Purpose: test whether activation offload moves a real compile-OOM boundary.

Run the same command family as above with `--max-length 4096`. The headline
retained variants were:

| CE env | Activation policy | XLA dump |
| --- | --- | --- |
| `TUNIX_ACCEL_DISABLE_CE=1` | `none` | `/tmp/xla_act_l4096_none` |
| `TUNIX_ACCEL_DISABLE_CE=1` | `split_offload` | `/tmp/xla_act_l4096_split_offload` |

A remat-only diagnostic can be reproduced by changing
`--activation-policy split_offload` to `--activation-policy split_remat`, but it
is not part of the final before/after comparison.

Example:

```bash
TUNIX_ACCEL_DISABLE_CE=1 \
TUNIX_ACCEL_DISABLE_TILED_MLP=1 \
XLA_FLAGS="--xla_dump_to=/tmp/xla_act_l4096_split_offload" \
python 04-ACTIVATION-POLICY/run_gemma_training_benchmark.py \
  --model-size 4b \
  --activation-policy split_offload \
  --batch-size 1 \
  --max-length 4096 \
  --max-steps 5 \
  --num-examples 64 \
  --lora-rank 16 \
  --mesh-fsdp 8 \
  --mesh-tp 1 \
  --skip-quality-eval \
  --outdir /tmp/activation-policy-04/context/l4096/split_offload
```

## 7. Reproduce Same-Model Parity

Purpose: compare baseline vs `split_offload` on one loaded model instance and
the same first batch.

```bash
TUNIX_ACCEL_DISABLE_AUTOPATCH=1 \
TUNIX_ACCEL_DISABLE_CE=1 \
TUNIX_ACCEL_DISABLE_TILED_MLP=1 \
python 04-ACTIVATION-POLICY/run_gemma3_activation_policy_parity.py \
  --model-size 4b \
  --activation-policy split_offload \
  --batch-size 1 \
  --max-length 512 \
  --num-examples 64 \
  --lora-rank 16 \
  --mesh-fsdp 8 \
  --mesh-tp 1 \
  --outdir /tmp/activation-policy-04/parity
```

Retain:

```text
/tmp/activation-policy-04/parity/4b/parity_summary.json
```

Final copy:

```text
04-ACTIVATION-POLICY/data/gemma3_4b_activation_policy_parity.json
```

## 8. Reproduce Small-Model Follow-Up

Purpose: test whether the same requested long-context stack also shows an
activation-offload boundary move on smaller TPU slices. The retained result is
diagnostic, not a clean Splash Attention proof, because long-context OOM logs
still exposed dense attention allocations.

Common settings:

| Field | Value |
| --- | --- |
| CCE | enabled |
| Tiled MLP | enabled |
| Splash Attention | requested |
| Batch | 1 |
| LoRA rank | 16 |
| Steps | 1 |
| Contexts | 8192, 16384, 32768 |
| Policies | `none`, `split_offload` |

270M on one v5e chip:

```bash
for L in 8192 16384 32768; do
  for POLICY in none split_offload; do
    TUNIX_ACCEL_DISABLE_CE=0 \
    TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION=1 \
    XLA_FLAGS="--xla_dump_to=/tmp/xla_act_270m_l${L}_${POLICY}" \
    python 04-ACTIVATION-POLICY/run_gemma_training_benchmark.py \
      --model-size 270m \
      --activation-policy "${POLICY}" \
      --enable-tiled-mlp \
      --enable-splash-attention \
      --batch-size 1 \
      --max-length "${L}" \
      --max-steps 1 \
      --num-examples 64 \
      --lora-rank 16 \
      --mesh-fsdp 1 \
      --mesh-tp 1 \
      --skip-quality-eval \
      --outdir "/tmp/activation-policy-04/small/270m/l${L}/${POLICY}"
  done
done
```

1B on four v5e chips:

```bash
for L in 8192 16384 32768; do
  for POLICY in none split_offload; do
    TUNIX_ACCEL_DISABLE_CE=0 \
    TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION=1 \
    XLA_FLAGS="--xla_dump_to=/tmp/xla_act_1b_l${L}_${POLICY}" \
    python 04-ACTIVATION-POLICY/run_gemma_training_benchmark.py \
      --model-size 1b \
      --activation-policy "${POLICY}" \
      --enable-tiled-mlp \
      --enable-splash-attention \
      --initialize-distributed \
      --batch-size 1 \
      --max-length "${L}" \
      --max-steps 1 \
      --num-examples 64 \
      --lora-rank 16 \
      --mesh-fsdp 4 \
      --mesh-tp 1 \
      --skip-quality-eval \
      --outdir "/tmp/activation-policy-04/small/1b/l${L}/${POLICY}"
  done
done
```

For each run, retain the run log, `status.txt`, `summary.json` when present,
`history.csv` when present, and the `*jit__train_step*memory-usage-report.txt`
file from the corresponding XLA dump.

Final retained copy:

```text
04-ACTIVATION-POLICY/results/small-model-splash-activation-ablation/
04-ACTIVATION-POLICY/data/gemma3_small_model_activation_followup.csv
04-ACTIVATION-POLICY/assets/gemma3_small_model_activation_followup.png
```

## 9. Regenerate Summary Figures

After copying the 4B raw retained files into `04-ACTIVATION-POLICY/data/raw/`,
run:

```bash
python 04-ACTIVATION-POLICY/analyze_activation_policy_results.py
```

This regenerates:

```text
04-ACTIVATION-POLICY/data/gemma3_4b_activation_policy_keypoints.csv
04-ACTIVATION-POLICY/data/MANIFEST.json
04-ACTIVATION-POLICY/assets/gemma3_4b_activation_hbm_headroom.png
04-ACTIVATION-POLICY/assets/gemma3_4b_activation_before_after_memory.png
04-ACTIVATION-POLICY/assets/gemma3_4b_l4096_activation_frontier.png
04-ACTIVATION-POLICY/assets/gemma3_4b_l2048_activation_tradeoff.png
```

After copying the 270M/1B raw follow-up directories into
`04-ACTIVATION-POLICY/results/small-model-splash-activation-ablation/`, run:

```bash
python 04-ACTIVATION-POLICY/analyze_small_model_activation_followup.py
```

This regenerates:

```text
04-ACTIVATION-POLICY/results/small-model-splash-activation-ablation/small_model_splash_activation_metrics.csv
04-ACTIVATION-POLICY/results/small-model-splash-activation-ablation/small_model_splash_activation_metrics.json
04-ACTIVATION-POLICY/data/gemma3_small_model_activation_followup.csv
04-ACTIVATION-POLICY/assets/gemma3_small_model_activation_followup.png
```
