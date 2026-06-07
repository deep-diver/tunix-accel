# Reproducing the Gemma CCE Rerun and Transfer Checks

This guide reproduces the final 01-CCE package: the exhaustive Gemma3 270M CCE
rerun, the four-chip Gemma3 1B / Gemma4 E2B transfer checks, and the focused
eight-chip Gemma3 4B / Gemma4 E4B transfer checks. It intentionally keeps
Packing, Tiled MLP, Activation Policy, and Splash Attention disabled.

## Retained Artifacts

Main report and figures:

- `01-CCE/TECHNICAL_REPORT.md`
- `01-CCE/assets/gemma3_270m_cce_frontier.png`
- `01-CCE/assets/gemma3_270m_cce_status_heatmap.png`
- `01-CCE/assets/gemma3_270m_cce_tuning.png`
- `01-CCE/assets/gemma3_270m_cce_quality.png`
- `01-CCE/assets/gemma3_270m_cce_mesh_2x2_repeat.png`
- `01-CCE/assets/gemma3_270m_cce_4chip_frontier.png`
- `01-CCE/assets/gemma3_270m_cce_outlier_hlo.png`
- `01-CCE/assets/gemma3_270m_cce_4chip_chunk_tuning.png`
- `01-CCE/assets/gemma3_270m_cce_4chip_chunk_axis_ablation.png`
- `01-CCE/assets/gemma3_270m_cce_4chip_quality.png`
- `01-CCE/assets/gemma_cce_transfer_frontier.png`
- `01-CCE/assets/gemma_cce_transfer_quality.png`
- `01-CCE/assets/gemma_cce_transfer_chunk_mesh.png`
- `01-CCE/assets/gemma_cce_large_transfer_frontier.png`
- `01-CCE/assets/gemma_cce_large_transfer_pressure.png`
- `01-CCE/assets/gemma_cce_large_transfer_chunk_tuning.png`

Compact rerun data:

- `01-CCE/data/gemma3_270m_full_cce/run_manifest.csv`
- `01-CCE/data/gemma3_270m_full_cce/all_runs.csv`
- `01-CCE/data/gemma3_270m_full_cce/frontier_runs.csv`
- `01-CCE/data/gemma3_270m_full_cce/frontier_summary.csv`
- `01-CCE/data/gemma3_270m_full_cce/pressure_points.csv`
- `01-CCE/data/gemma3_270m_full_cce/rank_sensitivity.csv`
- `01-CCE/data/gemma3_270m_full_cce/rank_frontier_summary.csv`
- `01-CCE/data/gemma3_270m_full_cce/chunk_tuning.csv`
- `01-CCE/data/gemma3_270m_full_cce/training_history.csv`
- `01-CCE/data/gemma3_270m_full_cce/training_summary.csv`
- `01-CCE/data/gemma3_270m_full_cce/generation_metrics.csv`
- `01-CCE/data/gemma3_270m_full_cce/generation_samples.jsonl`
- `01-CCE/data/gemma3_270m_full_cce/profile_summary.csv`
- `01-CCE/data/gemma3_270m_full_cce/oom_events.csv`
- `01-CCE/data/gemma3_270m_mesh_cce/run_manifest.csv`
- `01-CCE/data/gemma3_270m_mesh_cce/mesh_runs.csv`
- `01-CCE/data/gemma3_270m_mesh_cce/mesh_summary.csv`
- `01-CCE/data/gemma3_270m_mesh_cce/matched_memory.csv`
- `01-CCE/data/gemma3_270m_mesh_cce_repeat/repeat_summary.csv`
- `01-CCE/data/gemma3_270m_4chip_frontier/frontier_summary.csv`
- `01-CCE/data/gemma3_270m_outlier_hlo/hlo_op_counts.csv`
- `01-CCE/data/gemma3_270m_4chip_chunk/chunk_summary.csv`
- `01-CCE/data/gemma3_270m_4chip_chunk/chunk_axis_ablation.csv`
- `01-CCE/data/gemma3_270m_4chip_quality/training_summary.csv`
- `01-CCE/data/gemma_1b_e2b_cce_transfer/frontier_summary.csv`
- `01-CCE/data/gemma_1b_e2b_cce_transfer/matched_metrics.csv`
- `01-CCE/data/gemma_1b_e2b_cce_transfer/chunk_summary.csv`
- `01-CCE/data/gemma_1b_e2b_cce_transfer/training_summary.csv`
- `01-CCE/data/gemma_1b_e2b_cce_transfer/training_history.csv`
- `01-CCE/data/gemma_4b_e4b_cce_transfer/frontier_summary.csv`
- `01-CCE/data/gemma_4b_e4b_cce_transfer/matched_metrics.csv`
- `01-CCE/data/gemma_4b_e4b_cce_transfer/pressure_points.csv`
- `01-CCE/data/gemma_4b_e4b_cce_transfer/chunk_summary.csv`

Compressed raw worker outputs:

- `01-CCE/data/gemma3_270m_full_cce/raw_artifacts/*.tar.gz`
- `01-CCE/data/gemma3_270m_mesh_cce/raw_artifacts/*.tar.gz`
- `01-CCE/data/gemma3_270m_mesh_cce_repeat/raw_artifacts/*.tar.gz`
- `01-CCE/data/gemma3_270m_4chip_frontier/raw_artifacts/*.tar.gz`
- `01-CCE/data/gemma3_270m_4chip_chunk/raw_artifacts/*.tar.gz`
- `01-CCE/data/gemma3_270m_4chip_quality/raw_artifacts/*.tar.gz`
- `01-CCE/data/gemma_1b_e2b_cce_transfer/raw_artifacts/*.tar.gz`
- `01-CCE/data/gemma_4b_e4b_cce_transfer/raw_artifacts/*.tar.gz`

Do not commit extracted `raw/` directories. They are recreated by the collector
scripts from the tarballs.

## Local Patch Install

Use the same package controls as the rest of the repository:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Default CE baseline:

```bash
export TUNIX_ACCEL_DISABLE_AUTOPATCH=1
```

CCE run:

```bash
unset TUNIX_ACCEL_DISABLE_AUTOPATCH
export TUNIX_ACCEL_DISABLE_CE=0
export TUNIX_ACCEL_CE_TOKEN_CHUNK=128
export TUNIX_ACCEL_CE_VOCAB_CHUNK=8192
```

The conservative rerun default is `128/8192`. The four-chip mixed-mesh follow-up
also validated the convenience preset below, which maps to `512/65536` unless
explicit chunk variables override it:

```bash
export TUNIX_ACCEL_CE_PRESET=tpu_large_chunks
```

The rerun worker always disables the unrelated patches:

```bash
export TUNIX_ACCEL_DISABLE_TILED_MLP=1
export TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY=1
export TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION=0
```

## TPU Setup

The primary rerun artifacts were produced on:

| Field | Value |
| --- | --- |
| Project | `gcp-ml-172005` |
| Zone | `us-west4-a` |
| TPU type | `v5litepod-1` |
| Chips | 1 |
| VM image | `tpu-ubuntu2204-base` |
| Model | `google/gemma-3-270m-it` |
| Model checkpoint | `gs://gemma-data/checkpoints/gemma3-270m-it` |
| Tokenizer | `gs://gemma-data/tokenizers/tokenizer_gemma3.model` |

The mesh generalization check used the same project, zone, model, checkpoint,
tokenizer, and image, but ran on Cloud TPU `v5litepod-4` with four chips.
Gemma3 1B and Gemma4 E2B transfer checks also used `v5litepod-4`, four chips,
with `fsdp=4,tp=1` as the primary mesh.
Gemma3 4B and Gemma4 E4B focused transfer checks used `v5litepod-8`, eight
chips, with `fsdp=8,tp=1`.

Create one TPU VM per independent profile when you want maximum parallelism:

```bash
gcloud compute tpus tpu-vm create TPU_NAME \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-1 \
  --version=tpu-ubuntu2204-base
```

Copy the repository to the TPU VM and run the desired profile:

```bash
gcloud compute tpus tpu-vm scp --recurse . TPU_NAME:~/TUNIX-TRY \
  --project=gcp-ml-172005 \
  --zone=us-west4-a

gcloud compute tpus tpu-vm ssh TPU_NAME \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --command 'cd ~/TUNIX-TRY && bash 01-CCE/remote_gemma3_270m_cce_worker.sh frontier-low'
```

The remote worker installs Python 3.11 if needed, creates a venv, installs
JAX TPU wheels, runs the profile, trims XLA dumps to memory reports, and emits a
compressed artifact:

```text
/tmp/gemma3-270m-cce-rerun-PROFILE.tar.gz
```

Copy that tarball back into:

```text
01-CCE/data/gemma3_270m_full_cce/raw_artifacts/
```

Delete the TPU VM after copying the artifact.

## Profiles

The final rerun used these worker profiles:

| Profile | Purpose |
| --- | --- |
| `parity` | pytest parity checks plus one-step Gemma3 270M b1/b4 parity rows |
| `frontier-low` | b1/b2/b4/b8 context sweep through L32768 |
| `frontier-high` | b16/b32/b64/b128 context sweep through L32768 |
| `rank` | rank 4/16/64 sensitivity over b8/b16/b32/b64 and L512-L4096 |
| `chunk` | CCE chunk tuning plus representative pressure points |
| `quality-default` | OPUS100 EN-FR Default CE b16/L512 for 5,000 steps |
| `quality-cce` | OPUS100 EN-FR CCE b16/L512 for 5,000 steps |
| `quality-capacity` | OPUS100 EN-FR CCE b64/L512 for 1,250 steps |
| `mesh-fsdp4` | v5litepod-4 synthetic mesh check with `fsdp=4,tp=1` |
| `mesh-2x2` | v5litepod-4 synthetic mesh check with `fsdp=2,tp=2` |
| `mesh-tp4` | v5litepod-4 synthetic mesh check with `fsdp=1,tp=4` |
| `mesh-2x2-repeat` | repeated b16/b32 timing check for the mixed mesh |
| `mesh-repeat-rest` | repeated timing checks for `fsdp=4,tp=1` and `fsdp=1,tp=4` |
| `fourchip-frontier-fsdp4` | extended four-chip frontier for `fsdp=4,tp=1` |
| `fourchip-frontier-2x2` | extended four-chip frontier for `fsdp=2,tp=2` |
| `fourchip-frontier-tp4` | extended four-chip frontier for `fsdp=1,tp=4` |
| `outlier-hlo` | full XLA dump for the mixed-mesh outlier scan |
| `fourchip-chunk-2x2` | CCE chunk tuning for `fsdp=2,tp=2`, b16/L512 |
| `pilot-fsdp4` | small four-chip model-size pilot used before larger transfer sweeps |
| `fourchip-chunk-fsdp4` | CCE chunk tuning for `fsdp=4,tp=1`, b16/L512 |
| `fourchip-quality-fsdp4-default` | OPUS100 1,000-step four-chip Default CE parity row |
| `fourchip-quality-fsdp4-cce` | OPUS100 1,000-step four-chip CCE parity row |

Example parallel schedule:

```bash
bash 01-CCE/remote_gemma3_270m_cce_worker.sh parity
bash 01-CCE/remote_gemma3_270m_cce_worker.sh frontier-low
bash 01-CCE/remote_gemma3_270m_cce_worker.sh frontier-high
bash 01-CCE/remote_gemma3_270m_cce_worker.sh rank
bash 01-CCE/remote_gemma3_270m_cce_worker.sh chunk
bash 01-CCE/remote_gemma3_270m_cce_worker.sh quality-default
bash 01-CCE/remote_gemma3_270m_cce_worker.sh quality-cce
bash 01-CCE/remote_gemma3_270m_cce_worker.sh quality-capacity
```

Run the mesh profiles on `v5litepod-4` workers:

```bash
bash 01-CCE/remote_gemma3_270m_cce_worker.sh mesh-fsdp4
bash 01-CCE/remote_gemma3_270m_cce_worker.sh mesh-2x2
bash 01-CCE/remote_gemma3_270m_cce_worker.sh mesh-tp4
bash 01-CCE/remote_gemma3_270m_cce_worker.sh mesh-2x2-repeat
bash 01-CCE/remote_gemma3_270m_cce_worker.sh mesh-repeat-rest
bash 01-CCE/remote_gemma3_270m_cce_worker.sh fourchip-frontier-fsdp4
bash 01-CCE/remote_gemma3_270m_cce_worker.sh fourchip-frontier-2x2
bash 01-CCE/remote_gemma3_270m_cce_worker.sh fourchip-frontier-tp4
bash 01-CCE/remote_gemma3_270m_cce_worker.sh outlier-hlo
bash 01-CCE/remote_gemma3_270m_cce_worker.sh fourchip-chunk-2x2
bash 01-CCE/remote_gemma3_270m_cce_worker.sh fourchip-quality-fsdp4-default
bash 01-CCE/remote_gemma3_270m_cce_worker.sh fourchip-quality-fsdp4-cce
```

For the transfer checks, reuse the same worker and set `MODEL_SIZE`:

```bash
MODEL_SIZE=1b OUT_BASE=/tmp/gemma-cce-1b-exhaustive \
  bash 01-CCE/remote_gemma3_270m_cce_worker.sh fourchip-frontier-fsdp4

MODEL_SIZE=1b OUT_BASE=/tmp/gemma-cce-1b-exhaustive \
  bash 01-CCE/remote_gemma3_270m_cce_worker.sh fourchip-frontier-2x2

MODEL_SIZE=1b OUT_BASE=/tmp/gemma-cce-1b-followup2 \
  bash 01-CCE/remote_gemma3_270m_cce_worker.sh fourchip-chunk-fsdp4

MODEL_SIZE=e2b OUT_BASE=/tmp/gemma-cce-e2b-exhaustive \
  bash 01-CCE/remote_gemma3_270m_cce_worker.sh fourchip-frontier-fsdp4

MODEL_SIZE=e2b OUT_BASE=/tmp/gemma-cce-e2b-chunk \
  bash 01-CCE/remote_gemma3_270m_cce_worker.sh fourchip-chunk-fsdp4
```

Gemma4 E2B uses Hugging Face model loading. Keep a valid Hugging Face token in
the TPU VM cache or environment and reuse a shared model cache, for example:

```bash
export TUNIX_ACCEL_MODEL_DOWNLOAD_PATH=/tmp/gemma-cce-e2b-exhaustive/hf-cache/e2b
```

The matched OPUS100 transfer rows used direct `run_gemma3_270m_cce_sweep.py`
commands because the matched safe shapes differ by model: Gemma3 1B used
b8/L512, and Gemma4 E2B used b4/L256.

For the focused eight-chip 4B/E4B checks, use the new worker profiles:

```bash
MODEL_SIZE=4b OUT_BASE=/tmp/gemma-cce-4b-focused \
  bash 01-CCE/remote_gemma3_270m_cce_worker.sh eightchip-frontier-fsdp8

MODEL_SIZE=4b OUT_BASE=/tmp/gemma-cce-4b-focused \
  CHUNK_BATCHES=4 CHUNK_CONTEXTS=1024 \
  bash 01-CCE/remote_gemma3_270m_cce_worker.sh eightchip-chunk-fsdp8

MODEL_SIZE=e4b OUT_BASE=/tmp/gemma-cce-e4b-focused-hf3 \
  TUNIX_ACCEL_MODEL_DOWNLOAD_PATH=/tmp/gemma-cce-e4b-focused-hf3/hf-cache/e4b \
  FOCUSED_BATCHES=1,2,4,8 FOCUSED_CONTEXTS=256,512,1024,2048 \
  bash 01-CCE/remote_gemma3_270m_cce_worker.sh eightchip-frontier-fsdp8

MODEL_SIZE=e4b OUT_BASE=/tmp/gemma-cce-e4b-focused-hf3 \
  TUNIX_ACCEL_MODEL_DOWNLOAD_PATH=/tmp/gemma-cce-e4b-focused-hf3/hf-cache/e4b \
  CHUNK_BATCHES=4 CHUNK_CONTEXTS=512 \
  bash 01-CCE/remote_gemma3_270m_cce_worker.sh eightchip-chunk-fsdp8
```

Gemma4 E4B uses Hugging Face loading in this package. Keep the Hugging Face
token in `~/.cache/huggingface/token` or `HF_TOKEN`, and do not include the HF
model cache in copied artifacts.

For the focused eight-chip Gemma3 12B/27B boundary checks, use the generic
large-chip profile:

```bash
MODEL_SIZE=12b OUT_BASE=/tmp/gemma-cce-12b-focused \
  LARGE_FSDP=8 LARGE_TP=1 LARGE_TPU=v5litepod-8 LARGE_CHIPS=8 \
  FOCUSED_BATCHES=1,2,4 FOCUSED_CONTEXTS=512,1024,2048,4096 \
  FOCUSED_MAX_STEPS=2 \
  bash 01-CCE/remote_gemma3_270m_cce_worker.sh large-frontier-fsdp

MODEL_SIZE=27b OUT_BASE=/tmp/gemma-cce-27b-focused-x8 \
  LARGE_FSDP=8 LARGE_TP=1 LARGE_TPU=v5litepod-8 LARGE_CHIPS=8 \
  FOCUSED_BATCHES=1,2,4 FOCUSED_CONTEXTS=512,1024,2048 \
  FOCUSED_MAX_STEPS=2 \
  bash 01-CCE/remote_gemma3_270m_cce_worker.sh large-frontier-fsdp
```

For actual multi-host Tunix smoke tests, launch the worker command on all TPU VM
hosts and pass `INITIALIZE_DISTRIBUTED=1`. The smoke rows retained in this
package used batch 1, context 512, rank 16, synthetic data, and two steps:

```bash
gcloud compute tpus tpu-vm ssh TPU_12B_X16 \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --worker=all \
  --command 'cd ~/TUNIX-TRY && MODEL_SIZE=12b OUT_BASE=/tmp/tunix-dist-12b-smoke LARGE_FSDP=16 LARGE_TP=1 LARGE_TPU=v5litepod-16 LARGE_CHIPS=16 INITIALIZE_DISTRIBUTED=1 FOCUSED_VARIANTS=default,cce FOCUSED_BATCHES=1 FOCUSED_CONTEXTS=512 FOCUSED_MAX_STEPS=2 FOCUSED_NUM_EXAMPLES=128 bash 01-CCE/remote_gemma3_270m_cce_worker.sh large-frontier-fsdp'

gcloud compute tpus tpu-vm ssh TPU_27B_X32 \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --worker=all \
  --command 'cd ~/TUNIX-TRY && MODEL_SIZE=27b OUT_BASE=/tmp/tunix-dist-27b-smoke LARGE_FSDP=32 LARGE_TP=1 LARGE_TPU=v5litepod-32 LARGE_CHIPS=32 INITIALIZE_DISTRIBUTED=1 FOCUSED_VARIANTS=default,cce FOCUSED_BATCHES=1 FOCUSED_CONTEXTS=512 FOCUSED_MAX_STEPS=2 FOCUSED_NUM_EXAMPLES=128 bash 01-CCE/remote_gemma3_270m_cce_worker.sh large-frontier-fsdp'
```

The logs should include `jax_distributed_initialized=...`; the retained smoke
rows confirmed `process_count=4`, `global_devices=16` for 12B and
`process_count=8`, `global_devices=32` for 27B.

## Local Aggregation

After all tarballs are copied into `raw_artifacts/`, run:

```bash
python3 01-CCE/collect_gemma3_270m_cce_results.py
python3 01-CCE/collect_gemma3_270m_mesh_results.py
python3 01-CCE/collect_gemma3_270m_mesh_repeat_results.py
python3 01-CCE/collect_gemma3_270m_4chip_frontier_results.py
python3 01-CCE/collect_gemma3_270m_outlier_hlo_results.py
python3 01-CCE/collect_gemma3_270m_4chip_chunk_results.py
python3 01-CCE/collect_gemma3_270m_4chip_quality_results.py
python3 01-CCE/collect_gemma_1b_e2b_cce_transfer_results.py
python3 01-CCE/collect_gemma_4b_e4b_cce_transfer_results.py
python3 01-CCE/collect_gemma3_12b_27b_cce_focused_results.py
```

The collectors:

1. extract the tarballs into the relevant disposable `raw/` directories,
2. rebuild compact CSV/JSONL summaries,
3. redraw the report figures.

The extracted `raw/` tree is intentionally disposable. Remove it before
committing:

```bash
rm -rf 01-CCE/data/gemma3_270m_full_cce/raw
rm -rf 01-CCE/data/gemma3_270m_mesh_cce/raw
rm -rf 01-CCE/data/gemma3_270m_mesh_cce_repeat/raw
rm -rf 01-CCE/data/gemma3_270m_4chip_frontier/raw
rm -rf 01-CCE/data/gemma3_270m_4chip_chunk/raw
rm -rf 01-CCE/data/gemma3_270m_4chip_quality/raw
rm -rf 01-CCE/data/gemma_1b_e2b_cce_transfer/raw
rm -rf 01-CCE/data/gemma_4b_e4b_cce_transfer/raw
rm -rf 01-CCE/data/gemma3_12b_27b_cce_focused/raw
```

## Expected Checks

The rerun should reproduce the following qualitative findings:

- b16/L512: Default CE and CCE both fit; XLA planned HBM drops from about
  12.57 GiB/chip to 4.98 GiB/chip.
- b16/L1024: Default CE compile OOMs; CCE completes at about 9.65 GiB/chip.
- b32/L512 and b32/L1024: Default CE compile OOMs; CCE completes.
- b64/L512: Default CE compile OOMs; CCE completes at about 14.13 GiB/chip.
- OPUS100 b16/L512 5,000-step training keeps train/eval loss in the same band,
  while same-shape CCE steps are slower.
- On `v5litepod-4`, CCE works across `fsdp=4,tp=1`, `fsdp=2,tp=2`, and
  `fsdp=1,tp=4`; matched passing rows show about 53-66% per-chip XLA planned
  HBM reduction.
- The repeated `fsdp=2,tp=2` default chunk row is a throughput outlier, but
  larger TPU chunk settings reduce b16/L512 from about 15.4s/step to about
  0.83s/step at the same 2.65 GiB/chip XLA HBM.
- On `v5litepod-8`, Gemma3 4B b4/L1024 fails under Default CE at 16.54
  GiB/chip planned HBM and completes under CCE at 15.04 GiB/chip.
- On `v5litepod-8`, Gemma4 E4B b4/L512 and b8/L256 are CCE-only fits in the
  focused frontier grid.
- On `v5litepod-8`, Gemma3 12B and 27B still show lower CCE planned HBM, but
  CCE does not create a new passing shape in the retained focused boundary
  grid. For example, Gemma3 12B b4/L512 drops from 19.01 to 17.31 GiB/chip and
  still fails; Gemma3 27B b2/L512 drops from 24.38 to 23.62 GiB/chip and still
  fails.

Generation note: the CCE LoRA training hook intercepts hidden states in the
loss path. Before sampling, call the restore path so Tunix generation sees the
normal LM-head decode again. The worker and tests already do this.
