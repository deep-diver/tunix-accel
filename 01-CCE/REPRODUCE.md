# Reproducing the Gemma3 270M CCE Rerun

This guide reproduces the final 01-CCE rerun: Gemma3 270M LoRA SFT on TPU,
Default CE versus Cut Cross Entropy only. It intentionally keeps Packing, Tiled
MLP, Activation Policy, and Splash Attention disabled.

## Retained Artifacts

Main report and figures:

- `01-CCE/TECHNICAL_REPORT.md`
- `01-CCE/assets/gemma3_270m_cce_frontier.png`
- `01-CCE/assets/gemma3_270m_cce_status_heatmap.png`
- `01-CCE/assets/gemma3_270m_cce_tuning.png`
- `01-CCE/assets/gemma3_270m_cce_quality.png`

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

Compressed raw worker outputs:

- `01-CCE/data/gemma3_270m_full_cce/raw_artifacts/*.tar.gz`

Do not commit the extracted `01-CCE/data/gemma3_270m_full_cce/raw/` directory.
It is recreated by the collector script from the tarballs.

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

The rerun worker always disables the unrelated patches:

```bash
export TUNIX_ACCEL_DISABLE_TILED_MLP=1
export TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY=1
export TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION=0
```

## TPU Setup

All rerun artifacts were produced on:

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

## Local Aggregation

After all tarballs are copied into `raw_artifacts/`, run:

```bash
python3 01-CCE/collect_gemma3_270m_cce_results.py
```

The collector:

1. extracts the tarballs into `data/gemma3_270m_full_cce/raw/`,
2. rebuilds compact CSV/JSONL summaries,
3. redraws the four report figures.

The extracted `raw/` tree is intentionally disposable. Remove it before
committing:

```bash
rm -rf 01-CCE/data/gemma3_270m_full_cce/raw
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

Generation note: the CCE LoRA training hook intercepts hidden states in the
loss path. Before sampling, call the restore path so Tunix generation sees the
normal LM-head decode again. The worker and tests already do this.
