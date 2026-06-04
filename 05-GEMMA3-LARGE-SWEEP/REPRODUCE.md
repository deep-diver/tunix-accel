# Reproduce the Gemma3 Large Patch Sweep

This guide reproduces the corrected Gemma3 12B/27B large-model patch sweep. It
assumes the repository has been uploaded to a Cloud TPU VM and installed with
`python -m pip install -e .`.

## Important Correctness Check

Patched variants must run with:

```bash
export TUNIX_ACCEL_DISABLE_AUTOPATCH=0
```

The sweep runner handles this automatically. Do not reproduce the earlier
invalid behavior where the variable was absent; the training benchmark would set
it back to `1`, disabling all autopatches.

The corrected benchmark records patch status fields in each `summary.json`:

```text
accel.cce_installed
accel.gemma3_tiled_mlp_installed
accel.gemma3_activation_policy_installed
accel.gemma3_splash_attention_installed
```

## TPU Shapes

Create the 12B TPU:

```bash
gcloud compute tpus tpu-vm create tunix-gemma3-12b-rerun-v5e4 \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-4 \
  --version=tpu-ubuntu2204-base
```

Create the 27B TPU:

```bash
gcloud compute tpus tpu-vm create tunix-gemma3-27b-rerun-v5e8 \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-8 \
  --version=tpu-ubuntu2204-base
```

If quota is limited, run these sequentially. The corrected run used 4 chips for
12B and 8 chips for 27B.

## Install On The TPU VM

```bash
python -m pip install google-tunix==0.1.6 qwix==0.1.6 kagglehub==0.4.3 \
  importlib_resources tensorflow datasets gcsfs==2026.2.0 matplotlib \
  sacrebleu transformers pytest
python -m pip install -U "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install -e .
```

Verify TPU visibility:

```bash
python - <<'PY'
import jax
print(jax.__version__)
print(jax.devices())
PY
```

## Run 12B

```bash
python tools/run_gemma3_large_patch_sweep.py \
  --model-size 12b \
  --variants default,cce,tiled_mlp,split_offload,splash,stacked \
  --contexts 512,1024,2048 \
  --extra-stacked-contexts 4096,8192 \
  --batch-sizes 1 \
  --num-examples 64 \
  --max-steps 2 \
  --outdir /tmp/gemma3-large-patch-sweep-corrected \
  --force
```

## Run 27B

```bash
python tools/run_gemma3_large_patch_sweep.py \
  --model-size 27b \
  --variants default,cce,tiled_mlp,split_offload,splash,stacked \
  --contexts 512,1024 \
  --extra-stacked-contexts 2048,4096 \
  --batch-sizes 1 \
  --num-examples 64 \
  --max-steps 2 \
  --outdir /tmp/gemma3-large-patch-sweep-corrected \
  --force
```

## Collect Lightweight Summaries

From the TPU VM:

```bash
cd /tmp/gemma3-large-patch-sweep-corrected
find 12b -maxdepth 2 -name case_summary.json | sort > /tmp/12b-light-files.txt
for f in 12b/sweep_results.csv 12b/sweep_results.json; do test -f "$f" && echo "$f" >> /tmp/12b-light-files.txt; done
tar -czf /tmp/gemma3-large-patch-sweep-corrected-12b-light.tar.gz -T /tmp/12b-light-files.txt

find 27b -maxdepth 2 -name case_summary.json | sort > /tmp/27b-light-files.txt
for f in 27b/sweep_results.csv 27b/sweep_results.json; do test -f "$f" && echo "$f" >> /tmp/27b-light-files.txt; done
tar -czf /tmp/gemma3-large-patch-sweep-corrected-27b-light.tar.gz -T /tmp/27b-light-files.txt
```

Extract locally into:

```text
05-GEMMA3-LARGE-SWEEP/raw/12b_corrected/
05-GEMMA3-LARGE-SWEEP/raw/27b_corrected/
```

## Store Full Raw Archives In GCS

The full raw directories contain XLA dumps and are intentionally not committed.

```bash
cd /tmp/gemma3-large-patch-sweep-corrected
tar -czf /tmp/gemma3-large-patch-sweep-corrected-12b.tar.gz 12b
tar -czf /tmp/gemma3-large-patch-sweep-corrected-27b.tar.gz 27b

gcloud storage cp /tmp/gemma3-large-patch-sweep-corrected-12b.tar.gz \
  gs://gcp-ml-172005-ddpm-training/tunix-large-sweep/gemma3-large-patch-sweep-corrected-12b.tar.gz
gcloud storage cp /tmp/gemma3-large-patch-sweep-corrected-27b.tar.gz \
  gs://gcp-ml-172005-ddpm-training/tunix-large-sweep/gemma3-large-patch-sweep-corrected-27b.tar.gz
```

## Regenerate Figures

After the lightweight summaries are in place:

```bash
python 05-GEMMA3-LARGE-SWEEP/make_figures.py
```

This regenerates:

```text
05-GEMMA3-LARGE-SWEEP/data/gemma3_large_patch_sweep_corrected_summary.csv
05-GEMMA3-LARGE-SWEEP/assets/gemma3_large_l512_patch_impact.png
05-GEMMA3-LARGE-SWEEP/assets/gemma3_large_context_frontier.png
05-GEMMA3-LARGE-SWEEP/assets/gemma3_large_practical_readout.png
05-GEMMA3-LARGE-SWEEP/assets/gemma3_large_l1024_fit_line.png
05-GEMMA3-LARGE-SWEEP/assets/gemma3_large_frontier_vs_time.png
05-GEMMA3-LARGE-SWEEP/assets/gemma3_large_oom_gap.png
```

## Cleanup

Do not leave TPU VMs running after collection:

```bash
gcloud compute tpus tpu-vm delete TPU_NAME \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --quiet
```
