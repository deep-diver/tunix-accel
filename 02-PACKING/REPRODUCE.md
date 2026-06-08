# Reproducing the 02-PACKING Rerun

This guide reproduces the retained 02-PACKING package: local packing-density
sweeps, Gemma3 270M Tunix LoRA SFT runs on Cloud TPU `v5litepod-1`, the
Gemma3 1B transfer check on `v5litepod-32`.

## Local Setup

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 -m pytest \
  tests/test_packing.py \
  tests/test_tunix_packing.py \
  tests/test_packing_model_parity.py \
  tests/test_autopatch.py \
  tests/test_tunix_gemma_packing_smoke.py
```

The packing API is inert unless `packing=` is supplied to
`PeftTrainer.with_gen_model_input_fn`.

## Local Density Sweeps

No-model token proxy:

```bash
python3 02-PACKING/run_efficiency_benchmark.py \
  --dataset opus100 \
  --num-examples 5000 \
  --batch-sizes 8,16,32,64 \
  --max-lengths 256,512,1024,2048 \
  --outdir 02-PACKING/results/local-no-model-5k

python3 02-PACKING/run_efficiency_benchmark.py \
  --dataset opus100 \
  --num-examples 20000 \
  --batch-sizes 8,16,32,64 \
  --max-lengths 256,512,1024,2048 \
  --outdir 02-PACKING/results/local-no-model-20k
```

Gemma tokenizer:

```bash
python3 02-PACKING/run_gemma_tokenizer_benchmark.py \
  --num-examples 5000 \
  --batch-sizes 8,16,32,64 \
  --max-lengths 256,512,1024,2048 \
  --outdir 02-PACKING/results/local-gemma-tokenizer-5k

python3 02-PACKING/run_gemma_tokenizer_benchmark.py \
  --num-examples 20000 \
  --batch-sizes 8,16,32,64 \
  --max-lengths 256,512,1024,2048 \
  --outdir 02-PACKING/results/local-gemma-tokenizer-20k
```

The retained copies live under `02-PACKING/data/local_density/`.

## Dataset and Max-Length Preflight

Before spending TPU time on a new SFT dataset, run the tokenizer-only profile.
It measures target-mask density, row reduction, retained target tokens, and
packing opportunity without loading Gemma weights:

```bash
python3 02-PACKING/run_dataset_profile_benchmark.py \
  --datasets opus100,alpaca,oasst1 \
  --num-examples 5000 \
  --batch-sizes 8,16,32 \
  --max-lengths 256,512,1024,2048,4096 \
  --outdir 02-PACKING/results/dataset-profile-270m \
  --allow-download
```

Regenerate the retained profile/TPU ablation figures after TPU artifacts are
collected:

```bash
python3 02-PACKING/aggregate_dataset_sweep.py
```

The 1B transfer figures are regenerated separately after the transfer raw
artifact is present:

```bash
python3 02-PACKING/aggregate_gemma3_1b_transfer.py
```

## TPU Setup

The 270M base rerun used:

| Field | Value |
| --- | --- |
| Project | `gcp-ml-172005` |
| Zone | `us-west4-a` |
| TPU type | `v5litepod-1` |
| Chips | 1 |
| TPU VM version | `v2-alpha-tpuv5-lite` |
| Model | `google/gemma-3-270m-it` |
| Model checkpoint | `gs://gemma-data/checkpoints/gemma3-270m-it` |
| Tokenizer | `gs://gemma-data/tokenizers/tokenizer_gemma3.model` |
| Mesh | `fsdp=1,tp=1` |

Create three TPU VMs if you want the same parallel layout:

```bash
for name in tunix-pack270-short tunix-pack270-unpack tunix-pack270-packed; do
  gcloud compute tpus tpu-vm create "$name" \
    --project=gcp-ml-172005 \
    --zone=us-west4-a \
    --accelerator-type=v5litepod-1 \
    --version=v2-alpha-tpuv5-lite
done
```

For the dataset/max-length ablation, use one `v5litepod-1` per dataset so the
three 50-step sweeps can run in parallel:

```bash
for name in tunix-pack270-opus tunix-pack270-alpaca tunix-pack270-oasst; do
  gcloud compute tpus tpu-vm create "$name" \
    --project=gcp-ml-172005 \
    --zone=us-west4-a \
    --accelerator-type=v5litepod-1 \
    --version=v2-alpha-tpuv5-lite
done
```

Copy the repository to each VM:

```bash
for name in tunix-pack270-short tunix-pack270-unpack tunix-pack270-packed; do
  gcloud compute tpus tpu-vm scp --recurse . "$name":~/TUNIX-TRY \
    --project=gcp-ml-172005 \
    --zone=us-west4-a
done
```

## TPU Runs

Short throughput and fit-frontier matrix:

```bash
gcloud compute tpus tpu-vm ssh tunix-pack270-short \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --command 'cd ~/TUNIX-TRY && bash 02-PACKING/remote_gemma3_270m_packing_worker.sh short-throughput'
```

Useful-token budget sanity runs:

```bash
gcloud compute tpus tpu-vm ssh tunix-pack270-unpack \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --command 'cd ~/TUNIX-TRY && GENERATION_EXAMPLES=16 bash 02-PACKING/remote_gemma3_270m_packing_worker.sh quality-unpacked'

gcloud compute tpus tpu-vm ssh tunix-pack270-packed \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --command 'cd ~/TUNIX-TRY && GENERATION_EXAMPLES=16 bash 02-PACKING/remote_gemma3_270m_packing_worker.sh quality-packed'
```

The worker installs Python 3.11 if needed, creates a venv, installs JAX TPU
wheels, disables unrelated acceleration patches, and writes outputs under:

```text
/tmp/gemma3-270m-packing/
```

Dataset/max-length short-throughput sweeps:

```bash
gcloud compute tpus tpu-vm ssh tunix-pack270-opus \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --command 'cd ~/TUNIX-TRY && DATASET_MODE=opus100 LONG_EXAMPLE_POLICY=truncate BATCH_SIZES=4,8,16 CONTEXTS=256,512,1024,2048 NUM_EXAMPLES=5000 MAX_STEPS=50 OUT_BASE=/tmp/gemma3-270m-packing-dataset/opus100 bash 02-PACKING/remote_gemma3_270m_packing_worker.sh short-throughput'

gcloud compute tpus tpu-vm ssh tunix-pack270-alpaca \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --command 'cd ~/TUNIX-TRY && DATASET_MODE=alpaca LONG_EXAMPLE_POLICY=truncate BATCH_SIZES=4,8,16 CONTEXTS=256,512,1024,2048 NUM_EXAMPLES=5000 MAX_STEPS=50 OUT_BASE=/tmp/gemma3-270m-packing-dataset/alpaca bash 02-PACKING/remote_gemma3_270m_packing_worker.sh short-throughput'

gcloud compute tpus tpu-vm ssh tunix-pack270-oasst \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --command 'cd ~/TUNIX-TRY && DATASET_MODE=oasst1 LONG_EXAMPLE_POLICY=truncate BATCH_SIZES=4,8,16 CONTEXTS=256,512,1024,2048 NUM_EXAMPLES=5000 MAX_STEPS=50 OUT_BASE=/tmp/gemma3-270m-packing-dataset/oasst1 bash 02-PACKING/remote_gemma3_270m_packing_worker.sh short-throughput'

```

In the retained rerun, OPUS100, Alpaca, and OASST1 were collected successfully.

## Gemma3 1B Transfer Run

The retained Gemma3 1B transfer check used a larger TPU because b8/L2048 did
not fit under the initial FSDP-only attempts:

| Field | Value |
| --- | --- |
| TPU type | `v5litepod-32` |
| Chips | 32 |
| TPU VM version | `v2-alpha-tpuv5-lite` |
| Model | `google/gemma-3-1b-it` |
| Model checkpoint | `gs://gemma-data/checkpoints/gemma3-1b-it` |
| Tokenizer | `gs://gemma-data/tokenizers/tokenizer_gemma3.model` |
| Mesh | `fsdp=8,tp=4` |
| Datasets | OPUS100 EN-FR, Alpaca, OASST1 EN |
| Shapes | b4/b8 by L512/L1024/L2048 |
| Steps | 50 LoRA SFT steps per case |

Create the TPU and copy the repository:

```bash
gcloud compute tpus tpu-vm create tunix-pack1b-32 \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --accelerator-type=v5litepod-32 \
  --version=v2-alpha-tpuv5-lite

gcloud compute tpus tpu-vm scp --recurse . tunix-pack1b-32:~/TUNIX-TRY \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --worker=all
```

Run the three dataset sweeps. In practice, SSH to each worker can be launched
with `setsid -f` or a job scheduler so the 32-chip distributed job stays
attached to the TPU worker set; the important environment is:

```bash
for ds in opus100 alpaca oasst1; do
  gcloud compute tpus tpu-vm ssh tunix-pack1b-32 \
    --project=gcp-ml-172005 \
    --zone=us-west4-a \
    --worker=all \
    --command "cd ~/TUNIX-TRY && \
      SKIP_INSTALL=1 \
      MODEL_ID=google/gemma-3-1b-it \
      MODEL_SOURCE=gcs \
      MODEL_PATH=gs://gemma-data/checkpoints/gemma3-1b-it \
      TOKENIZER_SOURCE=sentencepiece \
      TOKENIZER_PATH=gs://gemma-data/tokenizers/tokenizer_gemma3.model \
      TPU_TYPE=v5litepod-32 \
      CHIPS=32 \
      MESH_FSDP=8 \
      MESH_TP=4 \
      INITIALIZE_DISTRIBUTED=1 \
      DATASET_MODE=${ds} \
      LONG_EXAMPLE_POLICY=truncate \
      BATCH_SIZES=4,8 \
      CONTEXTS=512,1024,2048 \
      NUM_EXAMPLES=5000 \
      MAX_STEPS=50 \
      OUT_BASE=/tmp/packing-transfer/gemma3-1b-transfer32-tp4/${ds} \
      bash 02-PACKING/remote_gemma3_270m_packing_worker.sh short-throughput"
done
```

Collect the worker-0 artifact:

```bash
gcloud compute tpus tpu-vm ssh tunix-pack1b-32 \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --worker=0 \
  --command 'cd /tmp/packing-transfer && tar -czf /tmp/gemma3_1b_transfer32_tp4_results.tar.gz gemma3-1b-transfer32-tp4'

mkdir -p 02-PACKING/data/transfer_1b/raw
gcloud compute tpus tpu-vm scp \
  tunix-pack1b-32:/tmp/gemma3_1b_transfer32_tp4_results.tar.gz \
  02-PACKING/data/transfer_1b/raw/ \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --worker=0
```

Then regenerate locally. The aggregator extracts the tarball if the expanded
raw directory is absent:

```bash
python3 02-PACKING/aggregate_gemma3_1b_transfer.py
```

## Collect Artifacts

Compress remote outputs:

```bash
gcloud compute tpus tpu-vm ssh tunix-pack270-short \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --command 'cd /tmp/gemma3-270m-packing && tar -czf /tmp/gemma3_270m_short_throughput_v5litepod1.tgz short-throughput'

gcloud compute tpus tpu-vm ssh tunix-pack270-unpack \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --command 'cd /tmp/gemma3-270m-packing && tar -czf /tmp/gemma3_270m_quality_unpacked_v5litepod1.tgz quality-unpacked'

gcloud compute tpus tpu-vm ssh tunix-pack270-packed \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --command 'cd /tmp/gemma3-270m-packing && tar -czf /tmp/gemma3_270m_quality_packed_v5litepod1.tgz quality-packed'
```

Copy tarballs back into:

```text
02-PACKING/data/raw_artifacts/gemma3_270m_short_throughput_v5litepod1/
02-PACKING/data/raw_artifacts/gemma3_270m_quality_unpacked_v5litepod1/
02-PACKING/data/raw_artifacts/gemma3_270m_quality_packed_v5litepod1/
```

Dataset sweep tarballs use:

```text
02-PACKING/data/raw_artifacts/gemma3_270m_dataset_sweep_opus100_v5litepod1/
02-PACKING/data/raw_artifacts/gemma3_270m_dataset_sweep_alpaca_v5litepod1/
02-PACKING/data/raw_artifacts/gemma3_270m_dataset_sweep_oasst1_v5litepod1/
```

Then regenerate processed tables and plots:

```bash
python3 02-PACKING/visualize_270m_results.py
python3 02-PACKING/aggregate_dataset_sweep.py
```

The visualizer extracts the raw `.tgz` files if the raw directories are not
present.

## Cleanup

Delete TPU VMs after copying artifacts:

```bash
for name in tunix-pack270-short tunix-pack270-unpack tunix-pack270-packed; do
  gcloud compute tpus tpu-vm delete "$name" \
    --project=gcp-ml-172005 \
    --zone=us-west4-a \
    --quiet
done

gcloud compute tpus tpu-vm delete tunix-pack1b-32 \
  --project=gcp-ml-172005 \
  --zone=us-west4-a \
  --quiet

```
