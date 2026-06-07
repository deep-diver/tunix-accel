# Reproducing the Gemma3 270M Packing Rerun

This guide reproduces the retained 02-PACKING package: local packing-density
sweeps and Gemma3 270M Tunix LoRA SFT runs on Cloud TPU `v5litepod-1`.

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

## TPU Setup

The rerun used:

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

Then regenerate processed tables and plots:

```bash
python3 02-PACKING/visualize_270m_results.py
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
```
