# Gemma3 12B LoRA TPU Probe

This probe checks whether the Tunix-registered `google/gemma-3-12b-it` model
can load, shard, compile, and run LoRA SFT on TPU without the repository's
acceleration patches.

## Result

Gemma3 12B LoRA SFT completed a 2-step smoke run on Cloud TPU v5e
`v5litepod-4`.

The important point: `tp=1` did not mean data-parallel replication. With
`fsdp=4,tp=1`, Tunix's Gemma3 sharding config sharded model weights across the
4 TPU chips using the `fsdp` axis.

## Setup

| Field | Value |
| --- | --- |
| Model | `google/gemma-3-12b-it` |
| Checkpoint | `gs://gemma-data/checkpoints/gemma3-12b-it` |
| Tokenizer | `gs://gemma-data/tokenizers/tokenizer_gemma3.model` |
| TPU | Cloud TPU v5e `v5litepod-4` |
| Zone | `us-west4-a` |
| Chips | 4 |
| Mesh | `fsdp=4,tp=1` |
| Training | LoRA SFT |
| LoRA rank / alpha | `16 / 32` |
| Batch / context | `1 / 512` |
| Steps | 2 |
| Autopatches | disabled with `TUNIX_ACCEL_DISABLE_AUTOPATCH=1` |

## Measurements

| Metric | Value |
| --- | ---: |
| Final loss | 1.58552 |
| Mean loss | 2.06123 |
| Wall time | 144.99 s |
| First step time | 90.70 s |
| Second step time | 0.35 s |
| Train-step XLA memory | 12.39 GiB/chip |
| Runtime peak HBM | 48.95 GB aggregate across 4 chips |
| Runtime HBM limit | 67.64 GB aggregate across 4 chips |

The first step includes compilation. The second step is a better signal that
the compiled training step executed normally, but this is still only a smoke
test with tiny data.

## Artifacts

Raw files are under `raw/v5litepod4_fsdp4_tp1_lora_l512/`:

- `summary.json`
- `history.csv`
- `training_comparison.png`
- `train_step_memory_usage_report.txt`

## Reproduce

```bash
export TUNIX_ACCEL_DISABLE_AUTOPATCH=1
export XLA_FLAGS="--xla_dump_to=/tmp/xla_gemma3_12b_lora_fsdp4_tp1_l512 --xla_dump_hlo_as_text"

python 02-PACKING/run_gemma_training_benchmark.py \
  --model-id google/gemma-3-12b-it \
  --model-source gcs \
  --model-path gs://gemma-data/checkpoints/gemma3-12b-it \
  --tokenizer-source sentencepiece \
  --tokenizer-path gs://gemma-data/tokenizers/tokenizer_gemma3.model \
  --num-examples 64 \
  --variants unpacked \
  --batch-size 1 \
  --max-length 512 \
  --max-steps 2 \
  --learning-rate 0.0002 \
  --lora-rank 16 \
  --lora-alpha 32.0 \
  --mesh-fsdp 4 \
  --mesh-tp 1 \
  --max-inflight 1 \
  --skip-quality-eval \
  --outdir /tmp/gemma3-12b-lora-probe/fsdp4_tp1_b1_l512_plain
```

## Conclusion

Gemma3 12B is a much cleaner target than Gemma4 12B in the current Tunix
version. The model is registered upstream, the GCS checkpoint path works, and
`fsdp=4,tp=1` is enough to complete a small LoRA SFT smoke run on v5litepod-4.

Next probes should increase context length and batch size before introducing
CCE, Tiled MLP, or activation policies.
