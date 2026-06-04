# Gemma3 27B LoRA TPU Probe

This probe checks the next registered Gemma3 size after 12B:
`google/gemma-3-27b-it`.

The run used plain Tunix LoRA SFT with repository acceleration patches disabled.

## Result

Gemma3 27B LoRA SFT completed a 2-step smoke run on Cloud TPU v5e
`v5litepod-8`.

As with the 12B probe, `tp=1` means tensor parallelism was disabled. The model
was still sharded across chips through the `fsdp` axis: `fsdp=8,tp=1`.

## Setup

| Field | Value |
| --- | --- |
| Model | `google/gemma-3-27b-it` |
| Checkpoint | `gs://gemma-data/checkpoints/gemma3-27b-it` |
| Tokenizer | `gs://gemma-data/tokenizers/tokenizer_gemma3.model` |
| TPU | Cloud TPU v5e `v5litepod-8` |
| Zone | `us-west4-a` |
| Chips | 8 |
| Mesh | `fsdp=8,tp=1` |
| Training | LoRA SFT |
| LoRA rank / alpha | `16 / 32` |
| Batch / context | `1 / 512` |
| Steps | 2 |
| Autopatches | disabled with `TUNIX_ACCEL_DISABLE_AUTOPATCH=1` |

## Measurements

| Metric | Value |
| --- | ---: |
| Final loss | 0.90183 |
| Mean loss | 1.66640 |
| Wall time | 196.37 s |
| First step time | 129.91 s |
| Second step time | 0.51 s |
| Train-step XLA memory | 14.52 GiB/chip |
| Runtime peak HBM | 114.08 GB aggregate across 8 chips |
| Runtime HBM limit | 135.27 GB aggregate across 8 chips |

The first step includes compilation. The second step confirms the compiled
training step executed normally. This is still only a small smoke run at
context length 512.

## Artifacts

Raw files are under `raw/v5litepod8_fsdp8_tp1_lora_l512/`:

- `summary.json`
- `history.csv`
- `training_comparison.png`
- `train_step_memory_usage_report.txt`

## Reproduce

```bash
export TUNIX_ACCEL_DISABLE_AUTOPATCH=1
export XLA_FLAGS="--xla_dump_to=/tmp/xla_gemma3_27b_lora_fsdp8_tp1_l512 --xla_dump_hlo_as_text"

python 02-PACKING/run_gemma_training_benchmark.py \
  --model-id google/gemma-3-27b-it \
  --model-source gcs \
  --model-path gs://gemma-data/checkpoints/gemma3-27b-it \
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
  --mesh-fsdp 8 \
  --mesh-tp 1 \
  --max-inflight 1 \
  --skip-quality-eval \
  --outdir /tmp/gemma3-27b-lora-probe/fsdp8_tp1_b1_l512_plain
```

## Conclusion

Gemma3 27B is also a clean Tunix target for a minimal LoRA SFT smoke run, but
it is much tighter than 12B on v5e. At L512, the train-step XLA report reaches
14.52 GiB/chip on a nominal 16 GiB class device. Larger contexts should be
tested carefully and probably need CCE, activation policy work, or a larger
slice.
