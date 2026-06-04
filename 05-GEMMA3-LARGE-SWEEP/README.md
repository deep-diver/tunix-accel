# Gemma3 Large Patch Sweep

This workstream checks whether the repository's drop-in memory patches still
move the boundary on large sharded Gemma3 LoRA runs. The corrected sweep covers
Gemma3 12B on 4 Cloud TPU v5e chips and Gemma3 27B on 8 Cloud TPU v5e chips.

## Result In One Sentence

Yes: memory reduction and context expansion both show up after the corrected
rerun. The strongest result is the stacked patch path: 12B moves from default
L1024 compile OOM to L4096 completion, and 27B moves from default L1024 compile
OOM to L2048 completion. The tradeoff is large step-time overhead on
offload-heavy paths.

## Setup

| Model | TPU | Chips | Mesh | Training | LoRA | Batch | Steps |
| --- | --- | ---: | --- | --- | --- | ---: | ---: |
| `google/gemma-3-12b-it` | `v5litepod-4` | 4 | `fsdp=4,tp=1` | SFT smoke | rank 16 / alpha 32 | 1 | 2 |
| `google/gemma-3-27b-it` | `v5litepod-8` | 8 | `fsdp=8,tp=1` | SFT smoke | rank 16 / alpha 32 | 1 | 2 |

Checkpoint and tokenizer:

- Checkpoints: `gs://gemma-data/checkpoints/gemma3-12b-it`,
  `gs://gemma-data/checkpoints/gemma3-27b-it`
- Tokenizer: `gs://gemma-data/tokenizers/tokenizer_gemma3.model`

Variants:

- `default`: acceleration patches disabled.
- `cce`: Cut Cross Entropy only.
- `tiled_mlp`: Tiled gated-MLP only.
- `split_offload`: activation split/offload policy only.
- `splash`: Splash Attention only.
- `stacked`: CCE + Tiled MLP + split/offload + Splash Attention.

The key metric is XLA train-step planned HBM **per chip** from the
buffer-assignment memory report. Aggregate memory is intentionally not used for
the headline because TPU OOM is decided per chip.

## Corrected Visual Summary

![Patch impact at L512](./assets/gemma3_large_l512_patch_impact.png)

![Batch-1 measured context frontier](./assets/gemma3_large_context_frontier.png)

![Practical readout](./assets/gemma3_large_practical_readout.png)

Additional diagnostics:

![L1024 fit-line check](./assets/gemma3_large_l1024_fit_line.png)

![Capacity frontier vs time cost](./assets/gemma3_large_frontier_vs_time.png)

![OOM gap](./assets/gemma3_large_oom_gap.png)

## Key Numbers

| Model | Variant | Longest completed context | First observed OOM | XLA HBM at longest success | Post-compile step time |
| --- | --- | ---: | --- | ---: | ---: |
| 12B | Default | 512 | L1024 at 17.42 GiB/chip | 12.39 GiB/chip | 0.35s |
| 12B | Tiled MLP | 1024 | L2048 at 20.42 GiB/chip | 14.13 GiB/chip | 0.44s |
| 12B | Split/offload | 2048 | not probed beyond L2048 | 13.67 GiB/chip | 100.86s |
| 12B | Stacked | 4096 | L8192 at 17.80 GiB/chip | 14.42 GiB/chip | 88.81s |
| 27B | Default | 512 | L1024 at 24.66 GiB/chip | 14.52 GiB/chip | 1.03s |
| 27B | Tiled MLP | 512 | L1024 at 19.01 GiB/chip | 13.58 GiB/chip | 0.43s |
| 27B | Split/offload | 1024 | not probed beyond L1024 | 14.53 GiB/chip | 150.89s |
| 27B | Stacked | 2048 | L4096 at 17.10 GiB/chip | 13.58 GiB/chip | 120.82s |

At the shared L512 baseline, the best memory-saving variants were:

| Model | Best L512 variant | Default HBM | Patched HBM | Savings |
| --- | --- | ---: | ---: | ---: |
| 12B | Stacked | 12.39 GiB/chip | 11.74 GiB/chip | 5.2% |
| 27B | Stacked | 14.52 GiB/chip | 13.29 GiB/chip | 8.5% |

The bigger story is not the L512 percentage. It is the boundary move at longer
context lengths. On 12B, Tiled MLP alone is enough to make L1024 fit. On 27B,
Tiled MLP still lowers the failed L1024 estimate from 24.66 to 19.01 GiB/chip,
but it does not cross the v5e fit line. Activation offload and the stacked path
do cross it, at the cost of very slow steps.

## Correction From The Earlier Run

An earlier sweep in this directory was invalid as patch evidence. The sweep
runner removed `TUNIX_ACCEL_DISABLE_AUTOPATCH` for patched variants, but the
training benchmark sets that variable back to `1` when it is absent. The fixed
runner now sets `TUNIX_ACCEL_DISABLE_AUTOPATCH=0` for patched variants and the
training summary records explicit patch-status fields.

The old invalid raw folders were removed from this package. The retained
corrected folders are:

- `raw/12b_corrected/`
- `raw/27b_corrected/`

## Artifacts

- Corrected summary data: `data/gemma3_large_patch_sweep_corrected_summary.csv`
- Compatibility copy: `data/gemma3_large_patch_sweep_summary.csv`
- Figure script: `make_figures.py`
- Full 12B raw archive:
  `gs://gcp-ml-172005-ddpm-training/tunix-large-sweep/gemma3-large-patch-sweep-corrected-12b.tar.gz`
- Full 27B raw archive:
  `gs://gcp-ml-172005-ddpm-training/tunix-large-sweep/gemma3-large-patch-sweep-corrected-27b.tar.gz`

This is a systems boundary sweep, not a translation-quality benchmark. Each
case ran only two steps, so the step-time numbers are useful for rough tradeoff
direction, not final throughput claims.
