# Gemma3 4B Long-Context Splash Attention Frontier

This run tested whether the Gemma3 4B LoRA training case could move beyond the
dense-attention L8192 failure point on a larger TPU slice.

## Environment

- TPU: `v5litepod-16` in `us-west4-a`
- Global chips: 16
- Per-chip HBM reported by JAX: 15.75 GiB
- Mesh: `fsdp=16`, `tp=1`
- Model: `google/gemma-3-4b-it`
- Training mode: LoRA, rank 16, alpha 32
- Batch size: 1
- Patch stack: Cut Cross Entropy + Tiled MLP + split activation offload/remat + Splash Attention
- JAX: 0.10.1
- google-tunix: 0.1.6

## Result

The previous dense attention L8192 attempt failed during XLA compilation at
about 56.77 GiB per chip because Gemma3 materialized dense attention tensors
with shape like `bf16[4,8192,8192,2]`.

With the Splash Attention patch, the same 4B LoRA training path completed at
L8192, and longer contexts also completed:

| Context | Status | XLA train_step peak per chip | Headroom vs 15.75 GiB HBM | Recorded first train step |
|---:|---|---:|---:|---:|
| 8,192 | OK, 3 steps | 4.59 GiB | 11.16 GiB | 84 sec |
| 16,384 | OK, 1 step | 8.48 GiB | 7.27 GiB | 117 sec |
| 32,768 | OK, 1 step | 15.07 GiB | 0.68 GiB | 231 sec |

The practical conclusion is that Splash Attention removes the quadratic
attention-logit HBM wall. In the current replicated `shard_map` implementation,
memory then scales roughly linearly with context length and L32768 is already
near the v5e per-chip HBM limit.

## Artifacts

- `frontier_metrics.csv`: normalized run table.
- `frontier_metrics.json`: same metrics as JSON.
- `long_context_splash_frontier.png`: memory/time chart.
- `raw/L*/summary.json`: Tunix runner summaries.
- `raw/L*/history.csv`: train histories.
- `raw/L*/train_step_memory_report.txt`: XLA train_step memory reports.
- `raw/L*/run.log`: run logs.
