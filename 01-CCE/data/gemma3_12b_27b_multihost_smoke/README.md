# Gemma3 12B/27B Multi-Host Tunix Smoke

This folder records actual Tunix/Gemma LoRA smoke tests on multi-host TPU
topologies.

## Scope

- Gemma3 12B IT on `v5litepod-16`, `fsdp=16`, `tp=1`
- Gemma3 27B IT on `v5litepod-32`, `fsdp=32`, `tp=1`
- Batch size 1, context length 512, LoRA rank 16
- Two train steps, synthetic data, quality evaluation skipped
- Both default CE and CCE-enabled runs

The runner called `jax.distributed.initialize()` before model setup. The logs
confirm:

- 12B: `process_count=4`, `local_devices=4`, `global_devices=16`
- 27B: `process_count=8`, `local_devices=4`, `global_devices=32`

## Files

- `multihost_smoke_summary.csv`: compact comparison table.
- `multihost_smoke_summary.json`: JSON version of the same summary.
- `raw_artifacts/*.tar.gz`: original per-run artifacts copied back from TPU VM
  worker 0. These contain runner logs, history CSVs, summaries, and XLA memory
  reports. They do not contain checkpoints.

## Metric Note

The raw artifacts are copied from one process/host. Therefore
`runtime_peak_hbm_gb_local_host` and `runtime_hbm_limit_gb_local_host` are local
host aggregates over that host's 4 TPU chips, not full-pod aggregates.
`xla_train_step_gib_per_chip` is the per-chip XLA train-step high-water estimate.
