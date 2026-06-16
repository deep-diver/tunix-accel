# Dense A100 PyTorch Eager Shard 0 Report

## Run

- Run name: `mesh-pathology-dense-a100-torch-s0`
- Hardware: 4x A100 80GB on dstack/RunPod spot
- Stack: PyTorch eager + CUDA/NCCL, no XLA
- Preset: `dense-synthetic`
- Shard: `--shard-index 0 --num-shards 8`
- Rows: 40/40 success
- Shape: `b16/L512`
- Operations: `chunked_matmul_loop`, `collective_loop`, `projection_collective_loop`

The fleet was deleted after artifact collection, and `dstack fleet list -v`
returned no active fleets.

## Key Observation

The slowdown is reproduced without XLA. In this shard, PyTorch eager shows
large within-operation slowdowns as chunk-loop count increases:

| operation | mesh | rows | worst/best |
| --- | --- | ---: | ---: |
| `chunked_matmul_loop` | `fsdp=4,tp=1` | 5 | 23.22x |
| `chunked_matmul_loop` | `fsdp=1,tp=4` | 5 | 6.04x |
| `chunked_matmul_loop` | `fsdp=2,tp=2` | 4 | 2.46x |
| `collective_loop` | `fsdp=1,tp=4` | 5 | 12.76x |
| `collective_loop` | `fsdp=4,tp=1` | 4 | 9.65x |
| `collective_loop` | `fsdp=2,tp=2` | 4 | 1.91x |
| `projection_collective_loop` | `fsdp=1,tp=4` | 5 | 24.60x |
| `projection_collective_loop` | `fsdp=4,tp=1` | 4 | 4.98x |
| `projection_collective_loop` | `fsdp=2,tp=2` | 4 | 3.70x |

The largest row is:

```text
projection_collective_loop, fsdp=1,tp=4, token/vocab chunk 64/4096,
chunk_loop_count=8192, steady_state_mean_step_time_sec=1.6495
```

The fastest row in that same operation/mesh group is:

```text
projection_collective_loop, fsdp=1,tp=4, token/vocab chunk 1024/65536,
chunk_loop_count=32, steady_state_mean_step_time_sec=0.0671
```

That is a 24.6x recovery from increasing chunk size on the same hardware, stack,
operation, shape, and mesh.

## Interpretation

This shard weakens the hypothesis that XLA is a necessary cause. The PyTorch
eager stack, using CUDA/NCCL and no XLA, still shows strong loop-granularity
pathology.

It does not prove XLA is irrelevant. The full dense grid still needs TPU/JAX and
GPU/JAX rows so we can compare slopes and determine whether XLA amplifies the
same underlying effect.

The shard also shows that collectives are not the only trigger: matmul-only rows
can be very slow at extreme loop counts. The current best formulation is:

```text
throughput failure = launch/control overhead from many tiny loop iterations
                     + TP/NCCL collective overhead when present
                     + mesh-dependent communication shape
                     + compiler/runtime lowering effects as an amplifier
```

## Generated Artifacts

- Raw shard rows:
  `data/mesh_pathology_dense_a100_torch/results/shard_0.jsonl`
- Merged rows:
  `data/mesh_pathology_dense_a100_torch/results/all_results.csv`
  and `data/mesh_pathology_dense_a100_torch/results/all_results.jsonl`
- Partial dense analysis:
  `data/mesh_pathology_dense_a100_torch/analysis/dense_sweep_analysis.md`
- Outlier table:
  `data/mesh_pathology_dense_a100_torch/analysis/outlier_report.csv`
- Missing-row table:
  `data/mesh_pathology_dense_a100_torch/analysis/missing_experiments.csv`
- Plots:
  `data/mesh_pathology_dense_a100_torch/analysis/gpu-a100-80gb-4_PyTorch_eager_chunk_loop_scatter.png`
  and
  `data/mesh_pathology_dense_a100_torch/analysis/gpu-a100-80gb-4_PyTorch_eager_b16_L512_token_vocab_heatmaps.png`

## Next Runs

1. Run the remaining seven A100/PyTorch eager shards to complete the no-XLA GPU
   plane.
2. Run the same dense plane on TPU v5e/JAX to compare with the TPU-side CCE
   failure.
3. Run A100/JAX/XLA on the same matrix to isolate GPU runtime versus XLA
   lowering.
4. Add one shape-scaling pass (`b16/L1024`, `b32/L512`) after the base plane is
   complete.
5. Keep torch.compile as a small diagnostic subset because earlier attempts
   produced failures/hangs rather than reliable timing rows.
