# Dense Mesh/Chunk Pathology Sweep Plan

## Objective

We are no longer treating the CCE slowdown as a CCE-only question. The dense
sweep asks whether the same throughput pathology appears in simpler workloads
that share CCE's ingredients:

- many chunk-loop iterations,
- projection-shaped matmuls,
- repeated TP collectives,
- FSDP/TP mesh layouts on 4 accelerators,
- JAX/XLA on TPU and GPU, and PyTorch eager on GPU without XLA.

The core question is whether slow rows are better explained by XLA itself, or by
the interaction between loop granularity, TP communication, and mesh layout.

## Default Dense Design

The `dense-synthetic` preset generates a full-factorial synthetic sweep:

- shape: `b16/L512`
- meshes: `fsdp=4,tp=1`, `fsdp=2,tp=2`, `fsdp=1,tp=4`
- token chunks: `64`, `128`, `256`, `512`, `1024`
- vocab chunks: `4096`, `8192`, `16384`, `32768`, `65536`, `131072`, `262144`
- operations:
  - `chunked_matmul_loop`
  - `collective_loop`
  - `projection_collective_loop`

This is 315 rows per accelerator/stack:

```text
1 shape * 3 meshes * 5 token chunks * 7 vocab chunks * 3 operations = 315
```

The shape-scaling extension adds `b16/L1024` and `b32/L512`, producing 945 rows
per accelerator/stack.

## Stacks

Run equivalent designs for:

- TPU v5e JAX/XLA
- GPU A100 JAX/XLA
- GPU A100 PyTorch eager

Optional follow-ups:

- GPU L40S PyTorch eager, to stress PCIe communication
- GPU H100 PyTorch eager, to test whether a faster accelerator hides the effect
- torch.compile diagnostic subset only, because earlier A100 attempts produced
  failures/hangs rather than reliable timing rows

## Identification Strategy

The analysis script creates features for:

- `token_loop_count`
- `vocab_loop_count`
- `chunk_loop_count`
- `tp_degree`
- `has_collective`
- `has_matmul`
- `log(chunk_loop_count) * tp_degree`
- `log(chunk_loop_count) * has_collective`
- `log(chunk_loop_count) * tp_degree * has_collective`

The key test is the sign and magnitude of the three-way interaction. A large
positive coefficient means the slowdown is not just "many loops"; it is many
loops under TP collective-bearing operations.

## Expected Interpretations

- If PyTorch eager reproduces the slowdown, XLA is not a necessary cause.
- If JAX/XLA is much worse than PyTorch eager at the same loop count and mesh,
  XLA may still amplify the problem.
- If matmul-only rows stay flat while collective rows explode, the failure is
  dominated by communication/control overhead.
- If `fsdp=4,tp=1` stays fast at high loop counts, TP communication is necessary
  for the worst pathology.
- If larger chunks recover speed with similar HBM, chunk granularity is a
  practical recovery lever independent of memory savings.

## Commands

Generate the TPU dense manifest without running:

```bash
python3 01-CCE/run_mesh_pathology_matrix.py \
  --preset dense-synthetic \
  --hardware-targets tpu-v5e-4 \
  --dry-run \
  --outdir /tmp/mesh-pathology-dense-tpu
```

Run one 8-way TPU shard:

```bash
python3 01-CCE/run_mesh_pathology_matrix.py \
  --preset dense-synthetic \
  --hardware-targets tpu-v5e-4 \
  --num-shards 8 \
  --shard-index 0 \
  --case-timeout-sec 900 \
  --outdir /tmp/mesh-pathology-dense-tpu \
  --force
```

Run one dstack A100 PyTorch eager shard:

```bash
uvx --from dstack dstack apply \
  -f 01-CCE/dstack_mesh_pathology_synthetic_gpu.yml \
  --gpu A100:4:80GB \
  -n mesh-pathology-dense-a100-torch-s0 \
  -- \
  --preset dense-synthetic \
  --hardware-targets gpu-a100-80gb-4 \
  --workloads torch_eager_chunked_matmul_loop,torch_eager_collective_loop,torch_eager_projection_collective_loop \
  --num-shards 8 \
  --shard-index 0 \
  --case-timeout-sec 900 \
  --outdir /tmp/mesh-pathology-dense-a100-torch
```

Analyze partial results:

```bash
python3 01-CCE/analyze_mesh_pathology_dense_sweep.py \
  --results /tmp/mesh-pathology-dense-a100-torch/results/shard_*.jsonl \
  --manifest /tmp/mesh-pathology-dense-a100-torch/manifest.jsonl \
  --outdir /tmp/mesh-pathology-dense-a100-torch/analysis
```

## Non-Negotiables

- Failed cases must be rows, not dropped work.
- Shards must be independently resumable.
- GPU instances/fleets must be stopped/deleted after artifact collection.
- The analysis must run on partial results.
- CCE dense training remains separate from this synthetic root-cause sweep.
