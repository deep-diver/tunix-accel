# 01-CCE

This directory contains the final artifacts for the Cut Cross Entropy experiment.
The main report is now rebuilt around a complete Gemma3 270M rerun before any
larger-model transfer checks are considered.

## Contents

- `TECHNICAL_REPORT.md`: final narrative report with embedded plots.
- `REPRODUCE.md`: guide for reproducing the experiment family.
- `GEMMA3_270M_EXPERIMENT_DESIGN.md`: clean rerun design for rebuilding the CCE
  evidence chain around Gemma3 270M before scaling the claim outward.
- `run_gemma3_270m_cce_sweep.py`: local/TPU sweep runner used by the rerun.
- `run_gemma_training_benchmark.py`: CCE/default Tunix training runner invoked
  by the sweep runner.
- `run_v5e_cce_matrix.py`: structured TPU v5e-only CCE mesh/chunk experiment
  runner with dry-run manifest generation, sharding, and optional profiler
  cases.
- `merge_v5e_cce_results.py`: merge sharded JSONL outputs into combined JSONL
  and CSV files.
- `analyze_v5e_cce_results.py`: partial-results analysis for fastest configs,
  CCE/default speedups, outliers, recovery cases, and hypothesis checks.
- `run_mesh_pathology_matrix.py`: generalized TPU/GPU matrix for checking
  whether the small-chunk slowdown appears outside CCE.
- `run_mesh_pathology_microbench.py`: JAX synthetic chunk-loop/collective
  microbenchmarks used by the generalized matrix.
- `analyze_mesh_pathology_results.py`: partial-results analysis for generalized
  hardware/workload bad-vs-good slowdown ratios.
- `remote_gemma3_270m_cce_worker.sh`: TPU VM profile wrapper.
- `collect_gemma3_270m_cce_results.py`: artifact collector and plot generator.
- `collect_gemma3_270m_mesh_results.py`: four-chip mesh compatibility collector.
- `collect_gemma3_270m_mesh_repeat_results.py`: repeated four-chip mesh timing
  collector.
- `collect_gemma3_270m_4chip_frontier_results.py`: extended four-chip context
  frontier collector.
- `collect_gemma3_270m_outlier_hlo_results.py`: compact HLO text-scan collector
  for the mixed-mesh throughput outlier.
- `collect_gemma3_270m_4chip_chunk_results.py`: mixed-mesh CCE chunk tuning
  collector.
- `collect_gemma3_270m_4chip_quality_results.py`: four-chip OPUS100 parity
  collector.
- `collect_gemma_1b_e2b_cce_transfer_results.py`: Gemma3 1B and Gemma4 E2B
  transfer-check collector.
- `collect_gemma_4b_e4b_cce_transfer_results.py`: focused Gemma3 4B and
  Gemma4 E4B eight-chip transfer-check collector.
- `collect_gemma3_12b_27b_cce_focused_results.py`: focused Gemma3 12B and
  27B eight-chip boundary-check collector.
- `assets/`: final plots used by the report.
- `data/`: compact CSV/JSONL summaries retained from the experiments,
  including the Gemma4 base boundary rows.
- `references/`: background notes used to frame the benchmark story.

Raw TPU profiler traces, XLA dump directories, smoke runs, intermediate plots,
and training checkpoints were intentionally removed. They were useful while
debugging, but they are not needed to read the result or reproduce the setup.

The patch implementation itself lives outside this directory in `tunix_accel/`.

## Gemma3 270M Evidence Package

The current primary evidence package is:

- Data: `data/gemma3_270m_full_cce/`
- Figures:
  - `assets/gemma3_270m_cce_frontier.png`
  - `assets/gemma3_270m_cce_status_heatmap.png`
  - `assets/gemma3_270m_cce_tuning.png`
  - `assets/gemma3_270m_cce_quality.png`
  - `assets/gemma3_270m_cce_mesh_2x2_repeat.png`
  - `assets/gemma3_270m_cce_4chip_frontier.png`
  - `assets/gemma3_270m_cce_outlier_hlo.png`
  - `assets/gemma3_270m_cce_4chip_chunk_tuning.png`
  - `assets/gemma3_270m_cce_4chip_chunk_axis_ablation.png`
  - `assets/gemma3_270m_cce_4chip_quality.png`
  - `assets/gemma_cce_transfer_frontier.png`
  - `assets/gemma_cce_transfer_quality.png`
  - `assets/gemma_cce_transfer_chunk_mesh.png`
  - `assets/gemma_cce_large_transfer_frontier.png`
  - `assets/gemma_cce_large_transfer_pressure.png`
  - `assets/gemma_cce_large_transfer_chunk_tuning.png`
  - `assets/gemma3_12b_27b_cce_focused_frontier.png`
  - `assets/gemma3_12b_27b_cce_focused_hbm.png`

The primary rerun rows used Cloud TPU `v5litepod-1`, one chip, in
`us-west4-a`. The mesh generalization check used `v5litepod-4`, four chips, in
the same zone, with `fsdp=4,tp=1`, `fsdp=2,tp=2`, and `fsdp=1,tp=4`.

Mesh check data:

- `data/gemma3_270m_mesh_cce/run_manifest.csv`
- `data/gemma3_270m_mesh_cce/mesh_runs.csv`
- `data/gemma3_270m_mesh_cce/mesh_summary.csv`
- `data/gemma3_270m_mesh_cce/matched_memory.csv`
- `data/gemma3_270m_mesh_cce_repeat/repeat_summary.csv`
- `data/gemma3_270m_4chip_frontier/frontier_summary.csv`
- `data/gemma3_270m_outlier_hlo/hlo_op_counts.csv`
- `data/gemma3_270m_4chip_chunk/chunk_summary.csv`
- `data/gemma3_270m_4chip_chunk/chunk_axis_ablation.csv`
- `data/gemma3_270m_4chip_quality/training_summary.csv`
- `data/gemma_1b_e2b_cce_transfer/frontier_summary.csv`
- `data/gemma_1b_e2b_cce_transfer/training_summary.csv`
- `data/gemma_1b_e2b_cce_transfer/chunk_summary.csv`
- `data/gemma_4b_e4b_cce_transfer/frontier_summary.csv`
- `data/gemma_4b_e4b_cce_transfer/pressure_points.csv`
- `data/gemma_4b_e4b_cce_transfer/chunk_summary.csv`
- `data/gemma3_12b_27b_cce_focused/frontier_summary.csv`
- `data/gemma3_12b_27b_cce_focused/boundary_hbm_points.csv`
- `data/gemma3_12b_27b_cce_focused/matched_metrics.csv`
- `data/gemma3_12b_27b_multihost_smoke/multihost_smoke_summary.csv`

The extracted `data/gemma3_270m_full_cce/raw/` directory is disposable and
should not be committed. Recreate it from `raw_artifacts/*.tar.gz` with:

```bash
python3 01-CCE/collect_gemma3_270m_cce_results.py
python3 01-CCE/collect_gemma3_270m_mesh_results.py
python3 01-CCE/collect_gemma3_270m_mesh_repeat_results.py
python3 01-CCE/collect_gemma3_270m_4chip_frontier_results.py
python3 01-CCE/collect_gemma3_270m_outlier_hlo_results.py
python3 01-CCE/collect_gemma3_270m_4chip_chunk_results.py
python3 01-CCE/collect_gemma3_270m_4chip_quality_results.py
python3 01-CCE/collect_gemma_1b_e2b_cce_transfer_results.py
python3 01-CCE/collect_gemma_4b_e4b_cce_transfer_results.py
python3 01-CCE/collect_gemma3_12b_27b_cce_focused_results.py
```

## TPU v5e CCE Failure Matrix

The structured v5e runner is for the current CCE/FSDP/TP failure-mode
experiment. It defaults to Gemma3 270M, `v5litepod-4`, synthetic data, 3 warmup
steps, and 10 measured steps. GPU execution is intentionally not included yet,
but result rows include a `backend` field so later GPU runners can share the
schema.

Generate the full core matrix without running it:

```bash
python3 01-CCE/run_v5e_cce_matrix.py \
  --preset core \
  --dry-run \
  --outdir /tmp/v5e-cce-core
```

Run one experiment from the generated matrix:

```bash
python3 01-CCE/run_v5e_cce_matrix.py \
  --preset core \
  --experiment-id 1 \
  --outdir /tmp/v5e-cce-core
```

Run the core v5e matrix:

```bash
python3 01-CCE/run_v5e_cce_matrix.py \
  --preset core \
  --outdir /tmp/v5e-cce-core
```

The core preset covers `fsdp=4,tp=1`, `fsdp=2,tp=2`, and `fsdp=1,tp=4` over
`b16/L512`, `b16/L1024`, and `b32/L512`, with Default CE plus CCE chunks
`128/8192`, `128/32768`, `256/32768`, `512/32768`, and `512/65536`.

Run the bad-row reproduction preset:

```bash
python3 01-CCE/run_v5e_cce_matrix.py \
  --preset bad-row \
  --outdir /tmp/v5e-cce-bad-row
```

This preset repeats CCE runs three times for `fsdp=2,tp=2`, `b16/L512` and
`b16/L1024`, with chunks `128/8192`, `256/32768`, and `512/65536`.

Run only the selected profiler cases:

```bash
python3 01-CCE/run_v5e_cce_matrix.py \
  --preset profiler \
  --enable-profiler \
  --keep-all-xla \
  --full-hlo-dump \
  --outdir /tmp/v5e-cce-profiler
```

Profiler traces are written under `/tmp/v5e-cce-profiler/profiler/<case_name>/`,
and each result row records `profiler_path`. `--keep-all-xla --full-hlo-dump`
keeps optimized HLO, StableHLO/compiler IR, and XLA memory reports for
post-run root-cause analysis. If `--enable-profiler` is used with the core
preset, only the four selected profiler signatures are traced.

Generate the profiler/HLO root-cause report from those four cases:

```bash
python3 01-CCE/analyze_v5e_cce_profiler_hlo.py \
  --results-jsonl /tmp/v5e-cce-profiler/results/shard_0.jsonl \
  --run-root /tmp/v5e-cce-profiler/runs \
  --trace-root /tmp/v5e-cce-profiler/profiler \
  --outdir 01-CCE/data/v5e_cce_profiler_hlo \
  --report-path 01-CCE/profiler_hlo_analysis.md \
  --plot-path 01-CCE/assets/v5e_cce_profiler_hlo_root_cause.png
```

The script writes a markdown report, a compact root-cause PNG, per-case metric
CSV, extracted CCE inner-loop shape CSV, and non-fatal extraction notes. Missing
trace/HLO files are logged rather than treated as fatal.

Shard the core matrix across four v5e instances:

```bash
python3 01-CCE/run_v5e_cce_matrix.py --preset core \
  --num-shards 4 --shard-index 0 --outdir /tmp/v5e-cce-core
python3 01-CCE/run_v5e_cce_matrix.py --preset core \
  --num-shards 4 --shard-index 1 --outdir /tmp/v5e-cce-core
python3 01-CCE/run_v5e_cce_matrix.py --preset core \
  --num-shards 4 --shard-index 2 --outdir /tmp/v5e-cce-core
python3 01-CCE/run_v5e_cce_matrix.py --preset core \
  --num-shards 4 --shard-index 3 --outdir /tmp/v5e-cce-core
```

Each shard writes one JSONL result file, for example
`/tmp/v5e-cce-core/results/shard_0.jsonl`. Failed experiments are recorded as
rows with `status=failure` and an `error_message`.

Merge shard outputs:

```bash
python3 01-CCE/merge_v5e_cce_results.py \
  --results-dir /tmp/v5e-cce-core/results
```

The merge script writes:

```text
/tmp/v5e-cce-core/results/all_results.jsonl
/tmp/v5e-cce-core/results/all_results.csv
```

Run the analysis on complete or partial results:

```bash
python3 01-CCE/analyze_v5e_cce_results.py \
  --results-dir /tmp/v5e-cce-core/results \
  --outdir /tmp/v5e-cce-core/analysis
```

The analysis produces fastest-configuration, CCE-vs-Default, outlier, recovery,
and hypothesis-check tables, plus plots for step time vs CCE loop count, step
time vs chunk configuration grouped by mesh, and memory saving vs shape.

## General Mesh/Chunk Pathology Matrix

The generalized matrix asks whether the small-chunk slowdown appears outside
Cut Cross Entropy and outside TPU v5e. The default pilot has 16 scenario groups:
four workload families times four hardware targets. Each group expands across
the four target signatures from the profiler/HLO study, so the concrete manifest
has 64 experiment rows.

Generate the manifest and dstack GPU launch commands:

```bash
python3 01-CCE/run_mesh_pathology_matrix.py \
  --dry-run \
  --write-dstack-commands \
  --outdir /tmp/mesh-pathology-pilot
```

Run the primary GPU target after inspecting offers:

```bash
uvx --from dstack dstack offer --gpu A100:4:80GB --max-offers 5

uvx --from dstack dstack apply \
  -f 01-CCE/dstack_mesh_pathology_gpu.yml \
  --gpu A100:4:80GB \
  -n mesh-pathology-gpu-a100-80gb-4 \
  -- \
  --hardware-targets gpu-a100-80gb-4 \
  --outdir /tmp/mesh-pathology-gpu-a100-80gb-4
```

Analyze complete or partial generalized results:

```bash
python3 01-CCE/analyze_mesh_pathology_results.py \
  --results-dir /tmp/mesh-pathology-gpu-a100-80gb-4/results \
  --outdir /tmp/mesh-pathology-gpu-a100-80gb-4/analysis
```

See `MESH_PATHOLOGY_EXPERIMENT_PLAN.md` for the workload definitions, hardware
target rationale, and the latest dstack offer snapshot used for planning.

### Dense Synthetic Sweep

Use the dense preset when the goal is root-cause identification rather than the
four-row CCE reproduction. The default dense design uses the same three
synthetic operations on one shape (`b16/L512`), three 4-device meshes, five
token chunk sizes, and seven vocab chunk sizes:

```bash
python3 01-CCE/run_mesh_pathology_matrix.py \
  --preset dense-synthetic \
  --hardware-targets tpu-v5e-4 \
  --dry-run \
  --outdir /tmp/mesh-pathology-dense-tpu
```

That default creates 315 rows per hardware/stack. To make the shape axis denser:

```bash
python3 01-CCE/run_mesh_pathology_matrix.py \
  --preset dense-synthetic \
  --hardware-targets tpu-v5e-4 \
  --shapes b16/L512,b16/L1024,b32/L512 \
  --dry-run \
  --outdir /tmp/mesh-pathology-dense-tpu-shapes
```

Run one TPU shard:

```bash
python3 01-CCE/run_mesh_pathology_matrix.py \
  --preset dense-synthetic \
  --hardware-targets tpu-v5e-4 \
  --shapes b16/L512,b16/L1024,b32/L512 \
  --num-shards 8 \
  --shard-index 0 \
  --case-timeout-sec 900 \
  --outdir /tmp/mesh-pathology-dense-tpu \
  --force
```

Run the same design on GPU/JAX or GPU/PyTorch eager by selecting workloads:

```bash
# GPU/JAX
python3 01-CCE/run_mesh_pathology_matrix.py \
  --preset dense-synthetic \
  --hardware-targets gpu-a100-80gb-4 \
  --workloads chunked_matmul_loop,collective_loop,projection_collective_loop \
  --num-shards 8 \
  --shard-index 0 \
  --case-timeout-sec 900 \
  --outdir /tmp/mesh-pathology-dense-a100-jax \
  --force

# GPU/PyTorch eager, no XLA
python3 01-CCE/run_mesh_pathology_matrix.py \
  --preset dense-synthetic \
  --hardware-targets gpu-a100-80gb-4 \
  --workloads torch_eager_chunked_matmul_loop,torch_eager_collective_loop,torch_eager_projection_collective_loop \
  --num-shards 8 \
  --shard-index 0 \
  --case-timeout-sec 900 \
  --outdir /tmp/mesh-pathology-dense-a100-torch \
  --force
```

Launch a dense GPU shard on dstack with the lighter synthetic environment:

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

After collecting logs/artifacts from a dstack run, stop the run/fleet so paid
capacity is released:

```bash
uvx --from dstack dstack stop -n mesh-pathology-dense-a100-torch-s0 --yes
uvx --from dstack dstack fleet list -v
```

The sweep axes can be overridden directly:

```bash
python3 01-CCE/run_mesh_pathology_matrix.py \
  --preset dense-synthetic \
  --hardware-targets tpu-v5e-4 \
  --mesh-configs 4x1,2x2,1x4 \
  --token-chunks 64,128,256,512,1024 \
  --vocab-chunks 4096,8192,16384,32768,65536,131072,262144 \
  --dry-run \
  --outdir /tmp/mesh-pathology-dense-custom
```

Merge shard outputs, tolerating missing shard files and failed rows:

```bash
python3 01-CCE/merge_v5e_cce_results.py \
  --results-dir /tmp/mesh-pathology-dense-a100-torch/results
```

Analyze complete or partial dense results:

```bash
python3 01-CCE/analyze_mesh_pathology_dense_sweep.py \
  --results /tmp/mesh-pathology-dense-a100-torch/results/shard_*.jsonl \
  --manifest /tmp/mesh-pathology-dense-a100-torch/manifest.jsonl \
  --outdir /tmp/mesh-pathology-dense-a100-torch/analysis
```

The dense analysis writes fastest-configuration, outlier, recovery, missing-row,
failure, and log-step-time model tables. The model explicitly tests whether the
slowdown is explained by chunk-loop count alone or by the interaction between
loop count, TP degree, and collective-bearing operations. Scatter and heatmap
plots are emitted per accelerator/stack so accelerators are not visually
compared against each other.

## Gemma3 1B / Gemma4 E2B Transfer Package

The transfer package asks whether the 270M CCE findings survive a larger
four-chip FSDP setup. It used Cloud TPU `v5litepod-4`, four chips, in
`us-west4-a`, with `fsdp=4,tp=1` as the primary mesh.

- Data: `data/gemma_1b_e2b_cce_transfer/`
- Raw artifacts:
  `data/gemma_1b_e2b_cce_transfer/raw_artifacts/*.tar.gz`
- Figures:
  - `assets/gemma_cce_transfer_frontier.png`
  - `assets/gemma_cce_transfer_quality.png`
  - `assets/gemma_cce_transfer_chunk_mesh.png`

## Gemma3 4B / Gemma4 E4B Focused Transfer Package

The larger-model focused transfer package asks whether the same CCE memory-wall
pattern survives on an eight-chip FSDP-only setup. It used Cloud TPU
`v5litepod-8`, eight chips, in `us-west4-a`, with `fsdp=8,tp=1`.

- Data: `data/gemma_4b_e4b_cce_transfer/`
- Raw artifacts:
  `data/gemma_4b_e4b_cce_transfer/raw_artifacts/*.tar.gz`
- Figures:
  - `assets/gemma_cce_large_transfer_frontier.png`
  - `assets/gemma_cce_large_transfer_pressure.png`
  - `assets/gemma_cce_large_transfer_chunk_tuning.png`

## Gemma3 12B / 27B Focused Boundary Package

The 12B/27B focused boundary package asks where CCE stops moving the fit
frontier under the same single-host eight-chip setup. It used Cloud TPU
`v5litepod-8`, eight chips, in `us-west4-a`, with `fsdp=8,tp=1`. In this rerun
CCE reduced XLA planned HBM, but did not create a new passing batch/context
shape.

- Data: `data/gemma3_12b_27b_cce_focused/`
- Raw artifacts:
  `data/gemma3_12b_27b_cce_focused/raw_artifacts/*.tar.gz`
- Figures:
  - `assets/gemma3_12b_27b_cce_focused_frontier.png`
  - `assets/gemma3_12b_27b_cce_focused_hbm.png`

The separate multi-host smoke package confirms actual Tunix distributed launch
for larger slices after enabling JAX distributed initialization:

- Gemma3 12B: `v5litepod-16`, `fsdp=16,tp=1`, `process_count=4`,
  `global_devices=16`
- Gemma3 27B: `v5litepod-32`, `fsdp=32,tp=1`, `process_count=8`,
  `global_devices=32`
- Data: `data/gemma3_12b_27b_multihost_smoke/`

## Gemma4 Rows

The Gemma4 base boundary rows are retained as transfer-check data from the
earlier integrated report. They are not part of the current Gemma3 270M evidence
claim, which is intentionally narrower:

- Data: `data/gemma4_base_cce_tpu_l2048_b1.csv`
- Complete cross-workstream boundary table:
  `data/gemma4_base_tpu_l2048_b1_all_variants.csv`
- Figure: `assets/gemma3_gemma4_cce_per_chip_hbm.png`
