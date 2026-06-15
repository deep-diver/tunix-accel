# General Mesh/Chunk Pathology Experiment Plan

Scope: test whether the small-chunk slowdown is a general mesh/loop/collective
pathology, not only a Cut Cross Entropy implementation issue.

## Matrix Size

There are 16 top-level scenario groups:

```text
4 workload families x 4 hardware targets = 16 scenario groups
```

The default pilot expands each scenario group across four target signatures
(`bad`, `good`, `control-fsdp4-tp1`, `control-fsdp1-tp4`), so the concrete
manifest has:

```text
16 scenario groups x 4 signatures = 64 experiment rows
```

## Workload Families

| workload | purpose |
| --- | --- |
| `cce_train` | Existing Tunix CCE training path. Tests the original phenomenon. |
| `chunked_matmul_loop` | Chunk-loop projection matmul with no explicit collectives. Isolates tiny matmul/lowering/launch behavior. |
| `collective_loop` | Repeated TP all-reduce over chunk-shaped tensors. Isolates collective granularity. |
| `projection_collective_loop` | Chunk-loop projection matmul plus TP all-reduce. Tests the CCE-like interaction without the CCE implementation. |

## Hardware Targets

| hardware target | role | dstack GPU spec |
| --- | --- | --- |
| `tpu-v5e-4` | TPU baseline | n/a |
| `gpu-a100-80gb-4` | Primary GPU baseline | `A100:4:80GB` |
| `gpu-l40s-48gb-4` | PCIe communication stress | `L40S:4:48GB` |
| `gpu-h100-80gb-4` | Upper-bound GPU | `H100:4:80GB` |

As of the dstack offer check on 2026-06-16 KST:

| GPU target | lowest observed offer |
| --- | --- |
| 4x A100 80GB | RunPod, `$5.96/hr`, available |
| 4x L40S 48GB | Nebius, `$9.1376/hr`, availability unknown |
| 4x H100 80GB | RunPod, `$13.16/hr`, available |

Recommendation: run `gpu-a100-80gb-4` first. Use L40S only if we want a PCIe
stress comparison, and H100 after A100/L40S establish whether the pathology is
general.

## Generate The Manifest

```bash
python3 01-CCE/run_mesh_pathology_matrix.py \
  --dry-run \
  --write-dstack-commands \
  --outdir /tmp/mesh-pathology-pilot
```

Expected output:

```text
scenario_groups=16
total_experiments=64
```

## Run Only The Primary GPU Target

First inspect offers:

```bash
uvx --from dstack dstack offer --gpu A100:4:80GB --max-offers 5
```

Then launch through dstack:

```bash
uvx --from dstack dstack apply \
  -f 01-CCE/dstack_mesh_pathology_gpu.yml \
  --gpu A100:4:80GB \
  -n mesh-pathology-gpu-a100-80gb-4 \
  -- \
  --hardware-targets gpu-a100-80gb-4 \
  --outdir /tmp/mesh-pathology-gpu-a100-80gb-4
```

## Run TPU v5e Target Locally On A TPU VM

```bash
python3 01-CCE/run_mesh_pathology_matrix.py \
  --hardware-targets tpu-v5e-4 \
  --outdir /tmp/mesh-pathology-tpu-v5e-4 \
  --force
```

## Analyze Partial Results

```bash
python3 01-CCE/analyze_mesh_pathology_results.py \
  --results-dir /tmp/mesh-pathology-gpu-a100-80gb-4/results \
  --outdir /tmp/mesh-pathology-gpu-a100-80gb-4/analysis
```

Important outputs:

| file | contents |
| --- | --- |
| `scenario_summary.csv` | success counts and median timing per hardware/workload group |
| `bad_good_ratios.csv` | bad/good slowdown ratio per hardware/workload group |
| `bad_good_slowdown_by_hardware_workload.png` | compact visualization of slowdown ratios |
| `hypothesis_report.md` | interpretation guide for whether the pathology generalized |
