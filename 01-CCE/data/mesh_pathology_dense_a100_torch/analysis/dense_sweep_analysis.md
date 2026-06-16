# Dense Mesh/Chunk Pathology Sweep Analysis

## Data status

- Rows loaded: 40
- Successful timing rows: 40
- Hardware targets: gpu-a100-80gb-4
- Execution stacks: PyTorch eager
- Operations: chunked_matmul_loop, collective_loop, projection_collective_loop

## Hypothesis Test

Hypothesis: CCE memory benefit is mostly governed by loss-logits dominance, while throughput failure is governed by the interaction between chunk-loop granularity and FSDP/TP mesh layout.

This script tests the throughput half with synthetic rows by modeling log step time using chunk-loop count, token/vocab loop counts, TP degree, collective presence, and their interactions. Memory evidence is carried through planned/runtime HBM columns when available; missing memory extraction is non-fatal.

- Log-time model coefficients: `log_step_time_model_coefficients.csv`
- Fastest configuration table: `fastest_per_stack_operation_mesh_shape.csv`
- Outlier report: `outlier_report.csv`
- Recovery report: `recovery_report.csv`

## Plots

- `gpu-a100-80gb-4_PyTorch_eager_chunk_loop_scatter.png`
- `gpu-a100-80gb-4_PyTorch_eager_b16_L512_token_vocab_heatmaps.png`

## Reading Guide

- If `log_loop_x_tp_x_collective` is large and positive, the strongest evidence points to loop granularity interacting with TP collectives rather than chunk count alone.
- If PyTorch eager rows show the same slope or outliers as JAX/XLA rows, XLA is not a necessary cause, although XLA may still amplify or hide the effect.
- If matmul-only rows are flat while collective/projection+collective rows explode at high loop counts, the root cause is communication/control overhead rather than loss projection math alone.
- If recovery rows show faster large chunks with similar HBM, chunk granularity is a performance knob independent of the memory-saving direction.

