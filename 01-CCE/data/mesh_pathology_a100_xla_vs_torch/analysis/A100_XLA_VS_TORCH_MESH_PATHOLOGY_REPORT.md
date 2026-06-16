# A100 XLA vs PyTorch Mesh Pathology

This report compares the same A100 4-GPU synthetic mesh-pathology rows across JAX/XLA, PyTorch eager, and torch.compile/Inductor. The PyTorch eager rows are XLA-free CUDA/NCCL workloads.

- Rows loaded: 29
- Successful timing rows: 24
- Failed or aborted rows: 5

## Bad/Good Slowdown

| operation_family           | execution_stack   |      bad |      good |   control-fsdp4-tp1 |   control-fsdp1-tp4 |   bad_good_slowdown |
|:---------------------------|:------------------|---------:|----------:|--------------------:|--------------------:|--------------------:|
| chunked_matmul_loop        | JAX/XLA           | 0.052492 | 0.0195353 |           0.0519734 |            0.051432 |             2.68703 |
| chunked_matmul_loop        | PyTorch eager     | 0.181349 | 0.0363235 |           0.184427  |            0.195019 |             4.99259 |
| collective_loop            | JAX/XLA           | 0.131361 | 0.0361542 |           0.0279454 |            0.134647 |             3.63335 |
| collective_loop            | PyTorch eager     | 0.32626  | 0.0492778 |           0.0846466 |            0.329967 |             6.62083 |
| projection_collective_loop | JAX/XLA           | 0.174196 | 0.0453639 |           0.0516805 |            0.17761  |             3.83997 |
| projection_collective_loop | PyTorch eager     | 0.402278 | 0.0617799 |           0.180239  |            0.407029 |             6.51146 |

## Failure Summary

|   experiment_id | operation_family    | execution_stack   | signature_label   | status   | failure_type   | error_message                                                                                                      |
|----------------:|:--------------------|:------------------|:------------------|:---------|:---------------|:-------------------------------------------------------------------------------------------------------------------|
|              12 | chunked_matmul_loop | torch.compile     | bad               | failure  | runner_failure | Runner recorded status=failure before the dstack run was aborted; per-case runner.log was unavailable after abort. |
|              13 | chunked_matmul_loop | torch.compile     | good              | failure  | runner_failure | Runner recorded status=failure before the dstack run was aborted; per-case runner.log was unavailable after abort. |
|              14 | chunked_matmul_loop | torch.compile     | control-fsdp4-tp1 | failure  | runner_failure | Runner recorded status=failure before the dstack run was aborted; per-case runner.log was unavailable after abort. |
|              15 | chunked_matmul_loop | torch.compile     | control-fsdp1-tp4 | failure  | runner_failure | Runner recorded status=failure before the dstack run was aborted; per-case runner.log was unavailable after abort. |
|              16 | collective_loop     | torch.compile     | bad               | failure  | aborted_hang   | Run was manually aborted after no new progress while this case was running.                                        |

## Plots

- `mesh_pathology_a100_xla_vs_torch/analysis/a100_step_time_by_operation_stack.png`
- `mesh_pathology_a100_xla_vs_torch/analysis/a100_bad_good_slowdown_xla_vs_torch.png`
- `mesh_pathology_a100_xla_vs_torch/analysis/a100_normalized_signature_heatmap_xla_vs_torch.png`
- `mesh_pathology_a100_xla_vs_torch/analysis/a100_layered_3d_heatmap_xla_vs_torch.png`

## Most Likely Interpretation

PyTorch eager reproduces a strong small-chunk slowdown on A100 without XLA. That means XLA is not required for the phenomenon. The likely base mechanism is the granularity of many small loop bodies and, for collective workloads, the interaction with TP communication shape.

This does not prove XLA is irrelevant. JAX/XLA can still change the magnitude through lowering, fusion, and scheduling. In this run torch.compile did not produce usable timing rows for the compiled variants, so compiler-backed CUDA requires a separate smaller reproduction path.
