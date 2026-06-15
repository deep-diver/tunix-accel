# Mesh/Chunk Pathology Hypothesis Report

Question: does the small-chunk slowdown generalize beyond CCE and beyond TPU v5e?

- Hardware targets observed: 1
- Workload families observed: 4
- Result rows: 16
- Successful rows: 12

Bad/good matched groups: 3
Groups with >2x bad/good slowdown: 3

Interpretation guide:
- If only `cce_train` shows a large ratio, suspect CCE implementation/lowering.
- If `projection_collective_loop` also shows it, suspect chunked matmul plus TP collectives.
- If `collective_loop` shows it, suspect collective granularity/latency directly.
- If `chunked_matmul_loop` shows it without collectives, suspect compile/lowering/kernel launch or tiny matmul shape.
