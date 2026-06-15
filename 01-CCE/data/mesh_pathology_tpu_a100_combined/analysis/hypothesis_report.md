# Mesh/Chunk Pathology Hypothesis Report

Question: does the small-chunk slowdown generalize beyond CCE and beyond TPU v5e?

- Hardware targets observed: 2
- Workload families observed: 4
- Result rows: 32
- Successful rows: 32

Bad/good matched groups: 8
Groups with >2x bad/good slowdown: 6

Interpretation guide:
- If only `cce_train` shows a large ratio, suspect CCE implementation/lowering.
- If `projection_collective_loop` also shows it, suspect chunked matmul plus TP collectives.
- If `collective_loop` shows it, suspect collective granularity/latency directly.
- If `chunked_matmul_loop` shows it without collectives, suspect compile/lowering/kernel launch or tiny matmul shape.
