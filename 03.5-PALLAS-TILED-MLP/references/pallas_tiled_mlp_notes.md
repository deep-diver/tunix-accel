# Pallas Tiled MLP Notes

The 03.5 experiment was motivated by the question of whether the Gemma3 tiled
MLP should move below ordinary XLA matmuls into Pallas TPU kernels.

What the retained experiment showed:

- Pallas backend integration works on the tested Gemma3 4B LoRA FSDP mesh.
- Same-model loss parity is exact on the parity batch.
- LoRA gradient norm is within 0.01% relative difference.
- L4096 planned HBM is slightly lower than the 03 XLA tiled backend.
- Step time is consistently slower than the 03 XLA tiled backend.
- 500-step runtime peak memory is slightly higher than the 03 XLA tiled backend.

Interpretation:

The high-leverage memory win in 03 came from tiled recompute of the
token-by-intermediate activation, not from replacing the matmul implementation
itself. A Pallas follow-up probably needs to cover a larger fused backward
region, a weight-gradient path, or another kernel family before it can plausibly
beat the current XLA tiled backend.
