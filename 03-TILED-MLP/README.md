# 03 Tiled MLP

This workstream targets the next Unsloth-style memory optimization after Cut
Cross Entropy and sequence packing: reduce MLP activation/intermediate memory
without changing the training objective.

## Current Status

Implemented:

- `tunix_accel/tiled_mlp.py`
- `tests/test_tiled_mlp.py`

The first implementation covers SwiGLU/GeGLU-style gated MLP blocks:

```text
output = (activation(x @ gate) * (x @ up)) @ down
```

The tiled path streams the token dimension and uses a custom VJP. During
backward it recomputes each tile's gate/up/intermediate activations instead of
depending on a full `[tokens, intermediate_dim]` activation tensor from the
forward pass.

## Why This Is the 03 Target

CCE removes the full-vocab loss logits tensor. Once that pressure is reduced,
long-context training can move the peak toward transformer-block activations,
especially the MLP intermediate:

```text
batch_size * context_length * intermediate_dim
```

For Gemma/Llama-style gated MLPs, both `gate` and `up` projections produce large
token-by-intermediate tensors. A tiled implementation should trade extra
recompute or smaller GEMMs for lower activation residency.

## Verified Locally

The local environment does not currently include Tunix/Flax, so the first tests
are Gemma-free JAX checks:

```bash
python -m pytest -q tests/test_tiled_mlp.py
```

Current result:

```text
5 passed
```

Covered checks:

- forward parity against dense MLP for `silu`, `gelu`, `gelu_approx`, and
  `relu`
- gradients with respect to hidden, gate, up, and down kernels
- JIT-compiled gradient parity
- simple dense-vs-tiled intermediate memory estimate helper

## API Sketch

```python
from tunix_accel.tiled_mlp import tiled_gated_mlp

out = tiled_gated_mlp(
    hidden,
    gate_kernel,
    up_kernel,
    down_kernel,
    token_chunk=256,
    activation="silu",
)
```

For comparison and tests:

```python
from tunix_accel.tiled_mlp import dense_gated_mlp
```

## Next Milestones

1. Inspect actual Tunix Gemma3 MLP module structure in a Tunix-enabled
   environment.
2. Add a best-effort Tunix/Gemma adapter that swaps supported `gate_proj`,
   `up_proj`, and `down_proj` MLP blocks to the tiled kernel.
3. Run tiny Gemma forward/loss parity with and without the patch.
4. Run TPU microbenchmarks:
   - dense MLP vs tiled MLP
   - CCE vs CCE + tiled MLP
   - context frontier and XLA planned HBM
5. Decide whether Pallas is needed. If XLA already lowers the tiled custom-VJP
   path well, Pallas may not be worth the added model-family complexity.

## Known Risks

- Tiling may reduce memory but slow steps because large GEMMs become several
  smaller sequential GEMMs.
- Drop-in replacement is more model-family-sensitive than CCE. Gemma, Llama,
  Qwen, and GeGLU models may expose different MLP module layouts.
- XLA might already optimize some MLP patterns well enough that a handwritten
  Pallas kernel is not the first bottleneck.
