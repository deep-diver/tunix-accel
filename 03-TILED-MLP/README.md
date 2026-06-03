# 03 Tiled MLP

This workstream targets the next Unsloth-style memory optimization after Cut
Cross Entropy and sequence packing: reduce Gemma3 MLP activation/intermediate
memory without changing the training objective.

## Scope

The scope is **Gemma3-only** until proven otherwise.

The current JAX kernel is generic for gated MLP math, but the drop-in patching
surface should not claim broad model-family support yet. The first production
target is Tunix Gemma3 modules exposing the familiar `gate_proj`, `up_proj`, and
`down_proj` MLP structure. Llama, Qwen, and GeGLU variants stay out of scope for
this workstream unless Gemma3 results justify a later adapter expansion.

## Current Status

Implemented:

- `tunix_accel/tiled_mlp.py`
- `tunix_accel/gemma3_tiled_mlp.py`
- `tests/test_tiled_mlp.py`
- `tests/test_gemma3_tiled_mlp.py`

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

For Gemma3-style gated MLPs, both `gate` and `up` projections produce large
token-by-intermediate tensors. A tiled implementation should trade extra
recompute or smaller GEMMs for lower activation residency.

## Verified Locally

Gemma-free JAX checks:

```bash
python -m pytest -q tests/test_tiled_mlp.py
```

Gemma3 integration checks:

```bash
python -m pytest -q tests/test_gemma3_tiled_mlp.py
```

Current combined result:

```text
9 passed
```

Covered checks:

- forward parity against dense MLP for `silu`, `gelu`, `gelu_approx`, and
  `relu`
- gradients with respect to hidden, gate, up, and down kernels
- JIT-compiled gradient parity
- simple dense-vs-tiled intermediate memory estimate helper
- Tunix Gemma3 `FeedForward.block` parity
- Tunix Gemma3 `remat_config=BLOCK` call-path parity
- Tunix Gemma3 default SFT loss parity on a tiny random model
- Qwix-LoRA safety behavior: fallback to the original MLP by default, strict
  error when fallback is disabled

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

For Gemma3 drop-in use:

```python
from tunix_accel import gemma3_tiled_mlp

gemma3_tiled_mlp.install(token_chunk=256)

trainer = peft_trainer.PeftTrainer(...)
trainer.train(...)

gemma3_tiled_mlp.uninstall()
```

The patch replaces `tunix.models.gemma3.model.FeedForward.block` process-wide.
That keeps Gemma3's original `FeedForward.__call__` path intact, including its
existing `remat_config=BLOCK` behavior.

When the package is installed, this Gemma3 replacement is also applied
automatically when `tunix.models.gemma3.model` is imported. Use these
environment variables to control the drop-in behavior:

```bash
export TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK=128
export TUNIX_ACCEL_DISABLE_TILED_MLP=1
```

For paired experiments, leave `TUNIX_ACCEL_DISABLE_TILED_MLP` unset for the
tiled run and set it to `1` for the Tunix default-MLP baseline.

## Next Milestones

1. Run TPU microbenchmarks:
   - dense MLP vs tiled MLP
   - CCE vs CCE + tiled MLP
   - context frontier and XLA planned HBM
2. Decide whether Pallas is needed. If XLA already lowers the tiled custom-VJP
   path well, Pallas may not be worth the added model-family complexity.
3. Add Qwix-LoRA-aware tiled projection support if LoRA runs become a primary
   target. Current behavior intentionally falls back to the original MLP when
   LoRA projection params are present.

## Known Risks

- Tiling may reduce memory but slow steps because large GEMMs become several
  smaller sequential GEMMs.
- Drop-in replacement is more model-family-sensitive than CCE. This workstream
  intentionally starts with Gemma3 only; other families may expose different MLP
  module layouts.
- Current tiled Gemma3 path is full-parameter/frozen-base only. Qwix-LoRA
  projection deltas fall back to the original MLP unless strict mode is enabled.
- XLA might already optimize some MLP patterns well enough that a handwritten
  Pallas kernel is not the first bottleneck.
