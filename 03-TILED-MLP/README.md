# 03-TILED-MLP

This directory contains the final artifacts for the Tiled MLP experiment. The
primary TPU result is Gemma3 4B, with Gemma4 base boundary rows retained in the
same workstream and folded into the main memory-boundary figure.

## Contents

- `TECHNICAL_REPORT.md`: final narrative report with embedded plots.
- `REPRODUCE.md`: guide for reproducing the experiment family.
- `assets/`: final plots used by the report.
- `data/`: compact CSV/JSON/Markdown summaries retained from the experiments.
- `data/raw/`: raw summary, history, translation, and XLA-report records for the
  final retained Gemma3 runs.
- `references/`: background notes used to frame the benchmark story.
- `run_gemma_training_benchmark.py`: TPU training runner for Default MLP vs
  Tiled MLP comparisons.
- `run_gemma3_tiled_mlp_parity.py`: same-model parity runner for Gemma3.
- `../tools/run_gemma4_base_benchmark.py`: shared Gemma4 base boundary runner
  used by the 01-04 boundary rows.

Intermediate result folders, old plot attempts, checkpoint directories, and TPU
logs are not part of the final package. They were useful while debugging, but
the retained CSV/JSON records are enough to read the result and reproduce the
setup.

The patch implementation itself lives outside this directory in `tunix_accel/`.

## Summary

Tiled MLP targets the large gated-MLP intermediate in Gemma3 decoder blocks:

```text
output = (activation(x @ gate) * (x @ up)) @ down
```

The implementation streams the token dimension and uses a custom VJP. Backward
recomputes each token tile's gate/up/intermediate activations instead of relying
on one full resident `[tokens, intermediate_dim]` activation tensor.

The supported drop-in scope is Gemma3 and Gemma4 through explicit
model-family adapters. The generic math kernel can express other gated MLPs,
but each new family should get its own adapter and parity tests.

## Headline Result

On Gemma3 4B LoRA, batch 1, max length 4096, TPU v5litepod-8:

| Variant | Status | Max/chip planned HBM |
| --- | --- | ---: |
| Default MLP | compile OOM | 20.19 GiB/chip |
| Tiled MLP | OK | 14.55 GiB/chip |

At the same 2048 context pressure point, XLA planned HBM moved from
10.36 GiB/chip to 7.06 GiB/chip. The equivalent 8-chip accounting is
82.9 GiB to 56.5 GiB, but the per-chip value is the fit/OOM criterion. In the
500-step OPUS100 EN-FR smoke run at length 2048, runtime peak memory moved from
21.29 GiB to 17.92 GiB aggregate, while mean step time increased by about 12.6%.

See `TECHNICAL_REPORT.md` for the full interpretation.

Gemma4 base boundary rows, batch 1, max length 2048, LoRA rank 16:

| Model | TPU | Default MLP | Tiled MLP |
| --- | --- | --- | --- |
| Gemma4 E2B | v5litepod-4, 4 chips | compile OOM | OK |
| Gemma4 E4B | v5litepod-8, 8 chips | compile OOM | OK |

## Drop-In Controls

Installed environments automatically patch Tunix Gemma3 and Gemma4 when the
corresponding Tunix model module is imported.

```bash
export TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK=128
export TUNIX_ACCEL_TILED_MLP_LORA_ALPHA=32.0
export TUNIX_ACCEL_DISABLE_TILED_MLP=1
export TUNIX_ACCEL_DISABLE_CE=1
```

Use `TUNIX_ACCEL_DISABLE_TILED_MLP=1` for the Default MLP baseline while keeping
other autopatches available. Use `TUNIX_ACCEL_DISABLE_CE=1` when isolating MLP
experiments from the Cut Cross Entropy patch.

For notebooks or scoped tests:

```python
from tunix_accel import gemma3_tiled_mlp

gemma3_tiled_mlp.install(token_chunk=128, lora_alpha=32.0)
```

For Gemma4:

```python
from tunix_accel import gemma4_tiled_mlp

gemma4_tiled_mlp.install(token_chunk=128, lora_alpha=32.0)
```

Normal Tunix training code should not need that explicit call after the package
is installed.

## Verification

Local tests:

```bash
python -m pytest -q \
  tests/test_tiled_mlp.py \
  tests/test_gemma3_tiled_mlp.py \
  tests/test_gemma4_tiled_mlp.py
```

The retained TPU validation artifacts are:

- Memory/keypoint data: `data/gemma3_4b_context_keypoints.csv`
- 500-step validation data: `data/gemma3_4b_validation_summary.csv`
- Same-model parity data: `data/gemma3_4b_direct_parity.json`
- Gemma4 boundary data: `data/gemma4_base_tiled_mlp_tpu_l2048_b1.csv`
- Gemma4 local parity data: `data/gemma4_local_parity_summary.csv`
- Final figures: `assets/gemma3_4b_context_boundary_memory.png` and
  `assets/gemma3_4b_validation_summary.png`
- Main memory-boundary figure with Gemma4 rows folded in:
  `assets/gemma3_4b_context_boundary_memory.png`
