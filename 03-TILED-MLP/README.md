# 03-TILED-MLP

This directory contains the final artifacts for the Gemma3 tiled-MLP
experiment.

## Contents

- `TECHNICAL_REPORT.md`: final narrative report with embedded plots.
- `REPRODUCE.md`: guide for reproducing the experiment family.
- `assets/`: final plots used by the report.
- `data/`: compact CSV/JSON/Markdown summaries retained from the experiments.
- `data/raw/`: raw summary, history, translation, and XLA-report records for the
  final retained runs.
- `references/`: background notes used to frame the benchmark story.
- `run_gemma_training_benchmark.py`: TPU training runner for Default MLP vs
  Tiled MLP comparisons.
- `run_gemma3_tiled_mlp_parity.py`: same-model parity runner for Gemma3.

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

The current scope is deliberately **Gemma3-only**. The generic math kernel can
express other gated MLPs, but the drop-in patching surface is tied to Tunix
Gemma3's `FeedForward.block` layout.

## Headline Result

On Gemma3 4B LoRA, batch 1, max length 4096, TPU v5litepod-8:

| Variant | Status | XLA planned HBM, aggregate |
| --- | --- | ---: |
| Default MLP | compile OOM | 161.5 GiB |
| Tiled MLP | OK | 116.4 GiB |

At the same 2048 context pressure point, XLA planned HBM moved from 82.9 GiB to
56.5 GiB aggregate. In the 500-step OPUS100 EN-FR smoke run at length 2048,
runtime peak memory moved from 21.29 GiB to 17.92 GiB aggregate, while mean step
time increased by about 12.6%.

See `TECHNICAL_REPORT.md` for the full interpretation.

## Drop-In Controls

Installed environments automatically patch Tunix Gemma3 when
`tunix.models.gemma3.model` is imported.

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

Normal Tunix training code should not need that explicit call after the package
is installed.

## Verification

Local tests:

```bash
python -m pytest -q tests/test_tiled_mlp.py tests/test_gemma3_tiled_mlp.py
```

The retained TPU validation artifacts are:

- Memory/keypoint data: `data/gemma3_4b_context_keypoints.csv`
- 500-step validation data: `data/gemma3_4b_validation_summary.csv`
- Same-model parity data: `data/gemma3_4b_direct_parity.json`
- Final figures: `assets/gemma3_4b_context_boundary_memory.png` and
  `assets/gemma3_4b_validation_summary.png`
