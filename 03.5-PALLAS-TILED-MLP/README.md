# 03.5-PALLAS-TILED-MLP

This directory contains the final artifacts for the Pallas-backend follow-up to
`03-TILED-MLP`.

The experiment asks a narrow question: after the Gemma3 tiled MLP is already
working with ordinary XLA matmuls, does lowering the tile matmuls to Pallas on
TPU improve the memory/time story?

## Contents

- `TECHNICAL_REPORT.md`: final interpretation with embedded plots.
- `REPRODUCE.md`: guide for reproducing the Pallas-backend experiment family.
- `assets/`: final plots used by the report.
- `data/`: compact CSV/JSON summaries retained from the experiments.
- `data/raw/`: raw summary, history, translation, parity, and parsed XLA-report
  records for the final retained runs.

The patch implementation itself lives outside this directory in `tunix_accel/`.

## Summary

The Pallas backend is opt-in:

```bash
export TUNIX_ACCEL_TILED_MLP_BACKEND=pallas
```

The default tiled MLP backend remains `xla`.

On Gemma3 4B LoRA, batch 1, max length 4096, TPU v5litepod-8:

| Metric | 03 XLA tiled | 03.5 Pallas tiled | Delta |
| --- | ---: | ---: | ---: |
| XLA planned HBM, aggregate | 116.4 GiB | 112.9 GiB | -3.0% |
| Mean step time, excl. first | 0.634s | 0.668s | +5.4% |

The Pallas backend also passed same-model loss parity and gradient-norm parity
on Gemma3 4B:

| Check | Result |
| --- | ---: |
| Forward loss diff | 0 |
| LoRA grad norm relative diff | 0.00998% |
| LoRA grad RMS abs diff | 0.000935 |

The useful conclusion is sober: Pallas integration works, and it composes with
the Cut Cross Entropy patch, but this first Pallas lowering is not yet a better
replacement for the XLA tiled MLP backend. It gives a small planned-HBM win at
L4096, but it is slower and does not improve the 500-step runtime memory peak.

## Drop-In Controls

Installed environments automatically patch Tunix Gemma3 when
`tunix.models.gemma3.model` is imported.

```bash
export TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK=128
export TUNIX_ACCEL_TILED_MLP_BACKEND=xla
export TUNIX_ACCEL_TILED_MLP_BACKEND=pallas
export TUNIX_ACCEL_DISABLE_TILED_MLP=1
export TUNIX_ACCEL_DISABLE_CE=1
```

Use `TUNIX_ACCEL_TILED_MLP_BACKEND=pallas` to opt into the Pallas matmul path.
Use `TUNIX_ACCEL_DISABLE_TILED_MLP=1` for the Default MLP baseline. Use
`TUNIX_ACCEL_DISABLE_CE=1` when isolating MLP experiments from Cut Cross
Entropy.

For notebooks or scoped tests:

```python
from tunix_accel import gemma3_tiled_mlp

gemma3_tiled_mlp.install(token_chunk=128, matmul_backend="pallas")
```

Normal Tunix training code should not need that explicit call after the package
is installed.

## Verification

Local tests:

```bash
python -m pytest -q tests/test_tiled_mlp.py tests/test_gemma3_tiled_mlp.py tests/test_autopatch.py
```

Retained TPU validation artifacts:

- Context data: `data/gemma3_4b_pallas_vs_xla_context.csv`
- 500-step smoke data: `data/gemma3_4b_pallas_vs_xla_validation_summary.csv`
- Same-model parity data: `data/gemma3_4b_pallas_direct_parity.json`
- Delta summary: `data/gemma3_4b_pallas_vs_03_delta_summary.csv`
- Final figures:
  - `assets/gemma3_4b_pallas_vs_xla_context.png`
  - `assets/gemma3_4b_pallas_validation_smoke.png`
  - `assets/gemma3_4b_pallas_cce_composition.png`
