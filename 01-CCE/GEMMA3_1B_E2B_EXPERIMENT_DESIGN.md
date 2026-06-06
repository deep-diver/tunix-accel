# Gemma3 1B and Gemma4 E2B CCE Experiment Design

This design extends the Gemma3 270M CCE evidence package to two larger-but-still
affordable models: Gemma3 1B IT and Gemma4 E2B. The goal is not to run every
possible mesh on every future model. The goal is to learn which CCE rules found
on 270M survive one step up in scale before using sparse checks on 4B/E4B/12B/27B.

## Mesh Policy

Primary mesh:

| Model | TPU | Chips | Mesh | Reason |
| --- | --- | ---: | --- | --- |
| Gemma3 1B IT | `v5litepod-4` | 4 | `fsdp=4,tp=1` | Enough memory for LoRA SFT while keeping the baseline comparison simple. |
| Gemma4 E2B | `v5litepod-4` | 4 | `fsdp=4,tp=1` | Same chip count and FSDP-only layout as 1B, with Hugging Face Gemma4 loading. |

Mesh sensitivity:

| Mesh | Role |
| --- | --- |
| `fsdp=4,tp=1` | Headline result and practical default. |
| `fsdp=2,tp=2` | Mixed FSDP/TP stress test; this is where 270M exposed CCE chunk granularity. |
| `fsdp=1,tp=4` | TP-heavy stress test; useful for compatibility and communication sensitivity, not expected to be the fastest small-model layout. |

We do not use different chip counts for 1B and E2B in the exhaustive tier. Keeping
both on four chips makes the interpretation cleaner: model size changes, TPU and
headline mesh stay fixed.

## Required Profiles

Run these for each model size with `MODEL_SIZE=1b` and `MODEL_SIZE=e2b`:

| Profile | Mesh | Purpose |
| --- | --- | --- |
| `fourchip-frontier-fsdp4` | `fsdp=4,tp=1` | Primary batch/context frontier. |
| `fourchip-frontier-2x2` | `fsdp=2,tp=2` | Mixed-mesh frontier transfer check. |
| `fourchip-frontier-tp4` | `fsdp=1,tp=4` | TP-heavy frontier transfer check. |
| `mesh-fsdp4-repeat` | `fsdp=4,tp=1` | Repeated same-shape timing. |
| `mesh-2x2-repeat` | `fsdp=2,tp=2` | Repeated mixed-mesh timing and outlier detection. |
| `mesh-tp4-repeat` | `fsdp=1,tp=4` | Repeated TP-heavy timing. |
| `fourchip-chunk-fsdp4` | `fsdp=4,tp=1` | Primary mesh chunk grid at b16/L512. |
| `fourchip-chunk-2x2` | `fsdp=2,tp=2` | Mixed mesh chunk grid and axis-ablation source. |
| `outlier-hlo` | all three meshes | Coarse HLO text scan if a timing outlier appears. |
| `fourchip-quality-fsdp4-default` | `fsdp=4,tp=1` | 1,000-step OPUS100 Default CE loss parity. |
| `fourchip-quality-fsdp4-cce` | `fsdp=4,tp=1` | 1,000-step OPUS100 CCE loss parity. |

The profiles intentionally match the final 270M four-chip evidence layer. The
older 270M single-chip-only profiles are not repeated because 1B/E2B are meant to
establish the four-chip practical scaling law.

## Expected Lessons

The combined 270M/1B/E2B set should answer:

1. Does the CCE frontier gain persist as model size grows while TPU/mesh stays fixed?
2. Does FSDP-only remain the clearest headline mesh?
3. Does the `fsdp=2,tp=2` chunk-granularity outlier repeat beyond 270M?
4. Does `tpu_large_chunks` remain a good TPU preset, or does it increase memory on
   larger models?
5. Does same-shape CCE overhead shrink, stay flat, or grow with model size?

Larger models should then be transfer checks, not another exhaustive Cartesian
product.
