# 06-LORA-FA

This directory tracks the LoRA-FA workstream for Tunix/Qwix LoRA training.

LoRA-FA freezes the LoRA projection-down matrix A and trains only projection-up
matrix B. The corrected variant applies a small rank-space gradient correction
to B so the induced low-rank update better approximates the full fine-tuning
gradient.

## Contents

- `TECHNICAL_REPORT.md`: final narrative report, populated after TPU runs.
- `REPRODUCE.md`: command guide for reproducing the experiment family.
- `METRICS.md`: shared metric schema and anomaly gates.
- `assets/`: final figures used by the report.
- `data/`: compact CSV/JSON summaries retained from experiments.
- `references/`: background notes used to frame the benchmark story.
- `results/`: temporary local staging area for retained summaries during this
  branch.
- `run_lora_fa_matrix.py`: variant matrix runner that reuses the shared Gemma
  training benchmark with LoRA-FA-specific env controls.

The implementation lives in `../tunix_accel/lora_fa.py`.

## Validation Matrix

All model rows must emit the same metric schema so trends are comparable.

| Family | Model | First gate | Scale run |
| --- | --- | --- | --- |
| Gemma3 | 270M IT | required | required |
| Gemma3 | 1B IT | after 270M passes | required |
| Gemma3 | 4B IT | after 270M passes | required |
| Gemma3 | 12B IT | after 270M passes | required |
| Gemma3 | 27B IT | after 270M passes | required |
| Gemma4 | E2B base | after 270M passes | required |
| Gemma4 | E4B base | after 270M passes | required |

The user-facing shorthand "13B" maps to the available Gemma3 12B checkpoint in
this repository's prior experiment naming.

## Variant Matrix

Start every model with the same minimum variants:

| Variant | A trainable | B correction | Intended comparison |
| --- | --- | --- | --- |
| standard_lora_r16 | yes | no | current baseline |
| standard_lora_r32 | yes | no | same-rank baseline for LoRA-FA r32 |
| standard_lora_r64 | yes | no | same-rank baseline for LoRA-FA r64 |
| freeze_a_r16 | no | no | isolate freeze-A effect |
| lorafa_r16 | no | yes | corrected same-rank comparison |
| lorafa_r32 | no | yes | rank increase without large memory growth |
| lorafa_r64 | no | yes | stronger rank stress, if compile permits |

Optional extensions:

- Composition rows with CCE, Tiled MLP, and Packing after the standalone story
  is clear.

## Gate Rule

Run Gemma3 270M first. Do not scale to the larger matrix until 270M passes:

- A gradients are absent from the compiled training graph.
- A values remain unchanged after optimizer steps.
- B values update.
- Corrected LoRA-FA has finite loss and no gradient explosions.
- Short-run loss is not obviously worse than standard LoRA.
- Step time does not regress enough to erase the memory story.

If any larger model shows an unexpected trend, stop the sweep, explain the
cause, patch the implementation or runner, and rerun from Gemma3 270M before
continuing.

## Current Local Status

Implemented:

- B-only NNX filter for Qwix LoRA parameters.
- Optional corrected B-gradient transform.
- Process-local Tunix `PeftTrainer` patch that rebuilds optimizer state for B
  only and uses the same B-only diff state in `_train_step`.

Verified locally:

```bash
python -m pytest -q tests/test_lora_fa.py
```

Dry-run the 270M gate commands:

```bash
python 06-LORA-FA/run_lora_fa_matrix.py --models gemma3-270m --dry-run
```
