# Reproducing The LoRA-FA Experiments

This guide will be filled in as the TPU runs are consolidated. The experiment
order is intentionally gated.

## 1. Local Mechanics

```bash
python -m pytest -q tests/test_lora_fa.py
```

This verifies B-only gradients, B-gradient correction, and no A update in a tiny
Gemma3/Qwix LoRA train step.

## 2. Gemma3 270M Gate

Run Gemma3 270M before any larger model. Required variants:

```text
standard_lora_r16
standard_lora_r32
standard_lora_r64
freeze_a_r16
lorafa_r16
lorafa_r32
lorafa_r64
```

Required outputs:

```text
summary.csv
history.csv
memory_report_paths.json
lora_fa_internal_metrics.json
quality_samples.md
```

Command scaffold:

```bash
python 06-LORA-FA/run_lora_fa_matrix.py \
  --models gemma3-270m \
  --variants standard_lora_r16,standard_lora_r32,standard_lora_r64,freeze_a_r16,lorafa_r16,lorafa_r32,lorafa_r64 \
  --batch-size 16 \
  --max-length 512 \
  --max-steps 50 \
  --outdir 06-LORA-FA/results/gemma3-270m-gate
```

Only continue if the gate conditions in `METRICS.md` pass.

## 3. Scale Sweep

After the 270M gate passes, run the same schema on:

```text
Gemma3 1B
Gemma3 4B
Gemma3 12B
Gemma3 27B
Gemma4 E2B
Gemma4 E4B
```

Use the smallest TPU topology that gives a meaningful comparison for that model,
but record TPU type, chip count, and mesh axes on every row.

Install Tunix from this repository's `requirements.txt` rather than pinning
`google-tunix==0.1.6`; the latter lacks `tunix.models.gemma4` and cannot run the
Gemma4 rows.

## 4. Stop-And-Rerun Rule

If a larger model shows an unexpected trend, stop the sweep, diagnose, patch,
and rerun from Gemma3 270M before continuing.
