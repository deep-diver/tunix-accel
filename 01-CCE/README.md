# 01-CCE

This directory contains the final artifacts for the CCE experiment.

## Contents

- `TECHNICAL_REPORT.md`: final narrative report with embedded plots.
- `REPRODUCE.md`: guide for reproducing the experiment family.
- `assets/`: final plots used by the report.
- `data/`: compact CSV/JSONL summaries retained from the experiments,
  including the Gemma4 base boundary rows.
- `references/`: background notes used to frame the benchmark story.

Raw TPU profiler traces, XLA dump directories, smoke runs, intermediate plots,
and training checkpoints were intentionally removed. They were useful while
debugging, but they are not needed to read the result or reproduce the setup.

The patch implementation itself lives outside this directory in `tunix_accel/`.

## Gemma4 Rows

The Gemma4 base boundary rows are folded into this directory rather than kept
as a separate workstream. The main CCE memory figure uses max per-chip HBM
pressure, with aggregate accounting retained in the report tables:

- Data: `data/gemma4_base_cce_tpu_l2048_b1.csv`
- Complete cross-workstream boundary table:
  `data/gemma4_base_tpu_l2048_b1_all_variants.csv`
- Figure: `assets/gemma3_gemma4_cce_per_chip_hbm.png`
