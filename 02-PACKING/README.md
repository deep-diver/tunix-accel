# 02-PACKING

This directory contains the final artifacts for the sequence-packing experiment.

## Contents

- `TECHNICAL_REPORT.md`: final narrative report with embedded plots.
- `REPRODUCE.md`: guide for reproducing the experiment family.
- `assets/`: final plots used by the report.
- `data/`: compact CSV/JSON/Markdown summaries retained from the experiments,
  including the Gemma4 base negative-control rows.
- `references/`: background notes used to frame the benchmark story.

Raw TPU run directories, per-step histories, checkpoint directories,
intermediate smoke outputs, and old result folders were intentionally removed.
They were useful while debugging, but they are not needed to read the result or
reproduce the setup.

The patch implementation itself lives outside this directory in `tunix_accel/`.

## Gemma4 Rows

The Gemma4 base boundary rows are retained here as a negative control for the
memory story and folded into the main throughput/density figure. Packing
improves useful-token density, but it does not change the compile-HBM boundary
for a fixed already-full model shape.

- Data: `data/gemma4_base_packing_tpu_l2048_b1.csv`
- Figure: `assets/gemma3_1b_4b_throughput_and_density.png`
