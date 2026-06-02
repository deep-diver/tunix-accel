# 02-PACKING

This directory contains the final artifacts for the sequence-packing experiment.

## Contents

- `TECHNICAL_REPORT.md`: final narrative report with embedded plots.
- `REPRODUCE.md`: guide for reproducing the experiment family.
- `assets/`: final plots used by the report.
- `data/`: compact CSV/JSON/Markdown summaries retained from the experiments.
- `references/`: background notes used to frame the benchmark story.

Raw TPU run directories, per-step histories, checkpoint directories,
intermediate smoke outputs, and old result folders were intentionally removed.
They were useful while debugging, but they are not needed to read the result or
reproduce the setup.

The patch implementation itself lives outside this directory in `tunix_accel/`.
