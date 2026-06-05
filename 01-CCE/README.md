# 01-CCE

This directory contains the final artifacts for the Cut Cross Entropy experiment.
The main report is now rebuilt around a complete Gemma3 270M rerun before any
larger-model transfer checks are considered.

## Contents

- `TECHNICAL_REPORT.md`: final narrative report with embedded plots.
- `REPRODUCE.md`: guide for reproducing the experiment family.
- `GEMMA3_270M_EXPERIMENT_DESIGN.md`: clean rerun design for rebuilding the CCE
  evidence chain around Gemma3 270M before scaling the claim outward.
- `run_gemma3_270m_cce_sweep.py`: local/TPU sweep runner used by the rerun.
- `remote_gemma3_270m_cce_worker.sh`: TPU VM profile wrapper.
- `collect_gemma3_270m_cce_results.py`: artifact collector and plot generator.
- `assets/`: final plots used by the report.
- `data/`: compact CSV/JSONL summaries retained from the experiments,
  including the Gemma4 base boundary rows.
- `references/`: background notes used to frame the benchmark story.

Raw TPU profiler traces, XLA dump directories, smoke runs, intermediate plots,
and training checkpoints were intentionally removed. They were useful while
debugging, but they are not needed to read the result or reproduce the setup.

The patch implementation itself lives outside this directory in `tunix_accel/`.

## Gemma3 270M Full Rerun

The current primary evidence package is:

- Data: `data/gemma3_270m_full_cce/`
- Figures:
  - `assets/gemma3_270m_cce_frontier.png`
  - `assets/gemma3_270m_cce_status_heatmap.png`
  - `assets/gemma3_270m_cce_tuning.png`
  - `assets/gemma3_270m_cce_quality.png`

All rows used Cloud TPU `v5litepod-1`, one chip, in `us-west4-a`.

The extracted `data/gemma3_270m_full_cce/raw/` directory is disposable and
should not be committed. Recreate it from `raw_artifacts/*.tar.gz` with:

```bash
python3 01-CCE/collect_gemma3_270m_cce_results.py
```

## Gemma4 Rows

The Gemma4 base boundary rows are folded into this directory rather than kept
as a separate workstream. The main CCE memory figure uses max per-chip HBM
pressure, with aggregate accounting retained in the report tables:

- Data: `data/gemma4_base_cce_tpu_l2048_b1.csv`
- Complete cross-workstream boundary table:
  `data/gemma4_base_tpu_l2048_b1_all_variants.csv`
- Figure: `assets/gemma3_gemma4_cce_per_chip_hbm.png`
