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
- `collect_gemma3_270m_mesh_results.py`: four-chip mesh compatibility collector.
- `collect_gemma3_270m_mesh_repeat_results.py`: repeated four-chip mesh timing
  collector.
- `collect_gemma3_270m_4chip_frontier_results.py`: extended four-chip context
  frontier collector.
- `collect_gemma3_270m_outlier_hlo_results.py`: compact HLO text-scan collector
  for the mixed-mesh throughput outlier.
- `collect_gemma3_270m_4chip_chunk_results.py`: mixed-mesh CCE chunk tuning
  collector.
- `collect_gemma3_270m_4chip_quality_results.py`: four-chip OPUS100 parity
  collector.
- `assets/`: final plots used by the report.
- `data/`: compact CSV/JSONL summaries retained from the experiments,
  including the Gemma4 base boundary rows.
- `references/`: background notes used to frame the benchmark story.

Raw TPU profiler traces, XLA dump directories, smoke runs, intermediate plots,
and training checkpoints were intentionally removed. They were useful while
debugging, but they are not needed to read the result or reproduce the setup.

The patch implementation itself lives outside this directory in `tunix_accel/`.

## Gemma3 270M Evidence Package

The current primary evidence package is:

- Data: `data/gemma3_270m_full_cce/`
- Figures:
  - `assets/gemma3_270m_cce_frontier.png`
  - `assets/gemma3_270m_cce_status_heatmap.png`
  - `assets/gemma3_270m_cce_tuning.png`
  - `assets/gemma3_270m_cce_quality.png`
  - `assets/gemma3_270m_cce_mesh_2x2_repeat.png`
  - `assets/gemma3_270m_cce_4chip_frontier.png`
  - `assets/gemma3_270m_cce_outlier_hlo.png`
  - `assets/gemma3_270m_cce_4chip_chunk_tuning.png`
  - `assets/gemma3_270m_cce_4chip_chunk_axis_ablation.png`
  - `assets/gemma3_270m_cce_4chip_quality.png`

The primary rerun rows used Cloud TPU `v5litepod-1`, one chip, in
`us-west4-a`. The mesh generalization check used `v5litepod-4`, four chips, in
the same zone, with `fsdp=4,tp=1`, `fsdp=2,tp=2`, and `fsdp=1,tp=4`.

Mesh check data:

- `data/gemma3_270m_mesh_cce/run_manifest.csv`
- `data/gemma3_270m_mesh_cce/mesh_runs.csv`
- `data/gemma3_270m_mesh_cce/mesh_summary.csv`
- `data/gemma3_270m_mesh_cce/matched_memory.csv`
- `data/gemma3_270m_mesh_cce_repeat/repeat_summary.csv`
- `data/gemma3_270m_4chip_frontier/frontier_summary.csv`
- `data/gemma3_270m_outlier_hlo/hlo_op_counts.csv`
- `data/gemma3_270m_4chip_chunk/chunk_summary.csv`
- `data/gemma3_270m_4chip_chunk/chunk_axis_ablation.csv`
- `data/gemma3_270m_4chip_quality/training_summary.csv`

The extracted `data/gemma3_270m_full_cce/raw/` directory is disposable and
should not be committed. Recreate it from `raw_artifacts/*.tar.gz` with:

```bash
python3 01-CCE/collect_gemma3_270m_cce_results.py
python3 01-CCE/collect_gemma3_270m_mesh_results.py
python3 01-CCE/collect_gemma3_270m_mesh_repeat_results.py
python3 01-CCE/collect_gemma3_270m_4chip_frontier_results.py
python3 01-CCE/collect_gemma3_270m_outlier_hlo_results.py
python3 01-CCE/collect_gemma3_270m_4chip_chunk_results.py
python3 01-CCE/collect_gemma3_270m_4chip_quality_results.py
```

## Gemma4 Rows

The Gemma4 base boundary rows are retained as transfer-check data from the
earlier integrated report. They are not part of the current Gemma3 270M evidence
claim, which is intentionally narrower:

- Data: `data/gemma4_base_cce_tpu_l2048_b1.csv`
- Complete cross-workstream boundary table:
  `data/gemma4_base_tpu_l2048_b1_all_variants.csv`
- Figure: `assets/gemma3_gemma4_cce_per_chip_hbm.png`
