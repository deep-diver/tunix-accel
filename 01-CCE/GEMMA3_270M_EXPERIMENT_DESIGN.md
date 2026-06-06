# Gemma3 270M CCE Experiment Design

## Execution Status

This design has been executed on Cloud TPU `v5litepod-1`, one chip, in
`us-west4-a`. A follow-up mesh generalization check has also been executed on
Cloud TPU `v5litepod-4`, four chips, in the same zone. The retained results live
under:

- `01-CCE/data/gemma3_270m_full_cce/`
- `01-CCE/data/gemma3_270m_mesh_cce/`
- `01-CCE/data/gemma3_270m_mesh_cce_repeat/`
- `01-CCE/data/gemma3_270m_4chip_frontier/`
- `01-CCE/data/gemma3_270m_4chip_chunk/`
- `01-CCE/data/gemma3_270m_4chip_quality/`
- `01-CCE/assets/gemma3_270m_cce_frontier.png`
- `01-CCE/assets/gemma3_270m_cce_status_heatmap.png`
- `01-CCE/assets/gemma3_270m_cce_tuning.png`
- `01-CCE/assets/gemma3_270m_cce_quality.png`
- `01-CCE/assets/gemma3_270m_cce_mesh_2x2_repeat.png`
- `01-CCE/assets/gemma3_270m_cce_4chip_frontier.png`
- `01-CCE/assets/gemma3_270m_cce_4chip_chunk_tuning.png`
- `01-CCE/assets/gemma3_270m_cce_4chip_chunk_axis_ablation.png`
- `01-CCE/assets/gemma3_270m_cce_4chip_quality.png`

The final narrative is `01-CCE/TECHNICAL_REPORT.md`, and reproduction steps are
in `01-CCE/REPRODUCE.md`.

This document defines the next clean 01-CCE experiment family. The goal is to
rebuild the CCE report around one complete model first: Gemma3 270M LoRA SFT on
TPU. Larger Gemma3 and Gemma4 rows should later be used as transfer checks, not
as loosely attached one-off additions.

## 1. Main Question

Does Cut Cross Entropy remove the full-vocab loss-logits memory wall in
JAX/Tunix SFT, and does the resulting memory gain preserve training behavior?

The 270M run family should answer this before we scale out:

- exactness: dense CE and CCE produce matching loss/gradients,
- frontier: CCE moves the feasible batch/context boundary,
- cost: CCE's same-shape step-time tradeoff is measured,
- quality: real EN-FR LoRA SFT keeps loss and generation sanity metrics in the
  same band,
- attribution: profiler/XLA evidence shows that the saved memory is actually
  the CE/logits path, not an unrelated artifact.

## 2. Fixed Scope

| Field | Value |
| --- | --- |
| Primary model | `google/gemma-3-270m-it` |
| Training mode | Tunix PEFT/LoRA |
| Primary LoRA rank | 16 |
| TPU | Cloud TPU v5e `v5litepod-1`, 1 chip |
| Mesh follow-up TPU | Cloud TPU v5e `v5litepod-4`, 4 chips |
| Main dataset for systems sweeps | deterministic synthetic SFT records |
| Main dataset for quality | OPUS100 EN-FR |
| Compared variants | Default CE vs CCE only |
| Other patches | disabled unless a row explicitly says otherwise |
| Primary memory axis | max per-chip XLA buffer-assignment planned HBM |

This 01-CCE design should not mix in Packing, Tiled MLP, Activation Offload, or
Splash Attention. Those belong to their own reports or later composition
sections.

## 3. Retention Rules

The previous 01-CCE report kept enough compact summaries to redraw most figures
but did not keep every source table. The rerun should keep a complete evidence
chain.

Retain these files under `01-CCE/data/gemma3_270m_full_cce/`:

| File | Purpose |
| --- | --- |
| `run_manifest.csv` | one row per run: command, commit, TPU, env vars, seed, dataset, status |
| `parity_summary.csv` | dense CE vs CCE loss/gradient/update parity |
| `kernel_matrix.csv` | isolated CE/logits microbenchmarks |
| `chunk_tuning.csv` | CCE token/vocab chunk sweep |
| `frontier_runs.csv` | every `(batch, context, variant)` status and memory row |
| `frontier_summary.csv` | max completed context per batch and variant |
| `pressure_points.csv` | selected representative shapes with profiler metadata |
| `training_history.csv` | raw per-step or per-log-interval train loss and timing |
| `training_summary.csv` | final loss, eval loss, memory, step time, token throughput |
| `generation_metrics.csv` | BLEU/chrF sanity metrics and eval sample count |
| `generation_samples.jsonl` | fixed side-by-side samples |
| `profile_summary.csv` | selected TPU profiler/XLA summaries |
| `oom_events.csv` | OOM type, failing shape, extracted error line, memory estimate |

Raw TPU profiler traces and full XLA dump trees can be large. Keep the compact
parsed summaries in git. Store raw traces outside git only if needed, but record
their path or GCS URI in `run_manifest.csv`.

## 4. Experiment Suite A: Numerical Parity

Purpose: prove that the replacement computes the same objective before looking
at memory.

Rows:

| Case | Device | Shape | What to compare |
| --- | --- | --- | --- |
| isolated dense CE vs CCE | CPU or TPU | small deterministic tensors | loss, hidden grad, LM-head grad |
| Tunix tiny LoRA path | CPU or TPU | tiny Gemma-like config | loss, LoRA gradients |
| Gemma3 270M one-step LoRA | TPU | b1/L128 and b4/L512 | loss, LoRA gradient norms, one optimizer update |
| trainable-head sanity | CPU or TPU | tiny config | full-FT loss/grad path, if supported |

Record absolute and relative differences. Suggested pass criteria:

- loss absolute difference <= `5e-3` for bf16 TPU rows,
- LoRA-gradient max absolute difference <= `5e-3` or explain the tolerance,
- one optimizer step produces matching train loss on the next batch.

## 5. Experiment Suite B: Isolated Kernel And Chunk Tuning

Purpose: isolate the loss-logits tensor before involving full Tunix training.

Use Gemma3 270M-like dimensions:

| Dimension | Value |
| --- | --- |
| hidden size | 640 |
| vocab size | Gemma tokenizer vocab used by model |
| batch/context shapes | selected from frontier grid |

Benchmarks:

- Default dense CE,
- CCE with token chunks `[64, 128, 256, 512]`,
- CCE with vocab chunks `[4096, 8192, 16384, 32768]`.

Metrics:

- compile status,
- compile plus first-run time,
- steady run time if available,
- largest temporary estimate,
- XLA planned HBM,
- failure mode.

Output plots:

- memory vs shape for dense CE and CCE,
- CCE chunk-size Pareto chart: memory vs step time.

## 6. Experiment Suite C: 270M Frontier Sweep

Purpose: produce the main Unsloth-style result: longer feasible context at the
same TPU size.

Primary grid:

| Axis | Values |
| --- | --- |
| batch size | `1, 2, 4, 8, 16, 32, 64, 128` |
| context length | `256, 512, 1024, 2048, 4096, 8192, 16384, 32768` |
| variant | Default CE, CCE |
| LoRA rank | 16 |
| steps | 1 compile/frontier step plus optional 3 timed steps for passing rows |

Stopping rule:

- For each `(batch, variant)`, continue increasing context until that variant
  fails. If Default CE fails, keep testing CCE until CCE also fails.
- Do not stop the whole batch row at the Default failure point.

Metrics per row:

- `ok`, `compile_oom`, `runtime_oom`, or `skipped_after_known_failure`,
- max per-chip XLA planned HBM,
- aggregate accounting as secondary metadata,
- compile time,
- first-step time,
- mean timed step excluding first step,
- runtime memory snapshot if available,
- path to XLA memory report or parsed OOM line.

Required plots:

- max completed context by batch, Default CE vs CCE,
- max per-chip XLA planned HBM vs context, with one panel per batch group,
- heatmap of pass/fail status over batch and context,
- frontier gain table.

## 7. Experiment Suite D: Rank And Shape Sensitivity

Purpose: make sure the 270M story is truly CE/logits-driven and not an accident
of the default LoRA rank.

Reduced grid:

| Axis | Values |
| --- | --- |
| LoRA rank | `4, 16, 64` |
| batch size | `8, 16, 32, 64` |
| context length | `512, 1024, 2048, 4096` |
| variant | Default CE, CCE |

Expected interpretation:

- CCE benefit should mainly track `batch * context * vocab`.
- LoRA rank can move total memory, but should not erase the CE-specific pattern.

Required output:

- frontier table by rank,
- memory-delta chart at matched shapes.

## 8. Experiment Suite E: Representative Pressure Points

Purpose: keep the report from becoming only a huge sweep. Pick a few shapes that
tell the story cleanly and profile them deeply.

Candidate rows:

| Label | Shape | Reason |
| --- | --- | --- |
| easy parity shape | b16/L512 | both variants fit; used for quality run |
| first boundary | b16/L1024 or b16/L2048 | Default CE fails, CCE should fit |
| high-pressure CCE | b16/L4096 or b32/L2048 | CCE near frontier |
| CCE failure | discovered by sweep | shows the new wall |

For each representative row, collect:

- XLA memory report,
- TPU profiler trace summary,
- step-time breakdown if available,
- top buffer classes or largest buffers,
- same shape Default/CCE comparison when both compile.

## 9. Experiment Suite F: Multi-Chip Mesh Generalization

Purpose: verify that CCE is not only valid on one-chip runs. This suite should
stay synthetic and short; it is a compatibility and frontier check, not another
quality run.

Executed grid:

| Axis | Values |
| --- | --- |
| TPU | `v5litepod-4`, 4 chips |
| Meshes | `fsdp=4,tp=1`, `fsdp=2,tp=2`, `fsdp=1,tp=4` |
| Batch size | `16, 32, 64` |
| Context length | `512, 1024, 2048` |
| Variant | Default CE, CCE |
| LoRA rank | 16 |
| Steps | 3 timed synthetic SFT steps for passing rows |

Required interpretation:

- CCE should compile and run under FSDP-only, mixed FSDP/TP, and TP-only meshes.
- Report max per-chip XLA planned HBM, not only aggregate accounting.
- Treat TP-heavy timing as a separate throughput caveat, because mesh choice can
  dominate the step-time result for a small model.

Required output:

- frontier by mesh/batch,
- matched passing shape memory reduction,
- a short timing caveat table or paragraph.

## 10. Experiment Suite G: Real EN-FR Training Parity

Purpose: verify that the systems win does not come from changing training
behavior.

Same-shape A/B:

| Field | Value |
| --- | --- |
| Model | `google/gemma-3-270m-it` |
| Dataset | OPUS100 EN-FR |
| Variants | Default CE, CCE |
| Batch | 16 |
| Max length | 512 |
| LoRA rank | 16 |
| LR | `2e-4` |
| Steps | 5000 minimum; 10000 if time budget allows |
| Eval examples | at least 128, preferably 256 |
| Generation samples | fixed 16-32 examples |

Keep raw training history. The previous report had the plotted loss curve but
not the source history table; the rerun must fix that.

Metrics:

- raw train loss history,
- smoothed train loss,
- eval loss,
- BLEU and chrF as sanity metrics,
- generation samples,
- same-batch step time,
- tokens/sec,
- XLA planned HBM,
- runtime peak if available.

Capacity-enabled CCE run:

- choose one CCE-only feasible shape from the frontier, such as b32/L512,
  b64/L512, or b16/L2048,
- train for the same total token budget as the same-shape A/B,
- report wall-clock/token tradeoff separately from quality parity.

This row should answer: "If CCE enables a larger feasible batch/context, can
that capacity offset the slower same-shape step?"

## 11. Experiment Suite H: Profiler Attribution

Purpose: make the report credible to readers who ask whether memory was saved
in the intended place.

Selected profiler rows:

- Default CE b16/L512,
- CCE b16/L512,
- Default CE first failing shape,
- CCE matched shape that fits,
- CCE near-frontier shape.

Summaries to extract:

- XLA planned HBM peak,
- top 20 buffers by size,
- whether full `[tokens, vocab]` logits-like buffers appear,
- train-step duration,
- compile time,
- host/device transfer red flags.

Required plot:

- top-buffer comparison for Default CE vs CCE at one matched passing shape,
- step-time vs memory scatter for representative rows.

## 12. Transfer Checks After 270M Is Complete

Only after the 270M report is internally complete, run reduced transfer checks:

| Model | Minimum check |
| --- | --- |
| Gemma3 1B | same frontier protocol, fewer batch rows if needed |
| Gemma3 4B | boundary-focused frontier plus representative profile |
| Gemma3 12B/27B | large-model boundary rows from the same question set |
| Gemma4 E2B/E4B | same protocol if model runner supports it; otherwise keep out of the main CCE claim |

Transfer rows should answer "does the 270M pattern transfer?" They should not
define the main message.

## 13. Report Redesign

The final 01-CCE report should be rebuilt around this order:

1. Question and mechanism,
2. exactness/parity,
3. Gemma3 270M frontier sweep,
4. 270M pressure points and profiler attribution,
5. 270M multi-chip mesh generalization,
6. 270M real EN-FR training parity,
7. transfer checks to larger Gemma3 and Gemma4,
8. tradeoffs and limits.

The hero figure should be the 270M frontier. Larger model rows should appear
only after the reader already understands the 270M result.
