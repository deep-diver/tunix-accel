# LoRA-FA Metric Schema

This file defines the maximal metric contract for the LoRA-FA experiments. The
goal is to collect more than the final report will need so the analysis can be
re-cut without rerunning TPU jobs.

## Required Identifiers

Every row must include:

- `run_id`
- `timestamp_utc`
- `git_commit`
- `branch`
- `model_family`
- `model_id`
- `model_size_label`
- `checkpoint_source`
- `training_mode`
- `variant`
- `lora_rank`
- `lora_alpha`
- `lora_scaling`
- `lorafa_mode`
- `lorafa_correction_eps`
- `lorafa_use_rslora`
- `dataset_name`
- `dataset_split`
- `tokenizer_id`
- `max_length`
- `batch_size`
- `gradient_accumulation_steps`
- `effective_batch_size`
- `max_steps`
- `learning_rate`
- `optimizer`
- `weight_decay`
- `warmup_steps`
- `seed`
- `tpu_type`
- `tpu_zone`
- `chip_count`
- `mesh_fsdp`
- `mesh_tp`
- `mesh_dp`

## Fit And Memory Metrics

Collect both compile-planned and runtime-observed numbers when possible.

- `status`
- `failure_type`
- `failure_message_short`
- `compile_success`
- `train_success`
- `oom_exceeded_gib_per_chip`
- `oom_limit_gib_per_chip`
- `xla_train_step_peak_gib_per_chip`
- `xla_train_step_peak_gib_aggregate`
- `xla_train_step_report_path`
- `runtime_peak_hbm_gib_per_chip`
- `runtime_peak_hbm_gib_aggregate`
- `runtime_hbm_limit_gib_per_chip`
- `runtime_hbm_limit_gib_aggregate`
- `runtime_hbm_headroom_gib_per_chip`
- `runtime_hbm_headroom_gib_aggregate`
- `model_param_gib_aggregate`
- `trainable_param_gib_aggregate`
- `optimizer_state_gib_aggregate`
- `gradient_state_gib_aggregate`
- `activation_estimate_gib_aggregate`
- `compile_wall_time_sec`
- `first_step_compile_or_trace_sec`

## LoRA-FA Internal Metrics

These are specific to the method and should be emitted for every model row.

- `lora_a_tensors`
- `lora_b_tensors`
- `lora_a_params`
- `lora_b_params`
- `lora_a_bytes`
- `lora_b_bytes`
- `lora_trainable_params`
- `lora_trainable_bytes`
- `lora_a_grad_tensors`
- `lora_b_grad_tensors`
- `lora_a_grad_global_norm`
- `lora_b_grad_global_norm_raw`
- `lora_b_grad_global_norm_corrected`
- `lorafa_correction_applied`
- `lorafa_correction_nan_count`
- `lorafa_correction_inf_count`
- `lorafa_gram_condition_mean`
- `lorafa_gram_condition_max`
- `lorafa_gram_rank_deficient_count`
- `lorafa_a_value_delta_max`
- `lorafa_b_value_delta_max`
- `lorafa_optimizer_state_tensors`
- `lorafa_optimizer_state_bytes`

Interpretation gates:

- `lora_a_grad_tensors` must be zero for LoRA-FA variants.
- `lorafa_a_value_delta_max` must be zero or within exact checkpoint/load noise.
- `lorafa_b_value_delta_max` must be positive after at least one optimizer step.
- `lorafa_correction_nan_count` and `lorafa_correction_inf_count` must be zero.

## Speed And Throughput Metrics

Collect raw histories and summarized fields.

- `wall_time_sec`
- `first_step_time_sec`
- `second_step_time_sec`
- `mean_step_time_sec`
- `median_step_time_sec`
- `p90_step_time_sec`
- `p99_step_time_sec`
- `mean_step_time_sec_excl_first`
- `median_step_time_sec_excl_first`
- `tokens_per_step`
- `valid_tokens_per_step`
- `loss_tokens_per_step`
- `tokens_per_sec`
- `valid_tokens_per_sec`
- `loss_tokens_per_sec`
- `tokens_per_sec_excl_first`
- `valid_tokens_per_sec_excl_first`
- `loss_tokens_per_sec_excl_first`
- `mfu_estimate`
- `tflops_estimate`
- `compile_cache_hit`

## Training Dynamics

Save per-step history, then summarize:

- `step`
- `train_loss`
- `eval_loss`
- `grad_norm`
- `learning_rate_at_step`
- `tokens_seen`
- `valid_tokens_seen`
- `loss_tokens_seen`
- `examples_seen`
- `nan_loss_count`
- `inf_loss_count`
- `final_train_loss`
- `final_eval_loss`
- `min_train_loss`
- `mean_train_loss_last_10pct`
- `loss_slope_last_10pct`
- `loss_area_under_curve`

Normalize loss curves by:

- optimizer step
- wall time
- valid tokens seen
- loss tokens seen

## Quality Metrics

For 270M, run the full quality gate first. For larger rows, keep the same schema
even if some fields are null during smoke runs.

- `quality_eval_enabled`
- `quality_task`
- `quality_num_examples`
- `generation_max_new_tokens`
- `generation_temperature`
- `bleu`
- `chrf`
- `exact_match`
- `accuracy`
- `pass_at_1`
- `mt_bench_score`
- `sample_outputs_path`
- `quality_notes`

For OPUS100 EN-FR, retain at least five side-by-side samples for final quality
rows, but do not require samples for every short scale smoke.

## Comparison Metrics

Derived after all variants for a model are present:

- `memory_ratio_vs_standard_lora_r16`
- `step_time_ratio_vs_standard_lora_r16`
- `throughput_ratio_vs_standard_lora_r16`
- `final_loss_delta_vs_standard_lora_r16`
- `eval_loss_delta_vs_standard_lora_r16`
- `quality_delta_vs_standard_lora_r16`
- `rank_memory_slope_standard_lora`
- `rank_memory_slope_lorafa`
- `rank_quality_slope_lorafa`
- `frontier_max_batch_at_length`
- `frontier_max_length_at_batch`

## Required Plots

Do not make the first report plot-heavy. Keep the final story readable, but
retain enough raw data to regenerate:

- Memory vs rank, grouped by model and variant.
- Step time vs rank, grouped by model and variant.
- Loss vs useful tokens for each model.
- Final loss delta vs memory saved.
- Context frontier or batch frontier if LoRA-FA changes fit boundaries.
- A/B parameter and optimizer-state breakdown.
- 270M quality comparison table with representative samples.

## Anomaly Stop Conditions

Stop the sweep and analyze before continuing if any of these happen:

- LoRA-FA uses more memory than standard LoRA at the same rank without a clear
  compile-sharding explanation.
- Corrected LoRA-FA is substantially slower than standard LoRA at the same rank.
- Corrected LoRA-FA loss diverges while freeze-A does not.
- A gradients appear in the gradient tree.
- A values change after a train step.
- Rank 32 or rank 64 LoRA-FA memory grows like standard LoRA.
- Larger models show the opposite trend from 270M without an obvious mesh or
  sharding cause.

After patching any anomaly, rerun the Gemma3 270M gate before resuming larger
models.

