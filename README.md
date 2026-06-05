# Tunix Accel

This repository contains drop-in acceleration and memory-efficiency experiments
for JAX/Tunix training.

- `tunix_accel/`: reusable patch code for Tunix decoder-LM training.
- `01-CCE/`: the final experiment report, retained summary data, figures, and
  reproduction guide.
- `02-PACKING/`: the final sequence-packing experiment report, retained summary
  data, figures, and reproduction guide.
- `03-TILED-MLP/`: the final Gemma3 tiled gated-MLP experiment report plus
  folded-in Gemma4 boundary rows,
  retained summary data, raw final records, figures, and reproduction guide.
- `04-ACTIVATION-POLICY/`: the final Gemma3 activation remat/offload policy
  experiment report plus folded-in Gemma4 boundary rows, retained data, figures, and
  reproduction guide.
- `05-GEMMA3-LARGE-SWEEP/`: Gemma3 12B/27B large-model TPU v5e patch sweep,
  retained summaries, figures, and reproduction guide.

Raw TPU traces, checkpoints, smoke outputs, and intermediate reports are kept
out of the final workstream packages after each result is consolidated.

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install .
```

When installed, the package registers a small `sitecustomize.py` hook. It waits
for supported Tunix modules to be imported, then applies process-local patches:
Cut Cross Entropy for Tunix SFT loss, Gemma3/Gemma4 tiled MLP replacement,
optional Gemma3 Splash Attention, and optional Gemma3/Gemma4 activation
remat/offload policies. Future patches can live under the same package without
renaming the project.

Use a regular wheel install when validating startup hooks. Editable installs are
fine for code hacking, but some environments already provide a system
`sitecustomize.py`; the wheel install also places a `.pth` startup hook in
site-packages so the drop-in patches still load there.

## Cut Cross Entropy Controls

```bash
export TUNIX_ACCEL_CE_TOKEN_CHUNK=128
export TUNIX_ACCEL_CE_VOCAB_CHUNK=8192
export TUNIX_ACCEL_DISABLE_CE=1
export TUNIX_ACCEL_DISABLE_AUTOPATCH=1
```

Use `TUNIX_ACCEL_DISABLE_CE=1` for a Default CE baseline while keeping other
autopatches available. Use `TUNIX_ACCEL_DISABLE_AUTOPATCH=1` to disable all
automatic patches.

## Tiled MLP Controls

By default, installed environments automatically patch Tunix Gemma3 and Gemma4
`FeedForward.block` when the corresponding model module is imported.

```bash
export TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK=128
export TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA=1
export TUNIX_ACCEL_TILED_MLP_LORA_ALPHA=32.0
export TUNIX_ACCEL_DISABLE_TILED_MLP=1
```

Use `TUNIX_ACCEL_DISABLE_TILED_MLP=1` for a Default MLP baseline while keeping
other autopatches available. Use `TUNIX_ACCEL_DISABLE_AUTOPATCH=1` to disable
all automatic patches.

## Activation Policy Controls

Activation remat/offload is opt-in. By default, installed environments leave
Tunix Gemma3 and Gemma4 decoder layers unchanged.

```bash
export TUNIX_ACCEL_ACTIVATION_POLICY=split_offload
export TUNIX_ACCEL_ACTIVATION_PREVENT_CSE=0
export TUNIX_ACCEL_ACTIVATION_OFFLOAD_SRC=device
export TUNIX_ACCEL_ACTIVATION_OFFLOAD_DST=pinned_host
export TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY=1
```

Use `TUNIX_ACCEL_ACTIVATION_POLICY=none` or leave it unset for the baseline.
Use `TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY=1` to disable only this patch while
keeping other autopatches available.

## Gemma3 Splash Attention Controls

Splash Attention is optional and Gemma3-specific.

```bash
export TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION=1
export TUNIX_ACCEL_SPLASH_ATTENTION_INTERPRET=0
```

Leave `TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION` unset for the baseline. The current
adapter is validated as part of the 04 long-context workstream, but the 270M/1B
follow-up exposed a coverage gap where long-context OOM logs still contained
dense attention allocations.

Gemma4 Splash Attention is not patched here; Tunix Gemma4 already exposes a
native `use_flash_attention` path.

## Cut Cross Entropy Explicit API

```python
from tunix_accel.tunix_lora_ce import use_frozen_lm_head_ce

trainer = peft_trainer.PeftTrainer(...).with_gen_model_input_fn(...)
trainer = use_frozen_lm_head_ce(
    trainer,
    token_chunk=128,
    vocab_chunk=8192,
)
trainer.train(train_ds, eval_ds)
```

For full fine-tuning where the LM head must receive gradients:

```python
from tunix_accel.tunix_lora_ce import use_trainable_lm_head_ce
```

## Tiled MLP Explicit API

```python
from tunix_accel import gemma3_tiled_mlp

gemma3_tiled_mlp.install(token_chunk=256)
```

Normal Tunix training code should not need this call. The explicit API is kept
for notebooks, tests, or scoped experiments; installed environments apply the
same replacement automatically when Gemma3 or Gemma4 is imported. The current
Gemma3 and Gemma4 adapters support both dense projection kernels and Qwix-LoRA
projection deltas.

For Gemma4:

```python
from tunix_accel import gemma4_tiled_mlp

gemma4_tiled_mlp.install(token_chunk=256)
```

## Activation Policy Explicit API

```python
from tunix_accel import gemma3_activation_policy

gemma3_activation_policy.install(policy="split_offload")
```

Normal Tunix training code should not need this call when the package is
installed and `TUNIX_ACCEL_ACTIVATION_POLICY` is set. The explicit API is kept
for notebooks, tests, or scoped experiments.

For Gemma4:

```python
from tunix_accel import gemma4_activation_policy

gemma4_activation_policy.install(policy="split_offload")
```

## Packing API

```python
from tunix_accel.packing import pack_records

packed = pack_records(
    tokenized_records,
    max_length=2048,
    pad_token_id=0,
    strategy="best_fit_decreasing",
)

batch = packed.as_numpy()

tunix_batch = packed.as_tunix()
```

The packed batch includes `input_ids`, `labels`, `loss_mask`, `input_mask`,
per-segment reset `positions`, `segment_ids`, and an optional block-causal
`attention_mask` that prevents attention leakage between packed samples.
`as_tunix()` maps `loss_mask` to Tunix's `input_mask` argument and keeps the
token-valid mask as `valid_mask`.

## Final Experiment Package

- Report: `01-CCE/TECHNICAL_REPORT.md`
- Reproduction guide: `01-CCE/REPRODUCE.md`
- Retained data: `01-CCE/data/`
- Figures: `01-CCE/assets/`
- Packing report: `02-PACKING/TECHNICAL_REPORT.md`
- Packing reproduction guide: `02-PACKING/REPRODUCE.md`
- Packing retained data: `02-PACKING/data/`
- Packing figures: `02-PACKING/assets/`
- Tiled MLP report: `03-TILED-MLP/TECHNICAL_REPORT.md`
- Tiled MLP reproduction guide: `03-TILED-MLP/REPRODUCE.md`
- Tiled MLP retained data: `03-TILED-MLP/data/`
- Tiled MLP figures: `03-TILED-MLP/assets/`
- Activation policy report: `04-ACTIVATION-POLICY/TECHNICAL_REPORT.md`
- Activation policy reproduction guide: `04-ACTIVATION-POLICY/REPRODUCE.md`
- Activation policy retained data: `04-ACTIVATION-POLICY/data/`
- Activation policy figures: `04-ACTIVATION-POLICY/assets/`
- Gemma3 large-model sweep report:
  `05-GEMMA3-LARGE-SWEEP/TECHNICAL_REPORT.md`
- Gemma3 large-model sweep reproduction guide:
  `05-GEMMA3-LARGE-SWEEP/REPRODUCE.md`
- Gemma3 large-model sweep retained data: `05-GEMMA3-LARGE-SWEEP/data/`
- Gemma3 large-model sweep figures: `05-GEMMA3-LARGE-SWEEP/assets/`
- Gemma4 base boundary rows: retained with the relevant workstream data. The
  current `01-CCE` report is intentionally rebuilt around the complete Gemma3
  270M rerun; Gemma4 rows remain transfer/boundary data rather than the primary
  CCE claim.
- Integrated Gemma3/Gemma4 report figures:
  `tools/plot_integrated_workstream_figures.py`
- Follow-up research directions: `RESEARCH_DIRECTIONS.md`

Headline retained results:

- Cut Cross Entropy reduced Gemma3 270M OPUS100 EN-FR b16/L512 train-step XLA
  planned HBM from 12.57 GiB/chip to 4.98 GiB/chip. The same-shape 5,000-step
  run kept train/eval loss in the same band, while CCE increased mean step time
  from 0.106s to 0.196s. Frontier sweeps on `v5litepod-1` show b64/L512 as a
  CCE-only fit.
- Sequence packing raised useful target-token throughput by 20x+ on short
  OPUS100 EN-FR SFT examples by removing padding waste.
- Gemma3 Tiled MLP moved the 4B LoRA v5litepod-8 keypoint from Default MLP
  L4096 compile OOM at 20.19 GiB/chip planned HBM to Tiled MLP L4096
  completion at 14.55 GiB/chip. At L2048, the same readout moved from
  10.36 GiB/chip to 7.06 GiB/chip.
- Gemma3 activation offload moved the 4B LoRA v5litepod-8 L4096 keypoint from
  Default CE/no-policy compile OOM at 22.16 GiB/chip planned HBM to
  `split_offload` completion at 14.40 GiB/chip planned HBM. Plain
  `split_remat` did not move the boundary. In a separate CCE + Tiled MLP +
  Splash Attention long-context ablation on v5litepod-16, no activation offload
  failed at L32768 with 23.50 GiB/chip planned HBM, while `split_offload`
  completed at 15.07 GiB/chip. A smaller 270M/1B follow-up retained in
  `04-ACTIVATION-POLICY/results/small-model-splash-activation-ablation/`
  found a useful 270M offload boundary move, but also exposed dense-attention
  coverage gaps in long-context OOM logs.
- The Gemma3 large-model sweep confirms that the patch stack still moves
  sharded 12B/27B boundaries on TPU v5e. Gemma3 12B on `v5litepod-4` moved from
  Default L1024 compile OOM at 17.42 GiB/chip to Stacked L4096 completion at
  14.42 GiB/chip. Gemma3 27B on `v5litepod-8` moved from Default L1024 compile
  OOM at 24.66 GiB/chip to Stacked L2048 completion at 13.58 GiB/chip. The
  tradeoff is large step-time overhead on offload-heavy rows.
- Gemma4 base boundary rows are folded into the same four workstreams. At LoRA rank
  16, batch 1, max length 2048: `google/gemma-4-E2B` on `v5litepod-4` shows
  Default CE, Packing, and Split Remat compile OOM while CCE, Tiled MLP, and
  Split Offload train; `google/gemma-4-E4B` on `v5litepod-8` shows Default CE,
  CCE, Packing, and Split Remat compile OOM while Tiled MLP and Split Offload
  train. These Gemma4 runs are base-checkpoint memory/compile/step-time checks;
  translation samples and BLEU/chrF are intentionally out of scope.
