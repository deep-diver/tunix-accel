# Tunix Accel

This repository contains drop-in acceleration and memory-efficiency experiments
for JAX/Tunix training.

- `tunix_accel/`: reusable patch code for Tunix decoder-LM training.
- `01-CCE/`: the final experiment report, retained summary data, figures, and
  reproduction guide.
- `02-PACKING/`: the final sequence-packing experiment report, retained summary
  data, figures, and reproduction guide.
- `03-TILED-MLP/`: the final Gemma3-only tiled gated-MLP experiment report,
  retained summary data, raw final records, figures, and reproduction guide.
- `03.5-PALLAS-TILED-MLP/`: the Pallas-backend follow-up for Gemma3 Tiled MLP,
  comparing the opt-in Pallas matmul backend against the 03 XLA tiled backend.

Raw TPU traces, checkpoints, smoke outputs, and intermediate reports are kept
out of the final workstream packages after each result is consolidated.

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install .
```

When installed, the package registers a small `sitecustomize.py` hook. It waits
for supported Tunix modules to be imported, then applies process-local patches:
Cut Cross Entropy for Tunix SFT loss, and the Gemma3 tiled MLP replacement for
`tunix.models.gemma3.model`. Future patches can live under the same package
without renaming the project.

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

## Gemma3 Tiled MLP Controls

By default, installed environments automatically patch Tunix Gemma3
`FeedForward.block` when `tunix.models.gemma3.model` is imported.

```bash
export TUNIX_ACCEL_TILED_MLP_TOKEN_CHUNK=128
export TUNIX_ACCEL_TILED_MLP_BACKEND=xla
export TUNIX_ACCEL_TILED_MLP_BACKEND=pallas
export TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA=1
export TUNIX_ACCEL_TILED_MLP_LORA_ALPHA=32.0
export TUNIX_ACCEL_DISABLE_TILED_MLP=1
```

Use `TUNIX_ACCEL_DISABLE_TILED_MLP=1` for a Default MLP baseline while keeping
other autopatches available. The backend defaults to `xla`; use
`TUNIX_ACCEL_TILED_MLP_BACKEND=pallas` to opt into the experimental TPU Pallas
matmul backend. Use `TUNIX_ACCEL_DISABLE_AUTOPATCH=1` to disable all automatic
patches.

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

## Gemma3 Tiled MLP API

```python
from tunix_accel import gemma3_tiled_mlp

gemma3_tiled_mlp.install(token_chunk=256, matmul_backend="xla")
gemma3_tiled_mlp.install(token_chunk=128, matmul_backend="pallas")
```

Normal Tunix training code should not need this call. The explicit API is kept
for notebooks, tests, or scoped experiments; installed environments apply the
same replacement automatically when Gemma3 is imported. The current adapter is
Gemma3-specific and supports both dense projection kernels and Qwix-LoRA
projection deltas. The Pallas backend is opt-in and currently retained as an
experimental backend rather than the default.

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
- Pallas Tiled MLP report: `03.5-PALLAS-TILED-MLP/TECHNICAL_REPORT.md`
- Pallas Tiled MLP reproduction guide: `03.5-PALLAS-TILED-MLP/REPRODUCE.md`
- Pallas Tiled MLP retained data: `03.5-PALLAS-TILED-MLP/data/`
- Pallas Tiled MLP figures: `03.5-PALLAS-TILED-MLP/assets/`
- Follow-up research directions: `RESEARCH_DIRECTIONS.md`

Headline retained results:

- Cut Cross Entropy reduced Gemma3 270M EN-FR b16 train-step XLA peak memory
  from 10.21 GiB to 2.21 GiB while keeping eval loss and BLEU essentially at
  parity. Same-batch CCE steps were slower.
- Sequence packing raised useful target-token throughput by 20x+ on short
  OPUS100 EN-FR SFT examples by removing padding waste.
- Gemma3 Tiled MLP moved the 4B LoRA v5litepod-8 keypoint from Default MLP
  L4096 compile OOM to Tiled MLP L4096 completion, with L2048 XLA planned HBM
  moving from 82.9 GiB to 56.5 GiB aggregate.
- The Pallas Tiled MLP follow-up proved an opt-in TPU Pallas backend works and
  composes with CCE, but it is not yet a better default than the XLA tiled
  backend: L4096 planned HBM improved by 3.0%, while step time slowed by 5.4%.
