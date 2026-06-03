# Tunix Accel

This repository contains drop-in acceleration and memory-efficiency experiments
for JAX/Tunix training.

- `tunix_accel/`: reusable patch code for Tunix decoder-LM training.
- `01-CCE/`: the final experiment report, retained summary data, figures, and
  reproduction guide.
- `02-PACKING/`: the active padding-free / uncontaminated packing workstream.
  It includes a no-model efficiency benchmark and a Gemma-tokenizer OPUS100
  benchmark.
- `03-TILED-MLP/`: the active Gemma3-only tiled gated-MLP workstream. It
  currently contains a Gemma-free JAX custom-VJP prototype with
  forward/gradient parity tests.

Raw TPU traces, checkpoints, smoke outputs, and intermediate reports were
removed after the CCE result was consolidated.

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
export TUNIX_ACCEL_TILED_MLP_FALLBACK_ON_LORA=1
export TUNIX_ACCEL_DISABLE_TILED_MLP=1
```

Use `TUNIX_ACCEL_DISABLE_TILED_MLP=1` for a Default MLP baseline while keeping
other autopatches available. Use `TUNIX_ACCEL_DISABLE_AUTOPATCH=1` to disable
all automatic patches.

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

gemma3_tiled_mlp.install(token_chunk=256)
```

Normal Tunix training code should not need this call. The explicit API is kept
for notebooks, tests, or scoped experiments; installed environments apply the
same replacement automatically when Gemma3 is imported. It currently targets
non-LoRA Gemma3 projection kernels; Qwix-LoRA projection params fall back to the
original MLP by default.

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
- Active packing notes: `02-PACKING/README.md`
- Active tiled MLP notes: `03-TILED-MLP/README.md`
- Follow-up research directions: `RESEARCH_DIRECTIONS.md`

The final result: Cut Cross Entropy reduced Gemma3 270M EN-FR b16 train-step
XLA peak memory from 10.21 GiB to 2.21 GiB while keeping eval loss and BLEU
essentially at parity. Same-batch CCE steps were slower.
