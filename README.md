# Tunix Accel

This repository contains drop-in acceleration and memory-efficiency experiments
for JAX/Tunix training.

- `tunix_accel/`: reusable patch code for Tunix decoder-LM training.
- `01-CCE/`: the final experiment report, retained summary data, figures, and
  reproduction guide.
- `02-PACKING/`: the active padding-free / uncontaminated packing workstream.
  It includes a no-model efficiency benchmark and a Gemma-tokenizer OPUS100
  benchmark.

Raw TPU traces, checkpoints, smoke outputs, and intermediate reports were
removed after the CCE result was consolidated.

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

When installed, the package registers a small `sitecustomize.py` hook. It waits
for `tunix.sft.peft_trainer` to be imported, then patches Tunix's default
decoder-LM loss for supported model families. Today that means the Cut Cross
Entropy (CCE) loss path; future patches can live under the same package without
renaming the project.

## Cut Cross Entropy Controls

```bash
export TUNIX_ACCEL_CE_TOKEN_CHUNK=128
export TUNIX_ACCEL_CE_VOCAB_CHUNK=8192
export TUNIX_ACCEL_DISABLE_AUTOPATCH=1
```

Use `TUNIX_ACCEL_DISABLE_AUTOPATCH=1` for a Default CE baseline. Leave it unset
for CCE.

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

The final result: Cut Cross Entropy reduced Gemma3 270M EN-FR b16 train-step
XLA peak memory from 10.21 GiB to 2.21 GiB while keeping eval loss and BLEU
essentially at parity. Same-batch CCE steps were slower.
