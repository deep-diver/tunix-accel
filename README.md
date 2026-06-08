# Tunix Accel

This repository currently publishes two retained workstreams:

- `01-CCE/`: Cut Cross Entropy for JAX/Tunix decoder-LM training on TPU.
- `02-PACKING/`: Sequence packing for JAX/Tunix SFT input batches, with
  Gemma3 270M base experiments plus Gemma3 1B and Qwen3 0.6B transfer checks.

The main branch is intentionally scoped to workstreams with a cleaned
implementation and retained technical report. Earlier exploratory workstreams
for tiled MLP, activation policy, and large-model patch stacks are preserved on:

```text
codex/archive-02-05-workstreams
```

Those branches are kept as research material, not as the current mainline
surface.

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install .
```

When installed, the package registers a small `sitecustomize.py` hook. It waits
for `tunix.sft.peft_trainer` to be imported, then applies a process-local CCE
patch unless disabled by environment variables.

## CCE Controls

Default CCE behavior:

```bash
export TUNIX_ACCEL_CE_TOKEN_CHUNK=128
export TUNIX_ACCEL_CE_VOCAB_CHUNK=8192
```

Use a TPU-oriented larger-chunk preset:

```bash
export TUNIX_ACCEL_CE_PRESET=tpu_large_chunks
```

Disable only CCE while leaving the startup hook available:

```bash
export TUNIX_ACCEL_DISABLE_CE=true
```

Disable all automatic patching:

```bash
export TUNIX_ACCEL_DISABLE_AUTOPATCH=true
```

Boolean controls accept `1/0`, `true/false`, `yes/no`, and `on/off`
case-insensitively.

## Explicit API

Existing Tunix code can also opt in explicitly:

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

When installed, the package also widens Tunix
`PeftTrainer.with_gen_model_input_fn` with optional `packing=` support. It is a
no-op unless `packing=` is supplied.

```python
from tunix_accel import TunixPackingConfig

trainer = peft_trainer.PeftTrainer(...).with_gen_model_input_fn(
    gen_model_input_fn,
    packing=TunixPackingConfig(pad_token_id=0),
)
trainer.train(train_ds, eval_ds)
```

Use `packing=True` for defaults, a mapping for lightweight configuration, or
omit `packing` to keep ordinary Tunix behavior.

## Report Package

- `01-CCE/`
  - Report: `01-CCE/TECHNICAL_REPORT.md`
  - Reproduction guide: `01-CCE/REPRODUCE.md`
  - Retained data: `01-CCE/data/`
  - Figures: `01-CCE/assets/`
- `02-PACKING/`
  - Report: `02-PACKING/TECHNICAL_REPORT.md`
  - Reproduction guide: `02-PACKING/REPRODUCE.md`
  - Retained data: `02-PACKING/data/`
  - Figures: `02-PACKING/assets/`
- Implementation notes for future patches: `FUTURE_PATCH_NOTES.md`

The retained artifacts are compact enough to audit the report without keeping
full extracted TPU dumps or checkpoints.
