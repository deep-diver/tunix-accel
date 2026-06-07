# 02-PACKING

This directory contains the retained artifacts for the Sequence Packing
workstream. It is rebuilt around Gemma3 270M as the base case before any larger
model transfer checks are considered.

## Contents

- `TECHNICAL_REPORT.md`: narrative report with embedded plots.
- `REPRODUCE.md`: commands for reproducing the local density sweeps and TPU
  Gemma3 270M runs.
- `GEMMA3_270M_EXPERIMENT_DESIGN.md`: experiment design used for the rerun.
- `run_efficiency_benchmark.py`: no-model packing-density sweep.
- `run_gemma_tokenizer_benchmark.py`: Gemma-tokenizer packing-density sweep.
- `run_gemma_training_benchmark.py`: Tunix/Gemma training runner.
- `run_gemma3_270m_packing_sweep.py`: TPU case-grid runner.
- `remote_gemma3_270m_packing_worker.sh`: TPU VM wrapper.
- `visualize_270m_results.py`: processed table and plot generator.
- `assets/`: final figures used by the report.
- `data/local_density/`: retained local density sweep CSVs.
- `data/processed/`: compact TPU result tables and summary JSON.
- `data/raw_artifacts/`: compressed raw TPU worker outputs.

Extracted raw TPU directories are intentionally ignored. Recreate them from the
`.tgz` files by running:

```bash
python3 02-PACKING/visualize_270m_results.py
```

The patch implementation itself lives outside this directory in
`tunix_accel/`.

## Drop-In Use

After installing this package, ordinary Tunix code is unchanged unless
`packing=` is passed to `PeftTrainer.with_gen_model_input_fn`.

```python
from tunix_accel import TunixPackingConfig

trainer = peft_trainer.PeftTrainer(...).with_gen_model_input_fn(
    gen_model_input_fn,
    packing=TunixPackingConfig(
        pad_token_id=0,
        strategy="best_fit_decreasing",
    ),
)
trainer.train(train_ds, eval_ds)
```

Short forms are also accepted:

```python
trainer = trainer.with_gen_model_input_fn(gen_model_input_fn, packing=True)
trainer = trainer.with_gen_model_input_fn(
    gen_model_input_fn,
    packing={"pad_token_id": 0, "drop_remainder": True},
)
```

If `packing` is omitted, the widened API is inert. If `packing=False`, packing
is explicitly disabled for that trainer.

## Main Finding

Sequence packing is not a fixed-shape memory optimizer. On Gemma3 270M
`v5litepod-1`, packed and unpacked runs hit the same XLA planned-HBM frontier.
The gain is instead useful-token density: at b16/L512, target-token throughput
rose from about 1.54k/s to 32.6k/s while same-shape step time stayed within
0.4%.
