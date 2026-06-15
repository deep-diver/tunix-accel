# A100 CCE Retry2 Summary

- Rows extracted: 4
- Success rows: 4
- Source log: `/Users/deep-diver/Developers/TUNIX-TRY/01-CCE/data/mesh_pathology_a100_cce_retry2/dstack_run.log`

| case | mesh | chunks | step time (s) | compile (s) | XLA HBM GiB/chip | runtime HBM GB | tokens/s | cce loops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bad | fsdp=2,tp=2 | 128/8192 | 4.278102 | 46.532 | 18.96 | 81.44 | 487.5 | 76 |
| good | fsdp=2,tp=2 | 512/65536 | 0.635943 | 45.513 | 19.32 | 83.02 | 3279.5 | 3 |
| control-fsdp4-tp1 | fsdp=4,tp=1 | 128/8192 | 0.815018 | 46.655 | 34.47 | 148.10 | 2559.0 | 76 |
| control-fsdp1-tp4 | fsdp=1,tp=4 | 128/8192 | 0.617338 | 39.452 | 10.10 | 44.05 | 3378.4 | 76 |

## Ratios

- bad/good step-time ratio: 6.73x
- bad/good planned-HBM ratio: 0.98x
- bad/good CCE-loop ratio: 25.33x

Interpretation: on A100, the small-chunk CCE failure reproduces as a steady-state throughput problem, not a compile-time-only issue. Planned HBM is effectively flat between bad and good while loop count changes sharply.
