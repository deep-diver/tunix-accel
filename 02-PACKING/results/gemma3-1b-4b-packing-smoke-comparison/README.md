# Gemma3 1B/4B Sequence Packing Scale Smoke

This is a short-run check for the same sequence-packing effect seen on Gemma3 270M. It deliberately stops at 50 optimizer steps: the goal is to see whether packed batches accumulate much more useful training signal before running a long quality experiment.

Within each model, packed and unpacked use the same model, TPU, batch size, max length, LoRA rank, dataset, and optimizer settings. The only thing changed is whether examples are packed together before they are passed into the normal Tunix SFT path.

![loss_vs_useful_tokens](loss_vs_useful_tokens.png)

![throughput_and_density](throughput_and_density.png)

## Run Matrix

| Model | TPU | Chips | Batch | Max length | Steps | Variant | Density | Target tok/s | Final target tokens | Final loss | JAX peak aggregate |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Gemma3 1B | v5litepod-4 / tunix-packing-1b | 4 | 8 | 512 | 50 | unpacked | 10.5% | 403 | 4408 | 2.3567 | 5.92 GB |
| Gemma3 1B | v5litepod-4 / tunix-packing-1b | 4 | 8 | 512 | 50 | packed | 99.3% | 8437 | 92878 | 1.5123 | 5.93 GB |
| Gemma3 4B | v5litepod-4 / tunix-packing-4b | 4 | 4 | 512 | 50 | unpacked | 10.5% | 199 | 2254 | 1.3675 | 16.51 GB |
| Gemma3 4B | v5litepod-4 / tunix-packing-4b | 4 | 4 | 512 | 50 | packed | 99.3% | 4615 | 51765 | 1.0673 | 16.51 GB |

## Ratios

| Model | Target-token throughput | Final target tokens in 50 steps | Density change | Step-time change |
| --- | ---: | ---: | ---: | ---: |
| Gemma3 1B | 20.9x | 21.1x | 9.5x | 1.001x |
| Gemma3 4B | 23.1x | 23.0x | 9.5x | 0.999x |

## Batch Sizing Notes

These OOMs were batch-search observations on the same v5litepod-4 hardware, not evidence that packing itself reduces model memory. Sequence packing keeps the tensor shape fixed; its win here is that the fixed shape contains far fewer padding tokens.

| Model | Tried condition | TPU | Result | Compiler detail |
| --- | --- | --- | --- | --- |
| Gemma3 1B | b16, L512, 50 steps | v5litepod-4 / 4 chips | compile OOM | XLA reported 16.48 GiB used vs 15.75 GiB HBM, exceeded by 747 MiB |
| Gemma3 4B | b16, L512, 50 steps | v5litepod-4 / 4 chips | compile OOM | XLA reported 28.93 GiB used vs 15.75 GiB HBM, exceeded by 13.18 GiB |
| Gemma3 4B | b8, L512, 50 steps | v5litepod-4 / 4 chips | compile OOM | XLA reported 17.25 GiB used vs 15.75 GiB HBM, exceeded by 1.50 GiB |

## Interpretation

The key signal reproduced on both larger models is not lower step time. The step time is nearly unchanged because the model still sees the same static batch shape. The difference is that the packed batch is almost full: about 99.3% non-padding density versus about 10.5% for ordinary fixed-length batches on this OPUS100 EN-FR prompt format.

That changes the unit economics of training. In 50 steps, Gemma3 1B processed about 92.9k target tokens with packing versus 4.4k without it. Gemma3 4B shows the same pattern: 51.8k versus 2.3k.

This is still a smoke test, not a final quality claim. It is enough to justify the next long run only if we care about final output quality for 1B or 4B; for throughput behavior, the effect is already clear.

The memory column above is the aggregate JAX device-memory snapshot recorded after training. It is useful as a run sanity check, but it is not the same as a compile-time XLA buffer-assignment peak.
