# Sequence Packing Benchmark Notes

These notes capture the framing used for the 02-PACKING experiment.

## Benchmark Question

For instruction-tuning datasets with many short examples, ordinary fixed-length
batches waste a large fraction of sequence slots on padding. Sequence packing
should convert those empty slots into valid training tokens while preserving the
per-example loss semantics.

## What To Measure

- non-padding token density
- target/loss token throughput
- optimizer step time
- loss vs consumed target tokens
- small quality sanity metrics after a real SFT run

## What Not To Claim

Packing should not be reported as direct model-memory reduction. At the same
batch size and max length, the static tensor shape is still the same. The win is
that the shape contains more useful tokens.

Packing is also dataset-dependent. It helps most when examples are short
relative to `max_length`, and least when examples already fill the context.
