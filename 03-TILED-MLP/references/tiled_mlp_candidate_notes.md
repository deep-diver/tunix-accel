# Tiled MLP Candidate Notes

After Cut Cross Entropy and sequence packing, the next plausible memory target
was the transformer block itself rather than the loss or the dataset layout.

For Gemma3-style gated MLPs, the expensive intermediate is:

```text
activation(x @ gate) * (x @ up)
```

At long context this grows with token count and intermediate dimension. Tiling
the token dimension is attractive because it does not change the objective: each
tile computes the same projection and gated product, then backward recomputes
per-tile intermediates through a custom VJP.

This candidate is more model-family-sensitive than CCE or packing:

- CCE patches the loss path and mostly depends on hidden states, labels, and the
  LM head.
- Packing patches the data layout and attention/loss masks.
- Tiled MLP patches a concrete model module with concrete projection names,
  activation choice, sharding expectations, and LoRA wrappers.

For that reason, the first production scope is Gemma3 only. Other families
should get explicit adapters only after Gemma3 shows enough value to justify the
extra surface area.
