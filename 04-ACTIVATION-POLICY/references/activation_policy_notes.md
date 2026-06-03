# Activation Policy Notes

Unsloth's long-context memory story repeatedly highlights gradient
checkpointing and activation offloading as a major contributor to longer
context windows. JAX already exposes the necessary primitives:

- `jax.checkpoint` / `flax.nnx.remat`
- checkpoint policies from `jax.checkpoint_policies`
- `jax.ad_checkpoint.checkpoint_name`
- host offload policies such as `save_and_offload_only_these_names`

The research question is therefore not whether remat exists. It is whether a
Gemma3/Tunix-specific placement of remat and named offload boundaries beats the
existing model plan enough to justify a drop-in patch.

Initial implementation:

- `layer_remat`: remat the whole decoder layer.
- `layer_offload`: remat the whole decoder layer and offload its input residual.
- `split_remat`: remat attention and MLP as separate regions.
- `split_offload`: split remat plus named offload of attention/MLP residuals.

Expected tradeoff:

- Lower planned or runtime HBM if saved activations dominate.
- Higher step time from recompute and host-device transfers.
- Exact loss/gradient parity because model math is unchanged.
