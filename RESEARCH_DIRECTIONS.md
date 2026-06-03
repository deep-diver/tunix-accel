# Research Directions

This document records candidate workstreams after the completed Cut Cross
Entropy and sequence-packing experiments. The framing is intentionally close to
the Unsloth benchmark story: reduce accelerator memory or wasted compute
relative to a plain training path, preserve loss/quality parity, and keep the
feature usable as a Tunix drop-in patch where possible.

## Completed Baselines

| Workstream | Status | Main result |
| --- | --- | --- |
| `01-CCE` | Done | Replaced dense LM-head cross entropy with Cut Cross Entropy, reducing the Gemma3 270M EN-FR b16 train-step XLA peak from 10.21 GiB to 2.21 GiB with eval-loss and BLEU parity. |
| `02-PACKING` | Done | Added uncontaminated sequence packing; on OPUS100 EN-FR it recovered padding waste and produced 20x+ useful target-token throughput in short Gemma runs. |
| `03-TILED-MLP` | Done | Added a Gemma3-only tiled gated-MLP patch. On Gemma3 4B LoRA v5litepod-8, the L4096 keypoint moved from Default MLP compile OOM to Tiled MLP completion. |
| `04-ACTIVATION-POLICY` | Done | Added an opt-in Gemma3 decoder-layer remat/offload policy. Plain `split_remat` did not help the L4096 boundary, but `split_offload` moved Default CE/no-policy from compile OOM at 177.3 GiB aggregate planned HBM to completion at 115.2 GiB. |

## Completed: Tiled / Fused MLP

**Scope:** Gemma3-only until proven otherwise. The kernel math is generic for
gated MLPs, but the drop-in adapter should first target Tunix Gemma3 modules and
their `gate_proj`, `up_proj`, and `down_proj` structure.

**Why it matters:** After CCE removes the full-vocab logits tensor, long-context
Gemma3 training can become dominated by MLP activations and MLP backward
intermediates. Unsloth also presents Tiled MLP as one of its major long-context
memory optimizations.

**Implementation hypothesis:** Split the sequence dimension before the heavy
MLP projections, run gate/up/down projections tile by tile, and use a custom VJP
or rematerialized tile body so the full MLP intermediate is not resident at
once. Start with `jax.lax`/`custom_vjp`; only move to Pallas if profiling shows
the tiled path is memory-correct but compute or memory traffic becomes the new
bottleneck.

**Benchmark story:**

- Compare Default MLP vs Tiled MLP at the same model, batch, and context.
- Then compare CCE alone vs CCE + Tiled MLP.
- Report max completed context, XLA planned HBM, TPU profiler memory, step time,
  final loss, and generation/eval parity.

**Main risks:**

- Tiling can add meaningful step-time overhead because one large GEMM becomes
  several smaller sequential GEMMs.
- The Gemma3 adapter may not generalize to Llama/Qwen-style SwiGLU or GeGLU
  layouts without separate work.

## Completed: Activation Memory Policy

Gradient checkpointing, rematerialization, and activation offloading are not a
custom-kernel target in the same way CCE or Tiled MLP are. JAX already provides
`jax.checkpoint`, `flax.nnx.remat`, checkpoint policies, checkpoint names, and
host offloading. The workstream still produced a useful Gemma3 drop-in:

- `split_remat` confirmed that recompute-only policy changes did not move the
  Gemma3 4B L4096 boundary.
- `split_offload` showed that explicit named residual offload can reduce HBM
  residency enough to fit the same L4096 training shape.
- The right default remains opt-in because offload buys memory with slower
  steps and longer compile/dispatch behavior.

The retained report is `04-ACTIVATION-POLICY/TECHNICAL_REPORT.md`.

## 05 Candidate: Fused QK RoPE

**Why it matters:** Unsloth explicitly advertises fused QK RoPE kernels, with
packing support, as part of its faster-training path. This is one of the most
kernel-shaped next targets if we want a TPU/Pallas analogue to their Triton
work.

**Implementation hypothesis:** Fuse Q and K rotary embedding application into a
single XLA/Pallas-friendly operation that avoids extra rotate-half
materialization and supports packed/variable positions. The first milestone
should be a JAX primitive-level rewrite; Pallas becomes interesting only if the
plain XLA lowering still leaves measurable memory traffic or launch overhead.

**Benchmark story:**

- Microbenchmark RoPE-only forward/backward on Gemma-like Q/K shapes.
- Validate numerical parity against the model's existing RoPE path.
- Run a short Gemma SFT smoke with packing on and off.
- Report RoPE kernel time, full step time, memory, and loss parity.

**Main risks:**

- XLA may already fuse the relevant elementwise work well enough on TPU, making
  the practical gain small.
- The implementation may become model-specific around RoPE layout, grouped
  query attention, and position handling.

## 06 Candidate: Fused LoRA Projections / Backward

**Why it matters:** Unsloth's earlier speed and memory gains are strongly tied
to custom LoRA kernels. A fused LoRA path could reduce memory traffic around
adapter projections and manual backward, especially for PEFT-heavy workloads.

**Implementation hypothesis:** Build fused LoRA projection helpers for common
linear modules, combining base projection, LoRA A/B projection, scaling, and
backward logic. Start with LoRA-only paths where base weights are frozen. Full
fine-tuning should remain outside the first milestone.

**Benchmark story:**

- Compare Tunix/Qwix LoRA default vs fused LoRA on isolated linear, QKV, and MLP
  projection shapes.
- Run Gemma3 LoRA SFT smoke and quality sanity checks.
- Report step time, peak memory, gradient parity, final loss, and adapter
  checkpoint compatibility.

**Main risks:**

- This is PEFT-specific, so it is less aligned with a universal drop-in story.
- Qwix/NNX LoRA interception and Tunix trainer internals may change, making the
  patch fragile across model families.

## 07 Candidate: RMSNorm / SwiGLU / GeGLU Fused Kernels

**Why it matters:** Unsloth ships optimized kernels for normalization and MLP
activation families. These are plausible memory-traffic reductions, but they are
also the area where XLA may already do a good job on TPU.

**Implementation hypothesis:** First inspect HLO/profile output for RMSNorm and
SwiGLU/GeGLU regions. If they remain as high-traffic unfused operations, create
JAX/Pallas replacements for the hottest shape family. Otherwise, keep them as
lower-priority research notes.

**Benchmark story:**

- Microbenchmark default vs replacement kernels for Gemma-like hidden sizes.
- Verify forward/backward parity and dtype behavior.
- Only run full Tunix training if the microbenchmark shows a real TPU-side
  benefit.

**Main risks:**

- XLA fusion can make hand-written replacements unnecessary.
- These kernels may improve step time but are less likely than CCE or Tiled MLP
  to produce a dramatic max-context or peak-memory story.
