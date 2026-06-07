# Future Patch Notes

These notes capture implementation lessons from the 01-CCE workstream. They are
intended for future Tunix acceleration patches that should install cleanly,
run on TPU, and leave auditable experiment artifacts.

## Mainline Scope

- Keep `main` focused on the current retained patch surface.
- Preserve exploratory workstreams on branches before removing them from
  `main`.
- Avoid cross-workstream runtime dependencies. If a report directory needs a
  runner, keep that runner inside the same directory or under a clearly shared
  top-level module.

## Drop-In Patch Surface

- Prefer a package install that works with ordinary Tunix user code.
- Use `sitecustomize.py` plus a tiny `.pth` startup hook so wheel installs and
  editable installs both activate the package.
- Keep startup code fail-closed. Import hooks should not import JAX, Tunix, or
  model-heavy modules until the target Tunix module is actually imported.
- Use import hooks for process-local monkey patches. Patch after the target
  module loads, and guard with an idempotent marker on the target module.
- Provide an explicit API in addition to automatic patching. It is useful for
  notebooks, tests, and scoped experiments.

## Environment Controls

- Use one global disable, for example `TUNIX_ACCEL_DISABLE_AUTOPATCH`.
- Use one patch-specific disable, for example `TUNIX_ACCEL_DISABLE_CE`.
- Parse booleans case-insensitively and accept `1/0`, `true/false`,
  `yes/no`, and `on/off`.
- Keep conservative defaults. Add named presets only after they have been
  validated on the relevant TPU mesh.
- Record all effective patch knobs in run summaries, including chunk sizes,
  mesh shape, model id, dataset mode, and whether the patch was actually
  installed.

## Model Integration

- Separate algorithm code from model-family adapters.
- A model-agnostic core is useful, but the drop-in adapter may need to be
  model-family-specific because Tunix model classes differ in module layout,
  projection naming, remat wrappers, and generation paths.
- Treat generation as a separate path from training. If training intercepts
  hidden states or LM-head behavior, add tests that generation restores the
  original decode behavior.
- Prefer fallback-to-original behavior for unsupported models unless the user
  explicitly asks for a hard failure.

## TPU And Distributed Runs

- Multi-host TPU slices need `jax.distributed.initialize()` before model or
  checkpoint setup.
- Launch multi-host tests with `gcloud compute tpus tpu-vm ssh --worker=all`;
  running on worker 0 alone is not a valid distributed smoke.
- Log `process_index`, `process_count`, `local_device_count`, and
  `global_device_count` at startup.
- Reuse TPU VMs for same-size experiments when possible, but always delete
  active TPU VMs after artifacts are copied back.

## Experiment Runner Design

- Keep runner outputs compact and self-describing:
  - `summary.json`
  - `history.csv`
  - `*_results.csv`
  - `runner.log`
  - trimmed XLA memory reports
- Do not keep checkpoints unless the experiment explicitly needs them.
- Archive raw worker outputs as small tarballs and keep extracted `raw/`
  directories disposable.
- Use one row per comparable run and include status, failure type, XLA memory,
  runtime memory, step time, mesh, chip count, and patch flags.

## Metrics

- Use XLA buffer-assignment planned HBM per chip as the primary compile-fit
  metric.
- Keep runtime memory snapshots, but label their scope precisely. A multi-host
  artifact copied from one worker usually reports that host's local chips, not
  full-pod aggregate memory.
- Keep step time separate from capacity. A patch can open a memory frontier and
  still be slower at the same shape.
- For quality sanity checks, compare matched training budgets and preserve raw
  loss history in addition to smoothed curves.

## Report Hygiene

- State the claim boundary early.
- Do not mix isolated-patch claims with stacked-patch claims unless the section
  explicitly says what is fixed and what varies.
- Put exact TPU type, chip count, mesh, batch, context length, LoRA rank, step
  count, and dataset mode near every headline result.
- Prefer one canonical report for a given workstream. If later experiments
  change the story, fold them into that report rather than scattering
  disconnected addenda.
