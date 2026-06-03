# Small-Model Splash + Activation-Offload Follow-Up

This directory retains the Gemma3 270M and 1B follow-up artifacts for the 04 activation-policy workstream.

Fixed requested stack for every row: CCE enabled, Tiled MLP enabled, Splash Attention requested, LoRA rank 16, batch size 1. The changed variable is activation policy: `none` vs `split_offload`.

Important caveat: long-context OOM logs for the small-model runs still contain dense attention score/mask allocation evidence. These results are therefore retained as adapter-coverage diagnostics, not as a clean proof that Splash Attention lowered every small-model path.

Retained files:

- `small_model_splash_activation_metrics.csv`
- `small_model_splash_activation_metrics.json`
- `270m/` raw run directories copied from `tunix-actoff-270m-1`
- `1b/` raw run directories copied from `tunix-actoff-1b-4`
