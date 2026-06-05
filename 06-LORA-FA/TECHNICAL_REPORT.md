# LoRA-FA Technical Report

Status: in progress.

This report will compare standard Qwix LoRA against LoRA-FA variants across
Gemma3 and Gemma4 models. The first implementation checkpoint is local only:
B-only gradients, corrected B gradients, and no A updates are verified by
`tests/test_lora_fa.py`.

