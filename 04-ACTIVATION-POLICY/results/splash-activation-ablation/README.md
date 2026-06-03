# Splash Attention + Activation Offload Ablation

This isolates activation offloading after Splash Attention has removed the dense attention O(T^2) wall.

Fixed setup: Gemma3 4B IT, LoRA rank 16, batch size 1, CCE enabled, Tiled MLP enabled, Splash Attention enabled, TPU v5litepod-16, 16 global chips, mesh fsdp=16,tp=1. The only changed variable is activation policy: `none` vs `split_offload`.

Key result: without activation offload, L32768 compile-time HBM is 23.50 GiB/chip and fails on v5e's 15.75 GiB/chip limit. With split activation offload, the same L32768 case compiles and runs at 15.07 GiB/chip, leaving 0.68 GiB/chip of headroom.

At shorter contexts, offload is not required but still lowers XLA train_step peak memory: L8192 drops from 6.95 to 4.59 GiB/chip; L16384 drops from 13.42 to 8.48 GiB/chip. The cost is step-time: no-offload is faster where it fits, while split_offload buys headroom and reaches 32K.

![Splash activation offload ablation](./splash_activation_offload_ablation.png)
