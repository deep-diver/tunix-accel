"""Drop-in acceleration helpers for JAX/Tunix training."""

from tunix_accel.tunix_lora_ce import chunked_lm_head_ce_loss_fn
from tunix_accel.tunix_lora_ce import frozen_lm_head_ce_loss_fn
from tunix_accel.tunix_lora_ce import trainable_lm_head_ce_loss_fn
from tunix_accel.tunix_lora_ce import use_frozen_lm_head_ce
from tunix_accel.tunix_lora_ce import use_trainable_lm_head_ce
from tunix_accel.tunix_packing import TunixPackingConfig


__all__ = [
    "TunixPackingConfig",
    "chunked_lm_head_ce_loss_fn",
    "frozen_lm_head_ce_loss_fn",
    "trainable_lm_head_ce_loss_fn",
    "use_frozen_lm_head_ce",
    "use_trainable_lm_head_ce",
]
