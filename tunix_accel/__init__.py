"""Drop-in acceleration and memory-efficiency patches for JAX/Tunix training."""

from tunix_accel.packing import PackingConfig
from tunix_accel.packing import PackedBatch
from tunix_accel.packing import TokenizedExample
from tunix_accel.packing import pack_examples
from tunix_accel.packing import pack_records
from tunix_accel.tunix_packing import TunixPackingConfig
from tunix_accel.tunix_packing import install as install_packing
from tunix_accel.tunix_packing import pack_tunix_batches
from tunix_accel.tunix_packing import packed_input_fn
from tunix_accel.tunix_packing import patch_trainer_api
from tunix_accel.tunix_packing import uninstall as uninstall_packing


__all__ = [
    "PackedBatch",
    "PackingConfig",
    "TunixPackingConfig",
    "TokenizedExample",
    "install_packing",
    "pack_examples",
    "pack_records",
    "pack_tunix_batches",
    "packed_input_fn",
    "patch_trainer_api",
    "uninstall_packing",
]
