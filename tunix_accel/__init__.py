"""Drop-in acceleration and memory-efficiency patches for JAX/Tunix training."""

from tunix_accel.packing import PackingConfig
from tunix_accel.packing import PackedBatch
from tunix_accel.packing import TokenizedExample
from tunix_accel.packing import pack_examples
from tunix_accel.packing import pack_records


__all__ = [
    "PackedBatch",
    "PackingConfig",
    "TokenizedExample",
    "pack_examples",
    "pack_records",
]
