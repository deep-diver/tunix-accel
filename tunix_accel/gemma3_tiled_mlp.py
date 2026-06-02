"""Gemma3-only drop-in patch for tiled gated MLP blocks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from flax import nnx
import jax

from tunix_accel.tiled_mlp import tiled_gated_mlp


@dataclass
class _PatchState:
  installed: bool = False
  original_block: Any | None = None
  token_chunk: int = 128
  fallback_to_original_on_lora: bool = True


_STATE = _PatchState()


def _has_lora_params(module: nnx.Module) -> bool:
  try:
    return bool(jax.tree_util.tree_leaves(nnx.state(module, nnx.LoRAParam)))
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def _has_projection_lora(module: nnx.Module) -> bool:
  return any(
      hasattr(getattr(module, name), "kernel_lora_a")
      or hasattr(getattr(module, name), "kernel_lora_b")
      for name in ("gate_proj", "up_proj", "down_proj")
      if hasattr(module, name)
  ) or _has_lora_params(module)


def _is_supported_gemma3_mlp(module: nnx.Module) -> bool:
  return all(
      hasattr(module, name)
      for name in ("config", "gate_proj", "up_proj", "down_proj")
  )


def _projection_kernel(module: nnx.Module, name: str):
  projection = getattr(module, name)
  if getattr(projection, "bias", None) is not None:
    raise TypeError(
        f"Gemma3 tiled MLP only supports bias-free projections; {name} has bias."
    )
  return projection.kernel[...]


def _tiled_block(self, x):  # pylint: disable=protected-access
  if not _is_supported_gemma3_mlp(self):
    if _STATE.original_block is not None:
      return _STATE.original_block(self, x)
    raise TypeError(f"Unsupported Gemma3 MLP module: {type(self)}")

  if _has_projection_lora(self):
    if _STATE.fallback_to_original_on_lora and _STATE.original_block is not None:
      return _STATE.original_block(self, x)
    raise TypeError(
        "Gemma3 tiled MLP does not support Qwix-LoRA projection deltas yet. "
        "Use fallback_to_original_on_lora=True or disable the tiled MLP patch "
        "for LoRA runs."
    )

  return tiled_gated_mlp(
      x,
      _projection_kernel(self, "gate_proj"),
      _projection_kernel(self, "up_proj"),
      _projection_kernel(self, "down_proj"),
      token_chunk=_STATE.token_chunk,
      activation="gelu_approx",
  )


def install(
    *,
    token_chunk: int = 128,
    fallback_to_original_on_lora: bool = True,
) -> None:
  """Installs a process-local tiled MLP override for Tunix Gemma3.

  The patch replaces `tunix.models.gemma3.model.FeedForward.block`, which keeps
  the original Gemma3 `FeedForward.__call__` and its remat behavior intact.
  Existing model instances and newly-created instances both see the replacement
  until `uninstall()` is called.
  """
  if token_chunk <= 0:
    raise ValueError(f"token_chunk must be positive, got {token_chunk}.")

  from tunix.models.gemma3 import model as gemma3_model  # pylint: disable=import-outside-toplevel

  if not _STATE.installed:
    _STATE.original_block = gemma3_model.FeedForward.block
    gemma3_model.FeedForward.block = _tiled_block
    _STATE.installed = True

  _STATE.token_chunk = int(token_chunk)
  _STATE.fallback_to_original_on_lora = bool(fallback_to_original_on_lora)


def uninstall() -> None:
  """Restores Tunix Gemma3's original FeedForward block."""
  if not _STATE.installed:
    return

  from tunix.models.gemma3 import model as gemma3_model  # pylint: disable=import-outside-toplevel

  gemma3_model.FeedForward.block = _STATE.original_block
  _STATE.installed = False


def is_installed() -> bool:
  return _STATE.installed


@contextmanager
def installed(
    *,
    token_chunk: int = 128,
    fallback_to_original_on_lora: bool = True,
):
  """Context manager form of `install()`."""
  was_installed = _STATE.installed
  old_token_chunk = _STATE.token_chunk
  old_fallback = _STATE.fallback_to_original_on_lora
  install(
      token_chunk=token_chunk,
      fallback_to_original_on_lora=fallback_to_original_on_lora,
  )
  try:
    yield
  finally:
    if was_installed:
      install(
          token_chunk=old_token_chunk,
          fallback_to_original_on_lora=old_fallback,
      )
    else:
      uninstall()


def validate_gemma3_model(model: nnx.Module, *, require_no_lora: bool = True) -> None:
  """Validates that a model is compatible with the first Gemma3 tiled-MLP patch."""
  if not hasattr(model, "layers"):
    raise TypeError(f"Expected a Tunix Gemma3-like model with layers, got {type(model)}")
  for index, layer in enumerate(model.layers):
    mlp = getattr(layer, "mlp", None)
    if mlp is None or not _is_supported_gemma3_mlp(mlp):
      raise TypeError(f"Layer {index} does not expose a supported Gemma3 MLP.")
    if require_no_lora and _has_projection_lora(mlp):
      raise TypeError(
          f"Layer {index} has LoRA projection params; Gemma3 tiled MLP currently "
          "supports full-parameter/frozen-base projection kernels only."
      )
