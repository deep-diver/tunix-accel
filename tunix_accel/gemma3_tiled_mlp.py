"""Gemma3-only drop-in patch for tiled gated MLP blocks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp

from tunix_accel.tiled_mlp import tiled_gated_mlp


@dataclass
class _PatchState:
  installed: bool = False
  original_block: Any | None = None
  token_chunk: int = 128
  fallback_to_original_on_lora: bool = True
  lora_alpha: float = 32.0


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


def _intermediate_sharding(module: nnx.Module) -> tuple[str | None, ...] | None:
  try:
    return tuple(module.config.shd_config.act_btf)
  except AttributeError:
    return None


def _loop_tiled_original_block(module: nnx.Module, x):
  """Tiles Gemma's original block while preserving projection/provider semantics."""
  if _STATE.original_block is None:
    raise RuntimeError("Gemma3 tiled MLP patch is missing the original block.")
  axis = x.ndim - 2
  n_tokens = int(x.shape[axis])
  token_chunk = min(_STATE.token_chunk, n_tokens)
  chunks = (n_tokens + token_chunk - 1) // token_chunk
  padded_tokens = chunks * token_chunk
  pad_tokens = padded_tokens - n_tokens
  if pad_tokens:
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_tokens)
    x_padded = jnp.pad(x, pad_width)
  else:
    x_padded = x
  out_padded = jnp.zeros_like(x_padded)

  def body(i, out):
    start = i * token_chunk
    starts = [0] * x_padded.ndim
    starts[axis] = start
    sizes = list(x_padded.shape)
    sizes[axis] = token_chunk
    tile = jax.lax.dynamic_slice(x_padded, tuple(starts), tuple(sizes))
    tile_out = _STATE.original_block(module, tile)
    return jax.lax.dynamic_update_slice(out, tile_out, tuple(starts))

  out_padded = jax.lax.fori_loop(0, chunks, body, out_padded)
  if not pad_tokens:
    return out_padded
  index = [slice(None)] * x.ndim
  index[axis] = slice(0, n_tokens)
  return out_padded[tuple(index)]


def _tiled_block(self, x):  # pylint: disable=protected-access
  if not _is_supported_gemma3_mlp(self):
    if _STATE.original_block is not None:
      return _STATE.original_block(self, x)
    raise TypeError(f"Unsupported Gemma3 MLP module: {type(self)}")

  if _has_projection_lora(self):
    return _loop_tiled_original_block(self, x)

  return tiled_gated_mlp(
      x,
      _projection_kernel(self, "gate_proj"),
      _projection_kernel(self, "up_proj"),
      _projection_kernel(self, "down_proj"),
      token_chunk=_STATE.token_chunk,
      activation="gelu_approx",
      intermediate_sharding=_intermediate_sharding(self),
  )


def install(
    *,
    token_chunk: int = 128,
    fallback_to_original_on_lora: bool = True,
    lora_alpha: float = 32.0,
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
  _STATE.lora_alpha = float(lora_alpha)


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
    lora_alpha: float = 32.0,
):
  """Context manager form of `install()`."""
  was_installed = _STATE.installed
  old_token_chunk = _STATE.token_chunk
  old_fallback = _STATE.fallback_to_original_on_lora
  old_lora_alpha = _STATE.lora_alpha
  install(
      token_chunk=token_chunk,
      fallback_to_original_on_lora=fallback_to_original_on_lora,
      lora_alpha=lora_alpha,
  )
  try:
    yield
  finally:
    if was_installed:
      install(
          token_chunk=old_token_chunk,
          fallback_to_original_on_lora=old_fallback,
          lora_alpha=old_lora_alpha,
      )
    else:
      uninstall()


def validate_gemma3_model(model: nnx.Module, *, require_no_lora: bool = False) -> None:
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
          "was asked to reject LoRA projection kernels."
      )
