"""Gemma4 drop-in patch for tiled gated MLP blocks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from flax import nnx
import jax

from tunix_accel.tiled_mlp import tiled_gated_mlp
from tunix_accel.tiled_mlp import tiled_lora_gated_mlp


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


def _is_supported_gemma4_mlp(module: nnx.Module) -> bool:
  return all(
      hasattr(module, name)
      for name in ("config", "gate_proj", "up_proj", "down_proj")
  )


def _projection_kernel(module: nnx.Module, name: str):
  projection = getattr(module, name)
  if getattr(projection, "bias", None) is not None:
    raise TypeError(
        f"Gemma4 tiled MLP only supports bias-free projections; {name} has bias."
    )
  return projection.kernel[...]


def _projection_lora(module: nnx.Module, name: str):
  projection = getattr(module, name)
  if getattr(projection, "bias", None) is not None:
    raise TypeError(
        f"Gemma4 tiled MLP only supports bias-free projections; {name} has bias."
    )
  if not hasattr(projection, "kernel_lora_a") or not hasattr(
      projection,
      "kernel_lora_b",
  ):
    raise TypeError(f"{name} does not expose Qwix LoRA projection params.")
  return (
      projection.kernel[...],
      projection.kernel_lora_a[...],
      projection.kernel_lora_b[...],
  )


def _lora_scale(*lora_as) -> float:
  ranks = {int(lora_a.shape[-1]) for lora_a in lora_as}
  if len(ranks) != 1:
    raise TypeError(f"Expected a single LoRA rank across MLP projections, got {ranks}.")
  rank = next(iter(ranks))
  if rank <= 0:
    raise TypeError(f"LoRA rank must be positive, got {rank}.")
  return float(_STATE.lora_alpha) / float(rank)


def _intermediate_sharding(module: nnx.Module) -> tuple[str | None, ...] | None:
  try:
    return tuple(module.config.shd_config.act_btf)
  except AttributeError:
    return None


def _tiled_block(self, x):  # pylint: disable=protected-access
  if not _is_supported_gemma4_mlp(self):
    if _STATE.original_block is not None:
      return _STATE.original_block(self, x)
    raise TypeError(f"Unsupported Gemma4 MLP module: {type(self)}")

  if _has_projection_lora(self):
    try:
      gate_kernel, gate_lora_a, gate_lora_b = _projection_lora(self, "gate_proj")
      up_kernel, up_lora_a, up_lora_b = _projection_lora(self, "up_proj")
      down_kernel, down_lora_a, down_lora_b = _projection_lora(self, "down_proj")
    except TypeError:
      if _STATE.fallback_to_original_on_lora and _STATE.original_block is not None:
        return _STATE.original_block(self, x)
      raise

    return tiled_lora_gated_mlp(
        x,
        gate_kernel,
        gate_lora_a,
        gate_lora_b,
        up_kernel,
        up_lora_a,
        up_lora_b,
        down_kernel,
        down_lora_a,
        down_lora_b,
        token_chunk=_STATE.token_chunk,
        activation="gelu_approx",
        lora_scale=_lora_scale(gate_lora_a, up_lora_a, down_lora_a),
        intermediate_sharding=_intermediate_sharding(self),
    )

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
  """Installs a process-local tiled MLP override for Tunix Gemma4.

  The patch replaces `tunix.models.gemma4.model.FeedForward.block`, preserving
  Gemma4's original `FeedForward.__call__` and remat behavior. Existing model
  instances and newly-created instances both see the replacement until
  `uninstall()` is called.
  """
  if token_chunk <= 0:
    raise ValueError(f"token_chunk must be positive, got {token_chunk}.")

  from tunix.models.gemma4 import model as gemma4_model  # pylint: disable=import-outside-toplevel

  if not _STATE.installed:
    _STATE.original_block = gemma4_model.FeedForward.block
    gemma4_model.FeedForward.block = _tiled_block
    _STATE.installed = True

  _STATE.token_chunk = int(token_chunk)
  _STATE.fallback_to_original_on_lora = bool(fallback_to_original_on_lora)
  _STATE.lora_alpha = float(lora_alpha)


def uninstall() -> None:
  """Restores Tunix Gemma4's original FeedForward block."""
  if not _STATE.installed:
    return

  from tunix.models.gemma4 import model as gemma4_model  # pylint: disable=import-outside-toplevel

  gemma4_model.FeedForward.block = _STATE.original_block
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


def validate_gemma4_model(model: nnx.Module, *, require_no_lora: bool = False) -> None:
  """Validates that a model is compatible with the Gemma4 tiled-MLP patch."""
  if not hasattr(model, "layers"):
    raise TypeError(f"Expected a Tunix Gemma4-like model with layers, got {type(model)}")
  for index, layer in enumerate(model.layers):
    mlp = getattr(layer, "mlp", None)
    if mlp is None or not _is_supported_gemma4_mlp(mlp):
      raise TypeError(f"Layer {index} does not expose a supported Gemma4 MLP.")
    if require_no_lora and _has_projection_lora(mlp):
      raise TypeError(
          f"Layer {index} has LoRA projection params; Gemma4 tiled MLP currently "
          "was asked to reject LoRA projection kernels."
      )
