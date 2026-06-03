"""Gemma3 activation remat/offload policy patch.

This module keeps the first activation-policy workstream deliberately narrow:
Gemma3 decoder layers in Tunix. It does not change model math. It only changes
how JAX autodiff saves or rematerializes intermediate activations.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from flax import nnx
import jax
from jax import checkpoint_policies
from jax.ad_checkpoint import checkpoint_name


ActivationPolicy = Literal[
    "none",
    "layer_remat",
    "layer_offload",
    "split_remat",
    "split_offload",
]

_LAYER_INPUT_NAME = "decoder_layer_input"
_ATTENTION_RESIDUAL_NAME = "attention_residual"
_MLP_RESIDUAL_NAME = "mlp_residual"


@dataclass
class _PatchState:
  installed: bool = False
  original_decoder_layer_call: Callable[..., Any] | None = None
  policy: ActivationPolicy = "none"
  prevent_cse: bool = True
  offload_src: str = "device"
  offload_dst: str = "pinned_host"


_STATE = _PatchState()


def _validate_policy(policy: str) -> ActivationPolicy:
  if policy not in {
      "none",
      "layer_remat",
      "layer_offload",
      "split_remat",
      "split_offload",
  }:
    raise ValueError(
        "policy must be one of 'none', 'layer_remat', 'layer_offload', "
        "'split_remat', or 'split_offload', got "
        f"{policy!r}."
    )
  return policy  # type: ignore[return-value]


def _policy_for_names(names: tuple[str, ...]):
  if not names:
    return None
  return checkpoint_policies.save_and_offload_only_these_names(
      names_which_can_be_saved=[],
      names_which_can_be_offloaded=names,
      offload_src=_STATE.offload_src,
      offload_dst=_STATE.offload_dst,
  )


def _checkpointed(fn, *, names: tuple[str, ...] = ()):
  return nnx.remat(
      fn,
      prevent_cse=_STATE.prevent_cse,
      policy=_policy_for_names(names),
  )


def _layer_policy_call(
    self,
    x,
    segment_pos,
    cache,
    attn_mask,
):
  original = _STATE.original_decoder_layer_call
  if original is None:
    raise RuntimeError("Gemma3 activation policy patch is not installed.")

  if _STATE.policy == "layer_offload":

    def layer_body(layer_self, x_arg, segment_pos_arg, cache_arg, attn_mask_arg):
      x_arg = checkpoint_name(x_arg, _LAYER_INPUT_NAME)
      return original(layer_self, x_arg, segment_pos_arg, cache_arg, attn_mask_arg)

    return _checkpointed(layer_body, names=(_LAYER_INPUT_NAME,))(
        self,
        x,
        segment_pos,
        cache,
        attn_mask,
    )

  def layer_body(layer_self, x_arg, segment_pos_arg, cache_arg, attn_mask_arg):
    return original(layer_self, x_arg, segment_pos_arg, cache_arg, attn_mask_arg)

  return _checkpointed(layer_body)(
      self,
      x,
      segment_pos,
      cache,
      attn_mask,
  )


def _split_policy_call(
    self,
    x,
    segment_pos,
    cache,
    attn_mask,
):
  """Runs attention and MLP as separate remat/offload regions."""
  offload = _STATE.policy == "split_offload"

  def attention_body(
      layer_self,
      residual,
      segment_pos_arg,
      cache_arg,
      attn_mask_arg,
  ):
    if offload:
      residual = checkpoint_name(residual, _ATTENTION_RESIDUAL_NAME)
    inputs_normalized = layer_self.pre_attention_norm(residual)
    new_cache, attn_output = layer_self.attn(
        inputs_normalized,
        segment_pos_arg,
        cache_arg,
        attn_mask_arg,
    )
    attn_output = layer_self.post_attention_norm(attn_output)
    return new_cache, attn_output + residual

  def mlp_body(layer_self, residual):
    if offload:
      residual = checkpoint_name(residual, _MLP_RESIDUAL_NAME)
    outputs = layer_self.pre_ffw_norm(residual)
    outputs = layer_self.mlp(outputs)
    outputs = layer_self.post_ffw_norm(outputs)
    return outputs + residual

  attention_names = (_ATTENTION_RESIDUAL_NAME,) if offload else ()
  mlp_names = (_MLP_RESIDUAL_NAME,) if offload else ()
  cache, attn_output = _checkpointed(attention_body, names=attention_names)(
      self,
      x,
      segment_pos,
      cache,
      attn_mask,
  )
  outputs = _checkpointed(mlp_body, names=mlp_names)(self, attn_output)
  return cache, outputs


def _policy_decoder_layer_call(self, x, segment_pos, cache, attn_mask):
  if _STATE.policy in {"layer_remat", "layer_offload"}:
    return _layer_policy_call(self, x, segment_pos, cache, attn_mask)
  if _STATE.policy in {"split_remat", "split_offload"}:
    return _split_policy_call(self, x, segment_pos, cache, attn_mask)

  original = _STATE.original_decoder_layer_call
  if original is None:
    raise RuntimeError("Gemma3 activation policy patch is not installed.")
  return original(self, x, segment_pos, cache, attn_mask)


def install(
    *,
    policy: str = "split_offload",
    prevent_cse: bool = True,
    offload_src: str = "device",
    offload_dst: str = "pinned_host",
) -> None:
  """Installs a process-local activation policy override for Tunix Gemma3."""
  policy = _validate_policy(policy)
  if policy == "none":
    uninstall()
    return

  from tunix.models.gemma3 import model as gemma3_model  # pylint: disable=import-outside-toplevel

  if _STATE.original_decoder_layer_call is None:
    _STATE.original_decoder_layer_call = gemma3_model.DecoderLayer.__call__

  gemma3_model.DecoderLayer.__call__ = _policy_decoder_layer_call
  _STATE.installed = True
  _STATE.policy = policy
  _STATE.prevent_cse = bool(prevent_cse)
  _STATE.offload_src = str(offload_src)
  _STATE.offload_dst = str(offload_dst)


def uninstall() -> None:
  """Restores Tunix Gemma3's original decoder layer call."""
  if _STATE.original_decoder_layer_call is None:
    _STATE.installed = False
    _STATE.policy = "none"
    return

  from tunix.models.gemma3 import model as gemma3_model  # pylint: disable=import-outside-toplevel

  gemma3_model.DecoderLayer.__call__ = _STATE.original_decoder_layer_call
  _STATE.installed = False
  _STATE.policy = "none"


def is_installed() -> bool:
  return _STATE.installed


@contextmanager
def installed(
    *,
    policy: str = "split_offload",
    prevent_cse: bool = True,
    offload_src: str = "device",
    offload_dst: str = "pinned_host",
):
  """Context manager form of `install()`."""
  was_installed = _STATE.installed
  old_policy = _STATE.policy
  old_prevent_cse = _STATE.prevent_cse
  old_offload_src = _STATE.offload_src
  old_offload_dst = _STATE.offload_dst
  install(
      policy=policy,
      prevent_cse=prevent_cse,
      offload_src=offload_src,
      offload_dst=offload_dst,
  )
  try:
    yield
  finally:
    if was_installed:
      install(
          policy=old_policy,
          prevent_cse=old_prevent_cse,
          offload_src=old_offload_src,
          offload_dst=old_offload_dst,
      )
    else:
      uninstall()


__all__ = [
    "ActivationPolicy",
    "install",
    "installed",
    "is_installed",
    "uninstall",
]
