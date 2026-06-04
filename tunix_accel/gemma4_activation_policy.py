"""Gemma4 activation remat/offload policy patch.

This module changes only JAX autodiff storage/rematerialization policy for
Tunix Gemma4 decoder layers. It preserves Gemma4's math, cache behavior,
per-layer inputs, KV sharing, and optional MoE path.
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

_LAYER_INPUT_NAME = "gemma4_decoder_layer_input"
_ATTENTION_RESIDUAL_NAME = "gemma4_attention_residual"
_MLP_RESIDUAL_NAME = "gemma4_mlp_residual"
_PER_LAYER_INPUT_NAME = "gemma4_per_layer_input_residual"


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
    per_layer_input=None,
    kv_shared_cache=None,
):
  original = _STATE.original_decoder_layer_call
  if original is None:
    raise RuntimeError("Gemma4 activation policy patch is not installed.")

  if _STATE.policy == "layer_offload":

    def layer_body(
        layer_self,
        x_arg,
        segment_pos_arg,
        cache_arg,
        attn_mask_arg,
        per_layer_input_arg,
        kv_shared_cache_arg,
    ):
      x_arg = checkpoint_name(x_arg, _LAYER_INPUT_NAME)
      return original(
          layer_self,
          x_arg,
          segment_pos_arg,
          cache_arg,
          attn_mask_arg,
          per_layer_input_arg,
          kv_shared_cache_arg,
      )

    return _checkpointed(layer_body, names=(_LAYER_INPUT_NAME,))(
        self,
        x,
        segment_pos,
        cache,
        attn_mask,
        per_layer_input,
        kv_shared_cache,
    )

  def layer_body(
      layer_self,
      x_arg,
      segment_pos_arg,
      cache_arg,
      attn_mask_arg,
      per_layer_input_arg,
      kv_shared_cache_arg,
  ):
    return original(
        layer_self,
        x_arg,
        segment_pos_arg,
        cache_arg,
        attn_mask_arg,
        per_layer_input_arg,
        kv_shared_cache_arg,
    )

  return _checkpointed(layer_body)(
      self,
      x,
      segment_pos,
      cache,
      attn_mask,
      per_layer_input,
      kv_shared_cache,
  )


def _split_policy_call(
    self,
    x,
    segment_pos,
    cache,
    attn_mask,
    per_layer_input=None,
    kv_shared_cache=None,
):
  """Runs attention and feed-forward paths as separate remat/offload regions."""
  offload = _STATE.policy == "split_offload"

  def attention_body(
      layer_self,
      residual,
      segment_pos_arg,
      cache_arg,
      attn_mask_arg,
      kv_shared_cache_arg,
  ):
    if offload:
      residual = checkpoint_name(residual, _ATTENTION_RESIDUAL_NAME)
    norm = layer_self.pre_attention_norm(residual)
    new_cache, attn, kv = layer_self.attn(
        norm,
        segment_pos_arg,
        cache_arg,
        attn_mask_arg,
        kv_shared_cache=kv_shared_cache_arg,
    )
    attn = layer_self.post_attention_norm(attn)
    return new_cache, attn + residual, kv

  def mlp_body(layer_self, residual, per_layer_input_arg):
    if offload:
      residual = checkpoint_name(residual, _MLP_RESIDUAL_NAME)
      if per_layer_input_arg is not None:
        per_layer_input_arg = checkpoint_name(
            per_layer_input_arg,
            _PER_LAYER_INPUT_NAME,
        )
    ffw = layer_self.pre_ffw_norm(residual)
    ffw = layer_self.mlp(ffw)
    if layer_self.config.enable_moe:
      ffw = layer_self.dense_post_ffw_norm(ffw)
      moe_norm_ffw = layer_self.moe_pre_ffw_norm(residual)
      moe_out = layer_self.moe(moe_norm_ffw)
      moe_out = layer_self.moe_post_ffw_norm(moe_out)
      ffw += moe_out
    ffw = layer_self.post_ffw_norm(ffw)
    ffw += residual

    if (
        layer_self.config.per_layer_input_dim > 0
        and per_layer_input_arg is not None
    ):
      mapped = layer_self.per_layer_input_gate(ffw)
      mapped = jax.nn.gelu(mapped) * per_layer_input_arg
      mapped = layer_self.per_layer_projection(mapped)
      mapped = layer_self.post_per_layer_input_norm(mapped)
      ffw += mapped

    return ffw * layer_self.skip_scale.value

  attention_names = (_ATTENTION_RESIDUAL_NAME,) if offload else ()
  mlp_names = (
      (_MLP_RESIDUAL_NAME, _PER_LAYER_INPUT_NAME)
      if offload
      else ()
  )
  cache, attn_output, kv = _checkpointed(
      attention_body,
      names=attention_names,
  )(
      self,
      x,
      segment_pos,
      cache,
      attn_mask,
      kv_shared_cache,
  )
  outputs = _checkpointed(mlp_body, names=mlp_names)(
      self,
      attn_output,
      per_layer_input,
  )
  return cache, outputs, kv


def _policy_decoder_layer_call(
    self,
    x,
    segment_pos,
    cache,
    attn_mask,
    per_layer_input=None,
    kv_shared_cache=None,
):
  if _STATE.policy in {"layer_remat", "layer_offload"}:
    return _layer_policy_call(
        self,
        x,
        segment_pos,
        cache,
        attn_mask,
        per_layer_input,
        kv_shared_cache,
    )
  if _STATE.policy in {"split_remat", "split_offload"}:
    return _split_policy_call(
        self,
        x,
        segment_pos,
        cache,
        attn_mask,
        per_layer_input,
        kv_shared_cache,
    )

  original = _STATE.original_decoder_layer_call
  if original is None:
    raise RuntimeError("Gemma4 activation policy patch is not installed.")
  return original(
      self,
      x,
      segment_pos,
      cache,
      attn_mask,
      per_layer_input,
      kv_shared_cache,
  )


def install(
    *,
    policy: str = "split_offload",
    prevent_cse: bool = True,
    offload_src: str = "device",
    offload_dst: str = "pinned_host",
) -> None:
  """Installs a process-local activation policy override for Tunix Gemma4."""
  policy = _validate_policy(policy)
  if policy == "none":
    uninstall()
    return

  from tunix.models.gemma4 import model as gemma4_model  # pylint: disable=import-outside-toplevel

  if _STATE.original_decoder_layer_call is None:
    _STATE.original_decoder_layer_call = gemma4_model.DecoderLayer.__call__

  gemma4_model.DecoderLayer.__call__ = _policy_decoder_layer_call
  _STATE.installed = True
  _STATE.policy = policy
  _STATE.prevent_cse = bool(prevent_cse)
  _STATE.offload_src = str(offload_src)
  _STATE.offload_dst = str(offload_dst)


def uninstall() -> None:
  """Restores Tunix Gemma4's original decoder layer call."""
  if _STATE.original_decoder_layer_call is None:
    _STATE.installed = False
    _STATE.policy = "none"
    return

  from tunix.models.gemma4 import model as gemma4_model  # pylint: disable=import-outside-toplevel

  gemma4_model.DecoderLayer.__call__ = _STATE.original_decoder_layer_call
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
