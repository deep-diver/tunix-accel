"""Gemma3-only drop-in patch for TPU Splash Attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P


@dataclass
class _State:
  installed: bool = False
  original_block: Any | None = None
  interpret: bool = False


_STATE = _State()


def _static_splash_mask(attn_type: Any, sliding_window_size: int, heads: int, seq_len: int):
  from jax.experimental.pallas.ops.tpu.splash_attention import (  # pylint: disable=import-outside-toplevel
      splash_attention_mask as mask_lib,
  )
  from tunix.models.gemma3 import model as gemma3_model  # pylint: disable=import-outside-toplevel

  if attn_type == gemma3_model.AttentionType.LOCAL_SLIDING:
    base = mask_lib.make_local_attention_mask(
        (seq_len, seq_len),
        (sliding_window_size - 1, 0),
    )
  else:
    base = mask_lib.make_causal_mask((seq_len, seq_len))
  return np.broadcast_to(base, (heads, seq_len, seq_len)).copy()


def _splash_gqa(query_scaled, key_proj, value_proj, *, attn_type, sliding_window_size):
  from jax.experimental.pallas.ops.tpu import splash_attention  # pylint: disable=import-outside-toplevel

  batch, seq_len, query_heads, head_dim = query_scaled.shape
  kv_heads = key_proj.shape[2]
  groups = query_heads // kv_heads
  if query_heads % kv_heads != 0:
    raise ValueError(
        f"Gemma3 GQA requires query heads divisible by KV heads, got "
        f"{query_heads=} {kv_heads=}."
    )

  mask = _static_splash_mask(
      attn_type,
      sliding_window_size,
      groups,
      seq_len,
  )
  kernel = splash_attention.make_splash_mqa(
      mask=mask,
      block_sizes=splash_attention.BlockSizes.get_default(),
      head_shards=1,
      q_seq_shards=1,
      interpret=_STATE.interpret,
  )

  q_by_kv = query_scaled.reshape(
      batch,
      seq_len,
      kv_heads,
      groups,
      head_dim,
  )
  q_by_kv = jnp.transpose(q_by_kv, (0, 2, 3, 1, 4))
  k_by_kv = jnp.transpose(key_proj, (0, 2, 1, 3))
  v_by_kv = jnp.transpose(value_proj, (0, 2, 1, 3))

  def call_kernel(q, k, v):
    return _call_kernel_in_current_mesh(kernel, q, k, v)

  def one_batch(q_b, k_b, v_b):
    return jax.vmap(call_kernel, in_axes=(0, 0, 0))(q_b, k_b, v_b)

  encoded = jax.vmap(one_batch, in_axes=(0, 0, 0))(q_by_kv, k_by_kv, v_by_kv)
  encoded = jnp.transpose(encoded, (0, 3, 1, 2, 4))
  return encoded.reshape(batch, seq_len, query_heads, head_dim)


def _call_kernel_in_current_mesh(kernel, q, k, v):
  mesh = jax.sharding.get_abstract_mesh()
  if getattr(mesh, "empty", False):
    return kernel(q, k, v)
  wrapped = jax.shard_map(
      kernel,
      mesh=mesh,
      in_specs=(P(), P(), P()),
      out_specs=P(),
      check_vma=False,
  )
  return wrapped(q, k, v)


def _splash_attention_block(self, x, segment_pos, cache, attn_mask):  # pylint: disable=unused-argument
  if cache is not None:
    return _STATE.original_block(self, x, segment_pos, cache, attn_mask)

  from tunix.models.gemma3 import model as gemma3_model  # pylint: disable=import-outside-toplevel
  from tunix.models.gemma3.model import apply_rope  # pylint: disable=import-outside-toplevel
  from tunix.utils import sharding_utils  # pylint: disable=import-outside-toplevel

  seq_len = x.shape[1]
  block_q = 128
  if seq_len < block_q or seq_len % block_q != 0:
    return _STATE.original_block(self, x, segment_pos, cache, attn_mask)

  if self.use_qkv_einsum:
    query_proj, key_proj, value_proj = self.qkv_einsum(x)
  else:
    query_proj = self.q_einsum(x)
    key_proj, value_proj = self.kv_einsum(x)

  query_proj = sharding_utils.shard(query_proj, self.shd_config.act_btnh)
  key_proj = sharding_utils.shard(key_proj, self.shd_config.act_btnh)
  value_proj = sharding_utils.shard(value_proj, self.shd_config.act_btnh)

  query_proj = self._query_norm(query_proj)
  key_proj = self._key_norm(key_proj)

  query_proj = apply_rope(
      query_proj,
      segment_pos,
      head_dim=self.head_dim,
      base_frequency=self.rope_base_frequency,
      scale_factor=self.rope_scale_factor,
  )
  query_scaled = query_proj * self.query_pre_attn_scalar
  key_proj = apply_rope(
      key_proj,
      segment_pos,
      head_dim=self.head_dim,
      base_frequency=self.rope_base_frequency,
      scale_factor=self.rope_scale_factor,
  )

  if self.use_gqa:
    encoded = _splash_gqa(
        query_scaled,
        key_proj,
        value_proj,
        attn_type=self.attn_type,
        sliding_window_size=self.sliding_window_size,
    )
  elif query_scaled.shape[2] == key_proj.shape[2]:
    heads = query_scaled.shape[2]
    mask = _static_splash_mask(
        self.attn_type,
        self.sliding_window_size,
        heads,
        seq_len,
    )
    from jax.experimental.pallas.ops.tpu import splash_attention  # pylint: disable=import-outside-toplevel

    kernel = splash_attention.make_splash_mha(
        mask=mask,
        block_sizes=splash_attention.BlockSizes.get_default(),
        head_shards=1,
        q_seq_shards=1,
        interpret=_STATE.interpret,
    )
    q = jnp.transpose(query_scaled, (0, 2, 1, 3))
    k = jnp.transpose(key_proj, (0, 2, 1, 3))
    v = jnp.transpose(value_proj, (0, 2, 1, 3))
    encoded = jax.vmap(
        lambda q_i, k_i, v_i: _call_kernel_in_current_mesh(
            kernel,
            q_i,
            k_i,
            v_i,
        ),
        in_axes=(0, 0, 0),
    )(q, k, v)
    encoded = jnp.transpose(encoded, (0, 2, 1, 3))
  else:
    return _STATE.original_block(self, x, segment_pos, cache, attn_mask)

  attn_output = self.attn_vec_einsum(encoded)
  attn_output = sharding_utils.shard(attn_output, self.shd_config.act_btd)
  return None, attn_output


def install(*, interpret: bool = False) -> None:
  from tunix.models.gemma3 import model as gemma3_model  # pylint: disable=import-outside-toplevel

  if _STATE.installed:
    _STATE.interpret = interpret
    return
  _STATE.original_block = gemma3_model.Attention.block
  _STATE.interpret = interpret
  gemma3_model.Attention.block = _splash_attention_block
  _STATE.installed = True


def uninstall() -> None:
  if not _STATE.installed:
    return
  from tunix.models.gemma3 import model as gemma3_model  # pylint: disable=import-outside-toplevel

  gemma3_model.Attention.block = _STATE.original_block
  _STATE.original_block = None
  _STATE.installed = False


def is_installed() -> bool:
  return _STATE.installed
