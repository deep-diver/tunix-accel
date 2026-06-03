#!/usr/bin/env python3
"""Gemma3 integration checks for activation remat/offload policies."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ["TUNIX_ACCEL_DISABLE_AUTOPATCH"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import pytest

nnx = pytest.importorskip("flax.nnx", exc_type=ImportError)
gemma3_model = pytest.importorskip("tunix.models.gemma3.model", exc_type=ImportError)
peft_trainer = pytest.importorskip("tunix.sft.peft_trainer", exc_type=ImportError)
utils = pytest.importorskip("tunix.sft.utils", exc_type=ImportError)

from tunix_accel import gemma3_activation_policy


def _tiny_model():
  config = gemma3_model.ModelConfig(
      num_layers=1,
      num_embed=64,
      embed_dim=32,
      hidden_dim=64,
      num_heads=4,
      head_dim=8,
      num_kv_heads=1,
      sliding_window_size=8,
      remat_config=gemma3_model.RematConfig.NONE,
      param_dtype=jnp.float32,
  )
  return gemma3_model.Gemma3(config, rngs=nnx.Rngs(0))


def _set_scale(module, value=1.0) -> None:
  module.scale[...] = jnp.full(module.scale.shape, value, dtype=jnp.float32)


def _randomize_layer(layer) -> None:
  keys = iter(jax.random.split(jax.random.key(1234), 16))
  for norm in (
      layer.pre_attention_norm,
      layer.post_attention_norm,
      layer.pre_ffw_norm,
      layer.post_ffw_norm,
      layer.attn._query_norm,
      layer.attn._key_norm,
  ):
    _set_scale(norm, 1.0)

  for einsum in (
      layer.attn.q_einsum,
      layer.attn.kv_einsum,
      layer.attn.attn_vec_einsum,
  ):
    key = next(keys)
    scale = jnp.sqrt(jnp.asarray(einsum.w.shape[-2], dtype=jnp.float32))
    einsum.w[...] = (
        jax.random.normal(key, einsum.w.shape, dtype=jnp.float32) / scale
    )

  for projection in (
      layer.mlp.gate_proj,
      layer.mlp.up_proj,
      layer.mlp.down_proj,
  ):
    key = next(keys)
    scale = jnp.sqrt(jnp.asarray(projection.kernel.shape[0], dtype=jnp.float32))
    projection.kernel[...] = (
        jax.random.normal(key, projection.kernel.shape, dtype=jnp.float32)
        / scale
    )


def _inputs():
  x = jax.random.normal(jax.random.key(5), (2, 7, 32), dtype=jnp.float32)
  mask = jnp.ones((2, 7), dtype=bool)
  positions = utils.build_positions_from_mask(mask)
  attention_mask = utils.make_causal_attn_mask(mask)
  return x, positions, attention_mask


def _batch():
  tokens = jnp.array(
      [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
      dtype=jnp.int32,
  )
  input_mask = jnp.ones_like(tokens, dtype=bool)
  positions = utils.build_positions_from_mask(input_mask)
  attention_mask = utils.make_causal_attn_mask(input_mask)
  return tokens, input_mask, positions, attention_mask


def _tree_allclose(actual, expected, *, atol, rtol) -> None:
  actual_leaves = jax.tree.leaves(actual)
  expected_leaves = jax.tree.leaves(expected)
  assert len(actual_leaves) == len(expected_leaves)
  for actual_leaf, expected_leaf in zip(
      actual_leaves,
      expected_leaves,
      strict=True,
  ):
    assert jnp.allclose(actual_leaf, expected_leaf, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "policy",
    ["layer_remat", "layer_offload", "split_remat", "split_offload"],
)
def test_activation_policy_layer_call_matches_original(policy):
  model = _tiny_model()
  layer = model.layers[0]
  _randomize_layer(layer)
  x, positions, attention_mask = _inputs()

  _, expected = layer(x, positions, None, attention_mask)
  with gemma3_activation_policy.installed(policy=policy, prevent_cse=False):
    _, actual = layer(x, positions, None, attention_mask)

  assert jnp.allclose(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize(
    "policy",
    ["layer_remat", "layer_offload", "split_remat", "split_offload"],
)
def test_activation_policy_model_gradients_match_original(policy):
  model = _tiny_model()
  layer = model.layers[0]
  _randomize_layer(layer)
  tokens, input_mask, positions, attention_mask = _batch()

  def loss_fn(model_arg):
    return peft_trainer._default_loss_fn(  # pylint: disable=protected-access
        model_arg,
        tokens,
        input_mask,
        positions,
        attention_mask,
    )

  expected_loss, expected_grads = nnx.value_and_grad(loss_fn)(model)
  with gemma3_activation_policy.installed(policy=policy, prevent_cse=False):
    actual_loss, actual_grads = nnx.value_and_grad(loss_fn)(model)

  assert jnp.allclose(actual_loss, expected_loss, atol=3e-5, rtol=3e-5)
  _tree_allclose(actual_grads, expected_grads, atol=4e-5, rtol=4e-5)


def test_activation_policy_context_manager_restores_original_call():
  model = _tiny_model()
  layer = model.layers[0]
  original_call = gemma3_model.DecoderLayer.__call__

  assert not gemma3_activation_policy.is_installed()
  with gemma3_activation_policy.installed(policy="split_remat"):
    assert gemma3_activation_policy.is_installed()
    assert gemma3_model.DecoderLayer.__call__ is not original_call
    x, positions, attention_mask = _inputs()
    _, _ = layer(x, positions, None, attention_mask)

  assert not gemma3_activation_policy.is_installed()
  assert gemma3_model.DecoderLayer.__call__ is original_call
