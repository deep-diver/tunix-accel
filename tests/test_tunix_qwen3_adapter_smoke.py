#!/usr/bin/env python3
"""Qwen3 smoke checks for CCE and sequence-packing assumptions."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ["TUNIX_ACCEL_DISABLE_AUTOPATCH"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp

try:
  import pytest
except ModuleNotFoundError:
  pytest = None


def _import_or_skip(module_name: str):
  if pytest is not None:
    return pytest.importorskip(module_name)
  return __import__(module_name, fromlist=["*"])


nnx = _import_or_skip("flax.nnx")
qwen3_model = _import_or_skip("tunix.models.qwen3.model")
peft_trainer = _import_or_skip("tunix.sft.peft_trainer")
utils = _import_or_skip("tunix.sft.utils")

from tunix_accel.tunix_lora_ce import chunked_lm_head_ce_loss_fn


def _tiny_qwen3():
  config = qwen3_model.ModelConfig(
      num_layers=1,
      vocab_size=64,
      embed_dim=32,
      hidden_dim=64,
      num_heads=4,
      head_dim=8,
      num_kv_heads=2,
      rope_theta=10000,
      norm_eps=1e-6,
      use_tied_embedding=True,
      param_dtype=jnp.float32,
  )
  return qwen3_model.Qwen3(config, rngs=nnx.Rngs(0))


def _batch():
  tokens = jnp.array(
      [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
      dtype=jnp.int32,
  )
  input_mask = jnp.ones_like(tokens, dtype=bool)
  positions = utils.build_positions_from_mask(input_mask)
  attention_mask = utils.make_causal_attn_mask(input_mask)
  return tokens, input_mask, positions, attention_mask


def _assert_tree_close(actual, expected, *, atol=5e-5, rtol=5e-5):
  for (actual_path, actual_value), (expected_path, expected_value) in zip(
      nnx.to_flat_state(actual),
      nnx.to_flat_state(expected),
      strict=True,
  ):
    assert actual_path == expected_path
    actual_array = actual_value[...]
    expected_array = expected_value[...]
    max_diff = jnp.max(jnp.abs(actual_array - expected_array))
    assert jnp.allclose(actual_array, expected_array, atol=atol, rtol=rtol), (
        f"{'/'.join(map(str, actual_path))} max_diff={float(max_diff)}"
    )


def test_tiny_qwen3_full_param_cce_gradient_parity():
  model = _tiny_qwen3()
  tokens, input_mask, positions, attention_mask = _batch()

  def default_loss(m):
    return peft_trainer._default_loss_fn(  # pylint: disable=protected-access
        m, tokens, input_mask, positions, attention_mask
    )

  def cce_loss(m):
    return chunked_lm_head_ce_loss_fn(
        token_chunk=3,
        train_lm_head=True,
        vocab_chunk=17,
    )(m, tokens, input_mask, positions, attention_mask)

  assert jnp.allclose(cce_loss(model), default_loss(model), atol=5e-5, rtol=5e-5)
  _assert_tree_close(nnx.grad(cce_loss)(model), nnx.grad(default_loss)(model))


def test_tiny_qwen3_packed_block_attention_matches_separate_segments():
  model = _tiny_qwen3()

  def causal(length: int):
    return jnp.tril(jnp.ones((1, length, length), dtype=bool))

  seq_a = jnp.array([[1, 2, 3]], dtype=jnp.int32)
  seq_b = jnp.array([[4, 5]], dtype=jnp.int32)
  logits_a, _ = model(seq_a, jnp.array([[0, 1, 2]], dtype=jnp.int32), None, causal(3))
  logits_b, _ = model(seq_b, jnp.array([[0, 1]], dtype=jnp.int32), None, causal(2))

  packed = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
  positions = jnp.array([[0, 1, 2, 0, 1]], dtype=jnp.int32)
  segment_ids = jnp.array([[1, 1, 1, 2, 2]], dtype=jnp.int32)
  idx = jnp.arange(5)
  attention_mask = (
      (idx[None, :, None] >= idx[None, None, :])
      & (segment_ids[:, :, None] == segment_ids[:, None, :])
  )
  logits_packed, _ = model(packed, positions, None, attention_mask)

  assert jnp.allclose(logits_packed[:, :3], logits_a, atol=5e-5, rtol=5e-5)
  assert jnp.allclose(logits_packed[:, 3:5], logits_b, atol=5e-5, rtol=5e-5)
