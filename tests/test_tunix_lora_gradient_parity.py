#!/usr/bin/env python3
"""Gradient parity checks for Tunix Gemma3 + Qwix LoRA integration."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ["TUNIX_ACCEL_DISABLE_AUTOPATCH"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
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
qwix = _import_or_skip("qwix")
gemma3_model = _import_or_skip("tunix.models.gemma3.model")
peft_trainer = _import_or_skip("tunix.sft.peft_trainer")
utils = _import_or_skip("tunix.sft.utils")

from tunix_accel.tunix_lora_ce import chunked_lm_head_ce_loss_fn
from tunix_accel import model_adapters


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
      param_dtype=jnp.float32,
  )
  return gemma3_model.Gemma3(config, rngs=nnx.Rngs(0))


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


def test_tiny_gemma_full_param_gradient_parity():
  model = _tiny_model()
  tokens, input_mask, positions, attention_mask = _batch()

  def default_loss(m):
    return peft_trainer._default_loss_fn(  # pylint: disable=protected-access
        m, tokens, input_mask, positions, attention_mask
    )

  def chunked_loss(m):
    return chunked_lm_head_ce_loss_fn(
        token_chunk=3,
        train_lm_head=True,
        vocab_chunk=17,
    )(m, tokens, input_mask, positions, attention_mask)

  assert jnp.allclose(chunked_loss(model), default_loss(model), atol=5e-5, rtol=5e-5)
  _assert_tree_close(nnx.grad(chunked_loss)(model), nnx.grad(default_loss)(model))


def test_tiny_gemma_qwix_lora_gradient_parity():
  base_model = _tiny_model()
  provider = qwix.LoraProvider(
      module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
      rank=4,
      alpha=8,
  )
  model = qwix.apply_lora_to_model(
      base_model,
      provider,
      **base_model.get_model_input(),
      rngs=nnx.Rngs(1),
  )
  chunked_model = nnx.clone(model)
  model_adapters.prepare_intercepted_lora_model(chunked_model)
  tokens, input_mask, positions, attention_mask = _batch()

  def default_loss(m):
    return peft_trainer._default_loss_fn(  # pylint: disable=protected-access
        m, tokens, input_mask, positions, attention_mask
    )

  def chunked_loss(m):
    return chunked_lm_head_ce_loss_fn(
        token_chunk=3,
        train_lm_head=False,
        vocab_chunk=17,
    )(m, tokens, input_mask, positions, attention_mask)

  assert jnp.allclose(
      chunked_loss(chunked_model), default_loss(model), atol=5e-5, rtol=5e-5
  )
  diff_state = nnx.DiffState(0, nnx.LoRAParam)
  _assert_tree_close(
      nnx.grad(chunked_loss, argnums=diff_state)(chunked_model),
      nnx.grad(default_loss, argnums=diff_state)(model),
  )


def test_lora_decode_restore_after_hidden_intercept():
  base_model = _tiny_model()
  provider = qwix.LoraProvider(
      module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
      rank=4,
      alpha=8,
  )
  model = qwix.apply_lora_to_model(
      base_model,
      provider,
      **base_model.get_model_input(),
      rngs=nnx.Rngs(1),
  )
  tokens, input_mask, positions, attention_mask = _batch()
  logits_before, _ = model(tokens, positions, None, attention_mask)
  assert logits_before.shape[-1] == model.config.num_embed
  assert model_adapters.prepare_intercepted_lora_model(model)
  hidden, _ = model(tokens, positions, None, attention_mask)
  assert hidden.shape[-1] == model.config.embed_dim
  assert model_adapters.restore_intercepted_lora_model(model)
  logits_after, _ = model(tokens, positions, None, attention_mask)
  assert logits_after.shape[-1] == model.config.num_embed
  assert not getattr(model.embedder, "_tunix_accel_decode_identity", False)


def test_tiny_gemma_qwix_lora_jit_loss_parity():
  base_model = _tiny_model()
  provider = qwix.LoraProvider(
      module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
      rank=4,
      alpha=8,
  )
  model = qwix.apply_lora_to_model(
      base_model,
      provider,
      **base_model.get_model_input(),
      rngs=nnx.Rngs(1),
  )
  chunked_model = nnx.clone(model)
  model_adapters.prepare_intercepted_lora_model(chunked_model)
  tokens, input_mask, positions, attention_mask = _batch()

  @nnx.jit
  def default_loss(m):
    return peft_trainer._default_loss_fn(  # pylint: disable=protected-access
        m, tokens, input_mask, positions, attention_mask
    )

  @nnx.jit
  def chunked_loss(m):
    return chunked_lm_head_ce_loss_fn(
        token_chunk=3,
        train_lm_head=False,
        vocab_chunk=17,
    )(m, tokens, input_mask, positions, attention_mask)

  assert jnp.allclose(
      chunked_loss(chunked_model), default_loss(model), atol=5e-5, rtol=5e-5
  )


def main() -> None:
  test_tiny_gemma_full_param_gradient_parity()
  print("tiny_gemma_full_param_gradient_parity=ok")
  test_tiny_gemma_qwix_lora_gradient_parity()
  print("tiny_gemma_qwix_lora_gradient_parity=ok")
  test_tiny_gemma_qwix_lora_jit_loss_parity()
  print("tiny_gemma_qwix_lora_jit_loss_parity=ok")


if __name__ == "__main__":
  main()
