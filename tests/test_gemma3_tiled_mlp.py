#!/usr/bin/env python3
"""Gemma3 integration checks for the tiled MLP patch."""

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
qwix = pytest.importorskip("qwix", exc_type=ImportError)
gemma3_model = pytest.importorskip("tunix.models.gemma3.model", exc_type=ImportError)
peft_trainer = pytest.importorskip("tunix.sft.peft_trainer", exc_type=ImportError)
utils = pytest.importorskip("tunix.sft.utils", exc_type=ImportError)

from tunix_accel import gemma3_tiled_mlp


def _tiny_model(remat_config=None):
  if remat_config is None:
    remat_config = gemma3_model.RematConfig.NONE
  config = gemma3_model.ModelConfig(
      num_layers=1,
      num_embed=64,
      embed_dim=32,
      hidden_dim=64,
      num_heads=4,
      head_dim=8,
      num_kv_heads=1,
      sliding_window_size=8,
      remat_config=remat_config,
      param_dtype=jnp.float32,
  )
  return gemma3_model.Gemma3(config, rngs=nnx.Rngs(0))


def _randomize_mlp(model) -> None:
  keys = iter(jax.random.split(jax.random.key(123), 3 * len(model.layers)))
  for layer in model.layers:
    for name in ("gate_proj", "up_proj", "down_proj"):
      projection = getattr(layer.mlp, name)
      key = next(keys)
      scale = jnp.sqrt(jnp.asarray(projection.kernel.shape[0], dtype=jnp.float32))
      projection.kernel[...] = jax.random.normal(
          key,
          projection.kernel.shape,
          dtype=jnp.float32,
      ) / scale


def _batch():
  tokens = jnp.array(
      [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
      dtype=jnp.int32,
  )
  input_mask = jnp.ones_like(tokens, dtype=bool)
  positions = utils.build_positions_from_mask(input_mask)
  attention_mask = utils.make_causal_attn_mask(input_mask)
  return tokens, input_mask, positions, attention_mask


def test_gemma3_mlp_block_matches_original():
  model = _tiny_model()
  _randomize_mlp(model)
  x = jax.random.normal(jax.random.key(0), (2, 7, model.config.embed_dim))
  mlp = model.layers[0].mlp

  expected = mlp.block(x)
  with gemma3_tiled_mlp.installed(token_chunk=3):
    actual = mlp.block(x)

  assert jnp.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_gemma3_mlp_call_matches_original_with_block_remat():
  model = _tiny_model(remat_config=gemma3_model.RematConfig.BLOCK)
  _randomize_mlp(model)
  x = jax.random.normal(jax.random.key(1), (2, 7, model.config.embed_dim))
  mlp = model.layers[0].mlp

  expected = mlp(x)
  with gemma3_tiled_mlp.installed(token_chunk=4):
    actual = mlp(x)

  assert jnp.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_gemma3_default_loss_matches_with_tiled_mlp_patch():
  model = _tiny_model()
  _randomize_mlp(model)
  tokens, input_mask, positions, attention_mask = _batch()

  expected = peft_trainer._default_loss_fn(  # pylint: disable=protected-access
      model,
      tokens,
      input_mask,
      positions,
      attention_mask,
  )
  with gemma3_tiled_mlp.installed(token_chunk=3):
    actual = peft_trainer._default_loss_fn(  # pylint: disable=protected-access
        model,
        tokens,
        input_mask,
        positions,
        attention_mask,
    )

  assert jnp.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_gemma3_tiled_mlp_lora_fallback_and_strict_error():
  base_model = _tiny_model()
  provider = qwix.LoraProvider(
      module_path=".*gate_proj|.*up_proj|.*down_proj",
      rank=4,
      alpha=8,
  )
  model = qwix.apply_lora_to_model(
      base_model,
      provider,
      **base_model.get_model_input(),
      rngs=nnx.Rngs(1),
  )
  x = jax.random.normal(jax.random.key(2), (2, 7, model.config.embed_dim))
  expected = model.layers[0].mlp.block(x)

  with gemma3_tiled_mlp.installed(
      token_chunk=3,
      fallback_to_original_on_lora=True,
  ):
    actual = model.layers[0].mlp.block(x)
  assert jnp.allclose(actual, expected, atol=2e-5, rtol=2e-5)

  with gemma3_tiled_mlp.installed(
      token_chunk=3,
      fallback_to_original_on_lora=False,
  ):
    with pytest.raises(TypeError, match="Qwix-LoRA"):
      model.layers[0].mlp.block(x)
