#!/usr/bin/env python3
"""Local LoRA-FA mechanics tests."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ["TUNIX_ACCEL_DISABLE_AUTOPATCH"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp
import optax

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
sft_utils = _import_or_skip("tunix.sft.utils")

from tunix_accel import lora_fa


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


def _lora_model():
  base_model = _tiny_model()
  provider = qwix.LoraProvider(
      module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
      rank=4,
      alpha=8,
  )
  return qwix.apply_lora_to_model(
      base_model,
      provider,
      **base_model.get_model_input(),
      rngs=nnx.Rngs(1),
  )


def _batch():
  tokens = jnp.array(
      [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
      dtype=jnp.int32,
  )
  input_mask = jnp.ones_like(tokens, dtype=bool)
  positions = sft_utils.build_positions_from_mask(input_mask)
  attention_mask = sft_utils.make_causal_attn_mask(input_mask)
  return {
      "input_tokens": tokens,
      "input_mask": input_mask,
      "positions": positions,
      "attention_mask": attention_mask,
  }


def _loss(m):
  return peft_trainer._default_loss_fn(m, **_batch())  # pylint: disable=protected-access


def _flat_lora_values(model, suffix: str):
  return {
      path: value[...]
      for path, value in nnx.iter_graph(model)
      if isinstance(value, nnx.LoRAParam) and str(path[-1]).endswith(suffix)
  }


def test_lora_fa_filter_selects_only_b_params():
  model = _lora_model()
  grads = nnx.grad(_loss, argnums=lora_fa.lora_fa_diff_state())(model)

  paths = ["/".join(map(str, path)) for path, _ in nnx.to_flat_state(grads)]
  assert paths
  assert all(path.endswith("_lora_b") for path in paths)
  assert not any(path.endswith("_lora_a") for path in paths)


def test_correct_lora_fa_grads_matches_rank_space_formula():
  model = _lora_model()
  config = lora_fa.LoRAFAConfig(lora_alpha=8, correction_eps=1e-8)
  grads = nnx.grad(_loss, argnums=lora_fa.lora_fa_diff_state())(model)
  corrected = lora_fa.correct_lora_fa_grads(model, grads, config)
  lora_state = dict(nnx.to_flat_state(nnx.state(model, nnx.LoRAParam)))

  for path, value in nnx.to_flat_state(corrected):
    a_path = (*path[:-1], str(path[-1]).removesuffix("_lora_b") + "_lora_a")
    a = lora_state[a_path][...]
    original_b_grad = dict(nnx.to_flat_state(grads))[path][...]
    rank = a.shape[-1]
    a_flat = jnp.reshape(a, (-1, rank))
    gram = a_flat.T @ a_flat
    inv_gram = jnp.linalg.pinv(gram + config.correction_eps * jnp.eye(rank))
    scaling = config.lora_alpha / rank
    expected = inv_gram @ jnp.reshape(original_b_grad, (rank, -1))
    expected = expected / (scaling * scaling)
    expected = jnp.reshape(expected, original_b_grad.shape)
    assert jnp.allclose(value[...], expected, atol=1e-5, rtol=1e-5)


def test_lora_fa_trainer_updates_b_but_not_a():
  model = _lora_model()
  snapshot = lora_fa.lora_value_snapshot(model)
  before_a = _flat_lora_values(model, "_lora_a")
  before_b = _flat_lora_values(model, "_lora_b")
  config = peft_trainer.TrainingConfig(eval_every_n_steps=100, max_steps=1)

  with lora_fa.patched(lora_fa.LoRAFAConfig(mode="freeze_a", lora_alpha=8)):
    trainer = peft_trainer.PeftTrainer(model, optax.sgd(1e-3), config)
    assert trainer.optimizer.wrt is lora_fa.lora_fa_trainable_filter
    trainer._train_step(model, trainer.optimizer, _batch())  # pylint: disable=protected-access

  after_a = _flat_lora_values(model, "_lora_a")
  after_b = _flat_lora_values(model, "_lora_b")
  assert before_a.keys() == after_a.keys()
  assert before_b.keys() == after_b.keys()
  assert all(jnp.array_equal(before_a[path], after_a[path]) for path in before_a)
  assert any(not jnp.array_equal(before_b[path], after_b[path]) for path in before_b)
  delta_summary = lora_fa.lora_value_delta_summary(snapshot, model)
  assert delta_summary["lorafa_a_value_delta_max"] == 0.0
  assert delta_summary["lorafa_b_value_delta_max"] > 0.0
